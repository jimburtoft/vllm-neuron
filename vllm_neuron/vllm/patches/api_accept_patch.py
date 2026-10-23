# SPDX-License-Identifier: Apache-2.0
"""Accept each HTTP connection on the least loaded API server worker.

With more than one API server process (``--api-server-count > 1``, the default
with internal data parallelism), vLLM up to 0.31 has every worker accept from
one shared listening socket. Whichever worker wakes first takes a burst of
connections, and a client that keeps its connections alive (most benchmark
clients and HTTP pools) then stays on that worker. One API process can end up
with most of the traffic and become CPU bound while the engines are under-fed.

This backports the upstream fix (vllm-project/vllm#60723, in vLLM 0.32): each
worker publishes its number of open connections in a shared array, and a
worker that has more connections than another one leaves a pending connection
to the less loaded worker for up to 5 ms before it accepts it itself.

Applied only when several API servers share one socket. Set
``VLLM_NEURON_API_ACCEPT_LEAST_LOADED=0`` to disable it. Remove this patch when
the plugin's vLLM version includes #60723.
"""

import asyncio
import logging
import multiprocessing
import os
import socket

logger = logging.getLogger(__name__)

_ARGS_ATTR = "_neuron_api_server_loads"
_ORIG_ATTR = "_neuron_orig_run_api_server_worker_proc"
_ACCEPT_DEFER_CHECKS = 5
_NOT_ACCEPTING = 2**31 - 1

_applied = False
_original_api_server_manager_init = None


def _enabled() -> bool:
    return os.environ.get("VLLM_NEURON_API_ACCEPT_LEAST_LOADED", "1") != "0"


def _api_server_manager_init(self, listen_address, sock, args, num_servers, *a, **k):
    if num_servers > 1 and _enabled() and getattr(args, _ARGS_ATTR, None) is None:
        loads = multiprocessing.get_context("spawn").RawArray(
            "i", [_NOT_ACCEPTING] * num_servers
        )
        setattr(args, _ARGS_ATTR, loads)
        logger.info(
            "API servers: least-loaded connection accept on (%d workers)", num_servers
        )
    _original_api_server_manager_init(
        self, listen_address, sock, args, num_servers, *a, **k
    )


def _run_api_server_worker_proc(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    from vllm.v1 import utils as vllm_utils

    loads = getattr(args, _ARGS_ATTR, None)
    if loads is not None:
        index = int((client_config or {}).get("client_index", 0))
        _install_balanced_server(loads, index)
    original = getattr(vllm_utils, _ORIG_ATTR, None) or vllm_utils.run_api_server_worker_proc
    if original is _run_api_server_worker_proc:  # pragma: no cover - defensive
        raise RuntimeError("api accept patch: original worker entry point not found")
    original(listen_address, sock, args, client_config, **uvicorn_kwargs)


def _install_balanced_server(loads, index: int) -> None:
    from vllm.entrypoints.launchers import launcher

    base = launcher.NoSignalServer
    if getattr(base, "_neuron_least_loaded", False):
        return

    class LeastLoadedServer(base):
        _neuron_least_loaded = True

        def __init__(self, config, *a, **k):
            super().__init__(config, *a, **k)
            self.peer_loads = (loads, index)
            self._accept_tasks = set()
            self._connecting = 0

        def _publish_load(self) -> None:
            arr, i = self.peer_loads
            arr[i] = len(self.server_state.connections) + self._connecting

        def _least_loaded(self) -> bool:
            arr, i = self.peer_loads
            return arr[i] <= min(arr)

        async def startup(self, sockets=None):
            if not sockets:
                return await super().startup(sockets)
            await super().startup(sockets=[])
            if not self.started:
                return
            self._publish_load()
            for s in sockets:
                s.listen(self.config.backlog)
                s.setblocking(False)
                self._track(self._accept(s))

        def _track(self, coro) -> None:
            task = asyncio.create_task(coro)
            self._accept_tasks.add(task)
            task.add_done_callback(self._accept_tasks.discard)

        async def _accept(self, s: socket.socket) -> None:
            loop = asyncio.get_running_loop()
            while True:
                await _readable(loop, s)
                self._publish_load()
                for _ in range(_ACCEPT_DEFER_CHECKS):
                    if self._least_loaded():
                        break
                    await asyncio.sleep(0.001)
                try:
                    conn, _ = s.accept()
                except (BlockingIOError, InterruptedError, ConnectionAbortedError):
                    continue
                except OSError:
                    logger.exception("Error accepting a connection")
                    await asyncio.sleep(1)
                    continue
                self._connecting += 1
                self._publish_load()
                self._track(self._connect(loop, conn))

        async def _connect(self, loop, conn: socket.socket) -> None:
            config = self.config
            try:
                await loop.connect_accepted_socket(
                    lambda: config.http_protocol_class(
                        config=config,
                        server_state=self.server_state,
                        app_state=self.lifespan.state,
                        _loop=loop,
                    ),
                    conn,
                    ssl=config.ssl,
                )
            except asyncio.CancelledError:
                conn.close()
                raise
            except Exception:
                logger.debug("Accepted connection failed to start", exc_info=True)
                conn.close()
            finally:
                self._connecting -= 1
                self._publish_load()

        async def shutdown(self, sockets=None):
            tasks = list(self._accept_tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            arr, i = self.peer_loads
            arr[i] = _NOT_ACCEPTING
            await super().shutdown(sockets)

    launcher.NoSignalServer = LeastLoadedServer
    logger.info("API server %d: least-loaded connection accept installed", index)


async def _readable(loop: asyncio.AbstractEventLoop, s: socket.socket) -> None:
    ready = loop.create_future()

    def wake() -> None:
        if not ready.done():
            ready.set_result(None)

    loop.add_reader(s, wake)
    try:
        await ready
    finally:
        loop.remove_reader(s)


def apply_api_accept_patch() -> None:
    """Install the least-loaded accept. Call after the DP load-balancer patch."""
    global _applied, _original_api_server_manager_init
    if _applied or not _enabled():
        return
    try:
        from vllm.entrypoints.launchers import launcher
        from vllm.v1 import utils as vllm_utils
    except ImportError:
        logger.debug("api accept patch: vLLM launcher not found; skipped")
        return
    if "peer_loads" in getattr(launcher.serve_http, "__code__", type("", (), {"co_varnames": ()})).co_varnames:
        logger.debug("api accept patch: vLLM already balances accepts; skipped")
        _applied = True
        return
    _original_api_server_manager_init = vllm_utils.APIServerProcessManager.__init__
    vllm_utils.APIServerProcessManager.__init__ = _api_server_manager_init
    if not hasattr(vllm_utils, _ORIG_ATTR):
        setattr(vllm_utils, _ORIG_ATTR, vllm_utils.run_api_server_worker_proc)
    vllm_utils.run_api_server_worker_proc = _run_api_server_worker_proc
    _applied = True
