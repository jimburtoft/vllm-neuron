# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the near-max_model_len multi-token decode disable.

The decode NEFF processes ``1 + width`` tokens for each request in the batch,
where ``width`` comes from speculation or from jump-forward decoding. The model
runner pads a batch that holds drafts for only some requests up to
``num_reqs * (1 + width)``, so the attention kernel gets the same
``s_active = 1 + width`` for every request. Near ``max_model_len`` the active
K/V then overwrites prior tokens that the request still needs. These tests cover
the scheduler gate that stops the multi-token decode step for the whole batch.
"""

import re
from unittest.mock import Mock

import pytest

from test.vllm_neuron.vllm.core.conftest import (
    SchedulerMockKVCacheConfig,
    _make_mock_vllm_config,
)

# Must match vllm_neuron.vllm.core.scheduler._SPEC_DISABLED_ATTR. The test
# asserts on the mark, so it names the attribute rather than importing a private
# constant that a refactor could silently rename.
SPEC_DISABLED_ATTR = "_spec_disabled_near_max_model_len"


class _StubRequest:
    """Minimal stand-in for a vLLM Request.

    A plain object, so ``getattr(request, SPEC_DISABLED_ATTR, False)`` gives
    False before the scheduler sets the mark. A Mock gives a truthy
    auto-created attribute instead, which hides the sticky logic.
    """

    def __init__(
        self,
        request_id: str,
        num_computed_tokens: int,
        spec_token_ids: list[int] | None = None,
        num_prompt_tokens: int | None = None,
    ) -> None:
        self.request_id = request_id
        self.num_computed_tokens = num_computed_tokens
        self.spec_token_ids = [] if spec_token_ids is None else list(spec_token_ids)
        # Default to a decode request: num_computed >= num_prompt, so
        # _is_prefill_request() is False and schedule() takes the decode branch.
        self.num_prompt_tokens = (
            num_computed_tokens if num_prompt_tokens is None else num_prompt_tokens
        )
        self.num_tokens = self.num_computed_tokens + 1
        self.num_tokens_with_spec = self.num_tokens + len(self.spec_token_ids)
        self.num_output_placeholders = 0
        self.use_structured_output = False
        self.is_prefill_chunk = False
        self.max_tokens = 16
        self.num_preemptions = 0

    def is_finished(self) -> bool:
        return False


class _TripwireRequest:
    """Fail the test if anything reads ``num_computed_tokens``.

    Plain decode has width 0, so the gate must return before it looks at any
    request.
    """

    def __init__(self, request_id: str = "tripwire") -> None:
        self.request_id = request_id
        self.spec_token_ids = [7, 8, 9]

    @property
    def num_computed_tokens(self) -> int:
        raise AssertionError("plain decode must not read num_computed_tokens")


def _make_scheduler(
    max_model_len: int = 2048,
    num_spec_tokens: int = 0,
    max_jump_forward_tokens: int = 0,
    async_scheduling: bool = False,
):
    """Build a NeuronScheduler (or NeuronAsyncScheduler) with mocked config."""
    from vllm_neuron.vllm.core.scheduler import (
        NeuronAsyncScheduler,
        NeuronScheduler,
    )

    config = _make_mock_vllm_config()
    config.scheduler_config.max_num_seqs = 8
    config.scheduler_config.max_num_batched_tokens = max_model_len
    config.scheduler_config.max_num_scheduled_tokens = max_model_len
    config.scheduler_config.async_scheduling = async_scheduling
    config.model_config.max_model_len = max_model_len
    config.num_speculative_tokens = num_spec_tokens
    # NeuronConfig.__post_init__ zeroes max_jump_forward_tokens unless
    # structured outputs are on, so jump-forward needs both keys.
    config.additional_config = {
        "neuron_config": {
            "max_jump_forward_tokens": max_jump_forward_tokens,
            "enable_structured_outputs": max_jump_forward_tokens > 0,
        }
    }
    config.parallel_config = Mock(
        data_parallel_rank=0,
        data_parallel_index=0,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    config.observability_config = Mock(
        kv_cache_metrics=False,
        kv_cache_metrics_sample=1.0,
        enable_mfu_metrics=False,
    )

    scheduler_cls = NeuronAsyncScheduler if async_scheduling else NeuronScheduler
    scheduler = scheduler_cls(
        vllm_config=config,
        kv_cache_config=SchedulerMockKVCacheConfig(num_blocks=4096, block_size=32),
        structured_output_manager=Mock(),
        block_size=32,
    )
    scheduler.running = []
    return scheduler


def _preempt_through_neuron_scheduler(scheduler, request):
    """Call NeuronScheduler._preempt_request, stubbing only the base's body.

    The base ``_preempt_request`` needs live KV-manager and queue state. Stubbing
    it keeps this test on the override's own contribution — dropping the mark —
    while still going through the real method, so deleting the override fails
    these tests.
    """
    from vllm_neuron.vllm.core.scheduler import NeuronScheduler

    calls = []
    original = NeuronScheduler.__mro__[1]._preempt_request
    try:
        NeuronScheduler.__mro__[1]._preempt_request = lambda self, req, ts: (
            calls.append(req)
        )
        NeuronScheduler._preempt_request(scheduler, request, 0.0)
    finally:
        NeuronScheduler.__mro__[1]._preempt_request = original
    assert calls == [request], "the override must still delegate to the base"


class TestMultiTokenDecodeWidth:
    """The width must come from one rule that covers both features."""

    def test_speculation_gives_the_speculative_token_count(self):
        scheduler = _make_scheduler(num_spec_tokens=4)
        assert scheduler._multi_token_decode_width() == 4

    def test_jump_forward_gives_the_jump_forward_token_count(self):
        scheduler = _make_scheduler(max_jump_forward_tokens=3)
        assert scheduler._multi_token_decode_width() == 3

    def test_plain_decode_gives_zero(self):
        scheduler = _make_scheduler()
        assert scheduler._multi_token_decode_width() == 0

    def test_speculation_wins_over_jump_forward(self):
        """The platform zeroes JF when speculation is on. Mirror that order."""
        scheduler = _make_scheduler(num_spec_tokens=4, max_jump_forward_tokens=3)
        assert scheduler._multi_token_decode_width() == 4


class TestSafeNumComputed:
    """The threshold is the async rule, shared with the sync path."""

    @pytest.mark.parametrize(
        "max_model_len,width,expected",
        [
            (2048, 4, 2048 - 3 - 8),
            (4096, 1, 4096 - 3 - 2),
            (8192, 7, 8192 - 3 - 14),
        ],
    )
    def test_threshold_matches_the_documented_formula(
        self, max_model_len, width, expected
    ):
        scheduler = _make_scheduler(max_model_len=max_model_len, num_spec_tokens=width)
        assert scheduler._multi_token_decode_safe_num_computed() == expected

    @pytest.mark.parametrize(
        "max_model_len,width",
        [(2048, 4), (4096, 1), (8192, 7), (512, 3), (128, 1)],
    )
    def test_the_allowed_context_plus_a_full_step_fits_max_model_len(
        self, max_model_len, width
    ):
        """The core safety property, in ``max_model_len`` tokens.

        The model assigns the active K/V into the last ``s_active = 1 + width``
        slots of the KV window. The largest context the gate permits must leave
        room for all of them inside ``max_model_len``. This asserts the bound in
        ``max_model_len`` tokens, which is the window width when
        ``decode_context_length_buckets`` is unset. A sliding-window frame sizes
        its window differently and is out of scope.
        """
        scheduler = _make_scheduler(max_model_len=max_model_len, num_spec_tokens=width)
        largest_allowed = scheduler._multi_token_decode_safe_num_computed()
        assert largest_allowed + 1 + width <= max_model_len

    @pytest.mark.parametrize(
        "max_model_len,width",
        [(2048, 4), (4096, 1), (8192, 7), (512, 3)],
    )
    def test_the_threshold_is_stricter_than_the_no_trim_bound(
        self, max_model_len, width
    ):
        """The base scheduler trims above ``max_model_len - 2 - width``."""
        scheduler = _make_scheduler(max_model_len=max_model_len, num_spec_tokens=width)
        no_trim_bound = max_model_len - 2 - width
        assert scheduler._multi_token_decode_safe_num_computed() < no_trim_bound

    @staticmethod
    def _runner_spec_decode_margin() -> int:
        """Read the model runner's ``skip_propose`` margin out of its source.

        The runner computes ``spec_decode_limit`` inline, so there is no symbol
        to import. Reading the expression is what makes the next test pin BOTH
        sides: a transcribed copy of the margin would keep passing after someone
        changed the runner.

        Either failure below means the coupling needs re-checking by hand. A
        reformat of the line, an appended term, or a reshaped expression such as
        ``self.max_model_len - (num_spec_tokens + 2)`` all drop to 0 matches and
        trip the first assertion.

        The pattern pins the literal in the INITIAL assignment only, so a LATER
        statement that adjusts ``spec_decode_limit`` — the shape of the
        ``-= 1 + num_spec_tokens`` the runner already writes for the async path —
        would move the real limit without changing that literal. The second
        assertion covers exactly that. It walks the runner's AST and counts every
        binding of the name ``spec_decode_limit``, so it sees a tuple,
        parenthesised, annotated or walrus target and an assignment folded onto
        an ``if`` line, and it does NOT count a ``spec_decode_limit=`` keyword
        argument or a docstring that shows the formula.
        """
        import ast
        import inspect

        from vllm_neuron.vllm.worker import neuron_model_runner

        source = inspect.getsource(neuron_model_runner)
        # Anchored at both ends, so an APPENDED term is a miss rather than a
        # silent match on the first number. A trailing comment is still allowed.
        matches = re.findall(
            r"^\s*spec_decode_limit\s*=\s*self\.max_model_len\s*-\s*"
            r"num_spec_tokens\s*-\s*(\d+)\s*(?:#.*)?$",
            source,
            re.MULTILINE,
        )
        assert len(matches) == 1, (
            "expected exactly one spec_decode_limit assignment in "
            f"neuron_model_runner, found {len(matches)}"
        )
        bindings = 0
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                bindings += sum(
                    isinstance(sub, ast.Name) and sub.id == "spec_decode_limit"
                    for sub in ast.walk(target)
                )
        assert bindings == 2, (
            f"neuron_model_runner now binds spec_decode_limit {bindings} times, "
            "not 2. The pattern above pins only the initial assignment, so a new "
            "binding can move the runner's real limit without changing that "
            "literal. Re-check the coupling with "
            "_multi_token_decode_safe_num_computed."
        )
        return int(matches[0])

    def test_the_runner_margin_is_readable(self):
        """Guard the guard: the helper must find the runner's real margin."""
        assert self._runner_spec_decode_margin() == 2

    @pytest.mark.parametrize(
        "max_model_len,width",
        [(2048, 1), (2048, 4), (4096, 7), (8192, 4), (131072, 4)],
    )
    def test_the_threshold_agrees_with_the_model_runner_gate(
        self, max_model_len, width
    ):
        """Both gates must first fire at the same context.

        The model runner stops the draft proposal for the whole batch when
        ``positions.max() >= max_model_len - width - margin`` (``skip_propose``).
        A speculative step covers positions ``ctx`` to ``ctx + width``, so the
        runner first sees the limit at ``ctx == max_model_len - 2*width -
        margin``. This gate blocks the same context, so it makes an existing
        batch-wide policy pre-emptive instead of adding a second policy.

        The runner's margin is READ FROM ITS SOURCE, so a change to the margin
        VALUE on either side fails this test. An appended term, a reshaped
        expression, or a new BINDING of ``spec_decode_limit`` all trip an
        assertion in the helper instead. See its docstring for what the two
        assertions cover, and for what they do not.

        Two qualifiers on the equality. It holds for the sync path only: with
        ``use_async_scheduling`` the runner subtracts a further ``1 + width``,
        which puts its limit ``width + 1`` positions earlier — out of scope,
        because async speculation takes this gate's early return. And it assumes
        the request holds ``1 + width`` tokens. In a mixed batch the near-limit
        request holds one, so it contributes ``ctx`` rather than
        ``ctx + width``, and the runner's trigger for it lands ``width``
        positions later than this gate's. Those are the steps that violate.
        """
        scheduler = _make_scheduler(max_model_len=max_model_len, num_spec_tokens=width)
        first_blocked_ctx = scheduler._multi_token_decode_safe_num_computed() + 1
        runner_limit = max_model_len - width - self._runner_spec_decode_margin()
        assert first_blocked_ctx == runner_limit - width


class TestBlockedAndSticky:
    """The mark is per request and it never clears."""

    def test_no_block_when_every_request_is_far_from_the_limit(self):
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        requests = [
            _StubRequest("a", num_computed_tokens=100),
            _StubRequest("b", num_computed_tokens=200),
        ]
        assert scheduler._mark_and_check_multi_token_decode(requests) is False
        for request in requests:
            assert getattr(request, SPEC_DISABLED_ATTR, False) is False

    def test_the_threshold_itself_is_still_allowed(self):
        """Off-by-one guard: equality must not block."""
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        safe = scheduler._multi_token_decode_safe_num_computed()
        at_limit = _StubRequest("at_limit", num_computed_tokens=safe)

        assert scheduler._mark_and_check_multi_token_decode([at_limit]) is False
        assert getattr(at_limit, SPEC_DISABLED_ATTR, False) is False

    def test_only_the_crossing_request_gets_the_mark(self):
        """One request above the threshold blocks the step and takes the mark.

        A batch mate does lose its drafts on each step it shares with a marked
        request. It must not carry the mark itself, or it would keep losing them
        after the crossing request finishes.
        """
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        safe = scheduler._multi_token_decode_safe_num_computed()
        near = _StubRequest("near", num_computed_tokens=safe + 1)
        far = _StubRequest("far", num_computed_tokens=100)

        assert scheduler._mark_and_check_multi_token_decode([far, near]) is True
        assert getattr(near, SPEC_DISABLED_ATTR, False) is True
        assert getattr(far, SPEC_DISABLED_ATTR, False) is False

    def test_the_mark_stays_after_the_context_moves_back_down(self):
        """Sticky: a rejection lowers num_computed_tokens on the next step."""
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        safe = scheduler._multi_token_decode_safe_num_computed()
        request = _StubRequest("rejecting", num_computed_tokens=safe + 1)

        assert scheduler._mark_and_check_multi_token_decode([request]) is True

        # A rejection correction takes the context back below the threshold.
        request.num_computed_tokens = safe - 50
        assert scheduler._mark_and_check_multi_token_decode([request]) is True
        assert getattr(request, SPEC_DISABLED_ATTR, False) is True


class TestBatchScopedDraftClearing:
    """One request near the limit stops the drafts of the whole batch."""

    def test_a_batch_mate_loses_its_drafts(self):
        """The mixed batch is the reachable case.

        The near-limit request already holds no drafts. Its batch mate does, and
        the runner pads both up to ``1 + width``. Only clearing the batch mate
        brings ``s_active`` back to 1.
        """
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        safe = scheduler._multi_token_decode_safe_num_computed()
        near = _StubRequest("near", num_computed_tokens=safe + 1, spec_token_ids=[])
        mate = _StubRequest(
            "mate", num_computed_tokens=100, spec_token_ids=[11, 12, 13, 14]
        )
        scheduler.running = [near, mate]

        scheduler._block_multi_token_decode_near_max_model_len()

        assert mate.spec_token_ids == []
        assert near.spec_token_ids == []
        assert scheduler._multi_token_decode_blocked_step is True

    def test_drafts_survive_when_no_request_is_near_the_limit(self):
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        mate = _StubRequest(
            "mate", num_computed_tokens=100, spec_token_ids=[11, 12, 13, 14]
        )
        scheduler.running = [mate]

        scheduler._block_multi_token_decode_near_max_model_len()

        assert mate.spec_token_ids == [11, 12, 13, 14]
        assert scheduler._multi_token_decode_blocked_step is False

    def test_a_crossed_request_keeps_losing_its_drafts_on_later_steps(self):
        """Disabled, and it stays disabled — through the gate, over two steps.

        The worker can send fresh draft tokens again once the batch context dips
        below its own limit, so the sticky mark has to keep clearing them.
        """
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        safe = scheduler._multi_token_decode_safe_num_computed()
        request = _StubRequest(
            "crosser", num_computed_tokens=safe + 1, spec_token_ids=[1, 2, 3, 4]
        )
        scheduler.running = [request]

        # Step 1: crosses the threshold, drafts go.
        scheduler._block_multi_token_decode_near_max_model_len()
        assert request.spec_token_ids == []
        assert getattr(request, SPEC_DISABLED_ATTR, False) is True

        # Step 2: a rejection took the context back down, and the worker sent
        # new drafts. The mark must still clear them.
        request.num_computed_tokens = safe - 100
        request.spec_token_ids = [5, 6, 7, 8]
        scheduler._block_multi_token_decode_near_max_model_len()
        assert request.spec_token_ids == []
        assert scheduler._multi_token_decode_blocked_step is True

    def test_the_step_flag_resets_on_every_call(self):
        """A stale True would suppress jump-forward for the rest of the run."""
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        safe = scheduler._multi_token_decode_safe_num_computed()
        near = _StubRequest("near", num_computed_tokens=safe + 1)
        scheduler.running = [near]
        scheduler._block_multi_token_decode_near_max_model_len()
        assert scheduler._multi_token_decode_blocked_step is True

        # The near-limit request finished. A fresh batch must not stay blocked.
        scheduler.running = [_StubRequest("fresh", num_computed_tokens=10)]
        scheduler._block_multi_token_decode_near_max_model_len()
        assert scheduler._multi_token_decode_blocked_step is False


class TestScheduleWiresTheGate:
    """The call site in schedule() must be pinned, not only the helpers.

    Every other test in this file calls the helpers directly, so deleting the
    ``schedule()`` call would leave them all green while the whole change went
    inert. These tests drive ``schedule()`` itself.
    """

    @staticmethod
    def _drive_schedule(scheduler, observe):
        """Run schedule(), capturing state at the moment the base scheduler runs.

        Only the collaborators DOWNSTREAM of the gate are stubbed, so the gate
        and its call site run for real. ``observe`` is called from inside the
        stub that replaces ``_call_base_schedule``, which is what makes the
        ordering assertion possible.
        """
        from test.vllm_neuron.vllm.core.conftest import MockSchedulerOutput

        def _fake_base(throttle_prefills=False):
            observe()
            return MockSchedulerOutput()

        scheduler._call_base_schedule = _fake_base
        scheduler._apply_padding_and_log_stats = lambda output: output
        scheduler.get_grammar_bitmask = lambda output: None
        scheduler._update_state = lambda output: None
        return scheduler.schedule()

    def test_schedule_clears_the_batch_before_the_base_scheduler_runs(self):
        """The load-bearing ordering claim, asserted from inside the base call."""
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        safe = scheduler._multi_token_decode_safe_num_computed()
        near = _StubRequest("near", num_computed_tokens=safe + 1)
        mate = _StubRequest(
            "mate", num_computed_tokens=100, spec_token_ids=[11, 12, 13, 14]
        )
        scheduler.running = [near, mate]

        seen = {}
        self._drive_schedule(
            scheduler, lambda: seen.update(mate_spec=list(mate.spec_token_ids))
        )

        # Empty AT THE MOMENT the base scheduler reserved tokens, not merely
        # empty by the time schedule() returned.
        assert seen["mate_spec"] == []
        assert getattr(near, SPEC_DISABLED_ATTR, False) is True
        assert scheduler._multi_token_decode_blocked_step is True

    def test_schedule_leaves_a_far_batch_alone(self):
        """Control: schedule() must not clear drafts away from the limit."""
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        mate = _StubRequest(
            "mate", num_computed_tokens=100, spec_token_ids=[11, 12, 13, 14]
        )
        scheduler.running = [mate]

        seen = {}
        self._drive_schedule(
            scheduler, lambda: seen.update(mate_spec=list(mate.spec_token_ids))
        )

        assert seen["mate_spec"] == [11, 12, 13, 14]
        assert scheduler._multi_token_decode_blocked_step is False

    def test_schedule_is_inert_for_plain_decode(self):
        """No speculation and no jump-forward: the canary must survive.

        The request sits at the last position, which is where a gate that ignored
        the width would fire.
        """
        scheduler = _make_scheduler(max_model_len=2048)
        request = _StubRequest(
            "plain", num_computed_tokens=2047, spec_token_ids=[7, 8, 9]
        )
        scheduler.running = [request]

        seen = {}
        self._drive_schedule(
            scheduler, lambda: seen.update(spec=list(request.spec_token_ids))
        )

        assert seen["spec"] == [7, 8, 9]
        assert getattr(request, SPEC_DISABLED_ATTR, False) is False
        assert scheduler._multi_token_decode_blocked_step is False


class TestPreemptionClearsTheMark:
    """A restarted request must not keep stopping the batch.

    Preemption sets ``num_computed_tokens`` back to 0, so the mark records a
    context the request no longer has. One marked request stops the drafts of
    the whole batch, so a stale mark would cost every batch mate its speculation
    for the rest of the restarted request's life.
    """

    def test_preemption_drops_the_mark(self):
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        safe = scheduler._multi_token_decode_safe_num_computed()
        victim = _StubRequest("victim", num_computed_tokens=safe + 1)
        scheduler.running = [victim]
        scheduler._block_multi_token_decode_near_max_model_len()
        assert getattr(victim, SPEC_DISABLED_ATTR, False) is True

        _preempt_through_neuron_scheduler(scheduler, victim)

        assert getattr(victim, SPEC_DISABLED_ATTR, False) is False

    def test_a_restarted_request_stops_blocking_the_batch(self):
        """The consequence, end to end: the batch keeps its drafts afterwards."""
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        safe = scheduler._multi_token_decode_safe_num_computed()
        victim = _StubRequest("victim", num_computed_tokens=safe + 1)
        scheduler.running = [victim]
        scheduler._block_multi_token_decode_near_max_model_len()

        _preempt_through_neuron_scheduler(scheduler, victim)
        victim.num_computed_tokens = 5
        mate = _StubRequest(
            "mate", num_computed_tokens=100, spec_token_ids=[11, 12, 13, 14]
        )
        scheduler.running = [victim, mate]

        scheduler._block_multi_token_decode_near_max_model_len()

        assert mate.spec_token_ids == [11, 12, 13, 14]
        assert scheduler._multi_token_decode_blocked_step is False

    def test_preemption_is_harmless_for_an_unmarked_request(self):
        scheduler = _make_scheduler(max_model_len=2048, num_spec_tokens=4)
        request = _StubRequest("fresh", num_computed_tokens=10)

        _preempt_through_neuron_scheduler(scheduler, request)

        assert getattr(request, SPEC_DISABLED_ATTR, False) is False


class TestPlainDecodeIsUntouched:
    """Width 0: the gate must return before it observes any request.

    ``TestScheduleWiresTheGate::test_schedule_is_inert_for_plain_decode`` covers
    the visible half — no mark, no cleared list, the step flag False.
    """

    def test_the_gate_never_reads_a_request(self):
        scheduler = _make_scheduler(max_model_len=2048)
        assert (
            scheduler._mark_and_check_multi_token_decode([_TripwireRequest()]) is False
        )

    def test_the_tripwire_request_would_fail_a_gate_that_ran(self):
        """Guard the guard: the tripwire must really raise when it is read."""
        with pytest.raises(AssertionError):
            _ = _TripwireRequest().num_computed_tokens


class TestAsyncSchedulerKeepsItsOwnSpecDisable:
    """The async path applies the same rule in _update_after_schedule."""

    def test_the_schedule_gate_does_not_touch_async_spec_drafts(self):
        """The sync control is
        ``TestBatchScopedDraftClearing::test_a_batch_mate_loses_its_drafts``,
        which runs this same batch and gets the drafts cleared.
        """
        scheduler = _make_scheduler(
            max_model_len=2048, num_spec_tokens=4, async_scheduling=True
        )
        safe = scheduler._multi_token_decode_safe_num_computed()
        near = _StubRequest("near", num_computed_tokens=safe + 1)
        mate = _StubRequest(
            "mate", num_computed_tokens=100, spec_token_ids=[11, 12, 13, 14]
        )
        scheduler.running = [near, mate]

        scheduler._block_multi_token_decode_near_max_model_len()

        assert mate.spec_token_ids == [11, 12, 13, 14]
        assert getattr(near, SPEC_DISABLED_ATTR, False) is False
        assert scheduler._multi_token_decode_blocked_step is False

    def test_the_async_scheduler_still_blocks_jump_forward(self):
        """Jump-forward has no placeholder machinery, so the gate applies."""
        scheduler = _make_scheduler(
            max_model_len=2048, max_jump_forward_tokens=3, async_scheduling=True
        )
        safe = scheduler._multi_token_decode_safe_num_computed()
        near = _StubRequest("near", num_computed_tokens=safe + 1)
        scheduler.running = [near]

        scheduler._block_multi_token_decode_near_max_model_len()

        assert scheduler._multi_token_decode_blocked_step is True
        assert getattr(near, SPEC_DISABLED_ATTR, False) is True


class TestJumpForwardInjectionIsSuppressed:
    """A blocked step must inject no jump-forward tokens at all."""

    @staticmethod
    def _make_output(req_id: str):
        from test.vllm_neuron.vllm.core.conftest import MockSchedulerOutput

        return MockSchedulerOutput(
            num_scheduled_tokens={req_id: 1},
            total_num_scheduled_tokens=1,
        )

    @staticmethod
    def _arrange(scheduler):
        """Make injection succeed, so only the blocked flag can stop it.

        Without this the blocked arm would also be stopped by
        ``_jf_slots_available`` returning 0 against a mock KV manager, and the
        test would pass whether or not the suppression exists — it would be
        measuring the fixture, not the code.
        """
        request = _StubRequest("jf", num_computed_tokens=100)
        scheduler.requests = {"jf": request}
        scheduler._pending_jf_tokens = {"jf": [21, 22]}
        scheduler._jf_slots_available = lambda _request: 8
        scheduler.structured_output_manager = Mock()
        scheduler.structured_output_manager.should_advance.return_value = False
        return request

    def test_no_tokens_are_injected_when_the_step_is_blocked(self):
        scheduler = _make_scheduler(max_model_len=2048, max_jump_forward_tokens=3)
        self._arrange(scheduler)
        scheduler._multi_token_decode_blocked_step = True
        output = self._make_output("jf")

        scheduler._inject_jump_forward_tokens(output)

        assert output.num_scheduled_tokens == {"jf": 1}
        assert output.total_num_scheduled_tokens == 1
        assert not getattr(output, "_jf_decode_tokens", {})
        # Dropped, exactly like the existing "no headroom" path drops them.
        assert scheduler._pending_jf_tokens == {}

    def test_tokens_are_injected_when_the_step_is_not_blocked(self):
        """Control: identical arrangement, blocked flag is the only difference."""
        scheduler = _make_scheduler(max_model_len=2048, max_jump_forward_tokens=3)
        self._arrange(scheduler)
        scheduler._multi_token_decode_blocked_step = False
        output = self._make_output("jf")

        scheduler._inject_jump_forward_tokens(output)

        assert output._jf_decode_tokens == {"jf": [21, 22]}
        assert output.num_scheduled_tokens == {"jf": 3}
        assert output.total_num_scheduled_tokens == 3
