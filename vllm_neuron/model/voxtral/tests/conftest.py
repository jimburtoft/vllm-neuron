# SPDX-License-Identifier: Apache-2.0
"""Local pytest config for the Voxtral model tests.

Registers the ``neuron_device`` marker so the device-gated correctness test
collects without a warning even when pytest is invoked directly on this
directory (outside the repo-root pyproject.toml context).
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "neuron_device: test requires a Neuron device and model weights; "
        "skipped unless VOXTRAL_NEURON_DEVICE_TESTS=1",
    )
