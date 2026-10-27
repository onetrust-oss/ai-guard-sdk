import pytest

import ai_guard.otel as otel_module


@pytest.fixture(autouse=True)
def _reset_tracer_provider_warning():
    """Suppress `warn_if_no_tracer_provider` by default.

    The suite runs with `filterwarnings = ["error"]`, and no test process
    configures a real OTel `TracerProvider`, so every `client.activate(...)`
    call would otherwise trip this warning-turned-error the first time it
    runs -- order-dependently, since it only fires once per process. Tests
    that specifically exercise the warning reset the flag to `False`
    themselves.
    """
    otel_module._warned_no_tracer_provider = True
    yield
    otel_module._warned_no_tracer_provider = True
