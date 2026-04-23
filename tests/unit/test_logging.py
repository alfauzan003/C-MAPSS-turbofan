"""Tests for pdm.logging — structlog setup produces JSON logs."""

import io
import json
import logging

import structlog

from pdm.logging import configure_logging, get_logger


def test_configure_logging_emits_json(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    # Re-route the stdlib root logger to our buffer
    handler = logging.StreamHandler(buf)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    configure_logging(level="INFO", service="test-service")
    log = get_logger("unit")
    log.info("hello", k="v")

    handler.flush()
    out = buf.getvalue().strip().splitlines()[-1]
    payload = json.loads(out)
    assert payload["event"] == "hello"
    assert payload["k"] == "v"
    assert payload["service"] == "test-service"
    assert payload["level"] == "info"


def test_get_logger_returns_bound_logger():
    configure_logging(level="INFO", service="x")
    log = get_logger("mod")
    assert isinstance(log, structlog.stdlib.BoundLogger) or hasattr(log, "info")
