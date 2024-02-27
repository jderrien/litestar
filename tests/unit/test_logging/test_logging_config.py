import logging
import sys
import time
from importlib.util import find_spec
from logging.handlers import QueueHandler
from queue import Queue
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, Optional
from unittest.mock import patch

import picologging
import pytest
from _pytest.logging import LogCaptureHandler, _LiveLoggingNullHandler

from litestar import Request, get
from litestar.exceptions import ImproperlyConfiguredException
from litestar.logging.config import (
    LoggingConfig,
    _get_default_handlers,
    _get_default_logging_module,
    default_handlers,
    default_picologging_handlers,
)
from litestar.logging.picologging import QueueListenerHandler as PicologgingQueueListenerHandler
from litestar.status_codes import HTTP_200_OK
from litestar.testing import create_test_client
from tests.helpers import cleanup_logging_impl

if sys.version_info >= (3, 12, 0):
    from litestar.logging.standard import LoggingQueueListener
else:
    from litestar.logging.standard import QueueListenerHandler as StandardQueueListenerHandler

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture


@pytest.fixture(autouse=True)
def cleanup_logging() -> None:
    with cleanup_logging_impl():
        pass


def test__get_default_logging_module() -> None:
    assert find_spec("picologging")
    assert _get_default_logging_module() == "picologging"
    with patch("litestar.logging.config.find_spec") as find_spec_mock:
        find_spec_mock.return_value = None
        assert _get_default_logging_module() == "logging"


def test__get_default_handlers() -> None:
    assert _get_default_handlers(logging_module="logging") == default_handlers
    assert _get_default_handlers(logging_module="picologging") == default_picologging_handlers


@pytest.mark.parametrize(
    "logging_module, dict_config_callable, expected_called, expected_default_handlers",
    [
        ["logging", "logging.config.dictConfig", True, default_handlers],
        ["logging", "picologging.config.dictConfig", False, default_handlers],
        ["picologging", "picologging.config.dictConfig", True, default_picologging_handlers],
        ["picologging", "logging.config.dictConfig", False, default_picologging_handlers],
    ],
)
def test_correct_dict_config_called(
    logging_module: str,
    dict_config_callable: str,
    expected_called: bool,
    expected_default_handlers: Dict[str, Dict[str, Any]],
) -> None:
    with patch(dict_config_callable) as dict_config_mock:
        log_config = LoggingConfig(logging_module=logging_module)
        log_config.configure()
        if expected_called:
            assert dict_config_mock.called
        else:
            assert not dict_config_mock.called
        assert log_config.handlers == expected_default_handlers


@pytest.mark.parametrize("picologging_exists", [True, False])
def test_correct_default_handlers_set(picologging_exists: bool) -> None:
    with patch("litestar.logging.config.find_spec") as find_spec_mock:
        find_spec_mock.return_value = picologging_exists
        log_config = LoggingConfig()

        if picologging_exists:
            assert log_config.handlers == default_picologging_handlers
        else:
            assert log_config.handlers == default_handlers


@pytest.mark.parametrize(
    "logging_module, expected_handler_class",
    [
        [logging, QueueHandler if sys.version_info >= (3, 12, 0) else StandardQueueListenerHandler],
        [picologging, PicologgingQueueListenerHandler],
    ],
)
def test_default_queue_listener_handler(
    logging_module: ModuleType, expected_handler_class: Any, capsys: "CaptureFixture[str]"
) -> None:
    def wait_log_queue(queue: Queue, sleep_time: float = 0.1, max_retries: int = 5) -> None:
        retry = 0
        while queue.qsize() > 0 and retry < max_retries:
            retry += 1
            time.sleep(sleep_time)

    def assert_log(queue: Queue, text: str, count: Optional[int] = None) -> None:
        wait_log_queue(queue)
        log_output = capsys.readouterr().err.strip()
        if count is not None:
            assert len(log_output.split("\n")) == count
        assert text in log_output

    get_logger = LoggingConfig(
        logging_module=logging_module.__name__,
        loggers={
            "test_logger": {
                "level": "INFO",
                "handlers": ["queue_listener"],
                "propagate": False,
            },
        },
    ).configure()

    logger = get_logger("test_logger")
    assert isinstance(logger, logging_module.Logger)

    handler = logger.handlers[0]  # type: ignore
    assert isinstance(handler, expected_handler_class)

    logger.info("Testing now!")
    assert_log(handler.queue, text="Testing now!", count=1)

    var = "test_var"
    logger.info("%s", var)
    assert_log(handler.queue, text=var, count=1)


def test_get_logger_without_logging_config() -> None:
    with create_test_client(logging_config=None) as client:
        with pytest.raises(
            ImproperlyConfiguredException,
            match="cannot call '.get_logger' without passing 'logging_config' to the Litestar constructor first",
        ):
            client.app.get_logger()


@pytest.mark.parametrize(
    "logging_module, expected_handler_class",
    [
        [logging, QueueHandler if sys.version_info >= (3, 12, 0) else StandardQueueListenerHandler],
        [picologging, PicologgingQueueListenerHandler],
    ],
)
def test_get_default_loggers(logging_module: ModuleType, expected_handler_class: Any) -> None:
    with create_test_client(logging_config=LoggingConfig(logging_module=logging_module.__name__)) as client:
        root_logger = client.app.get_logger()
        assert isinstance(root_logger, logging_module.Logger)
        assert root_logger.name == "root"
        assert isinstance(root_logger.handlers[0], expected_handler_class)

        litestar_logger = client.app.logger
        assert isinstance(litestar_logger, logging_module.Logger)
        assert litestar_logger.name == "litestar"
        assert isinstance(litestar_logger.handlers[0], expected_handler_class)

        handler = litestar_logger.handlers[0]
        assert isinstance(handler.queue, Queue)

        if logging_module == logging:
            if sys.version_info >= (3, 12, 0):
                expected_listener_class = LoggingQueueListener
            else:
                expected_listener_class = logging.handlers.QueueListener
            assert isinstance(handler.listener, expected_listener_class)
            assert isinstance(handler.listener.handlers[0], logging.StreamHandler)
        else:
            assert isinstance(handler.listener, picologging.handlers.QueueListener)
            assert isinstance(handler.listener.handlers[0], picologging.StreamHandler)


@pytest.mark.parametrize(
    "logging_module, expected_handler_class",
    [
        ["logging", QueueHandler if sys.version_info >= (3, 12, 0) else StandardQueueListenerHandler],
        ["picologging", PicologgingQueueListenerHandler],
    ],
)
def test_connection_logger(logging_module: str, expected_handler_class: Any) -> None:
    @get("/")
    def handler(request: Request) -> Dict[str, bool]:
        return {"isinstance": isinstance(request.logger.handlers[0], expected_handler_class)}  # type: ignore

    with create_test_client(
        route_handlers=[handler],
        logging_config=LoggingConfig(logging_module=logging_module),
    ) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.json()["isinstance"]


@pytest.mark.parametrize("logging_module", [logging, picologging])
def test_validation(logging_module) -> None:
    logging_config = LoggingConfig(logging_module=logging_module.__name__, handlers={}, loggers={})
    assert logging_config.handlers["queue_listener"] == _get_default_handlers(logging_module.__name__)["queue_listener"]
    assert "litestar" in logging_config.loggers


@pytest.mark.parametrize(
    "logging_module, expected_handler_class",
    [
        [logging, QueueHandler if sys.version_info >= (3, 12, 0) else StandardQueueListenerHandler],
        [picologging, PicologgingQueueListenerHandler],
    ],
)
def test_root_logger(logging_module: ModuleType, expected_handler_class: Any) -> None:
    logging_config = LoggingConfig(logging_module=logging_module.__name__)
    get_logger = logging_config.configure()
    root_logger = get_logger()
    assert isinstance(root_logger, logging_module.Logger)
    assert root_logger.name == "root"  # type: ignore[attr-defined]
    assert isinstance(root_logger.handlers[0], expected_handler_class)  # type: ignore[attr-defined]


@pytest.mark.parametrize("logging_module", [logging, picologging])
def test_root_logger_no_config(logging_module: ModuleType) -> None:
    logging_config = LoggingConfig(logging_module=logging_module.__name__, configure_root_logger=False)
    get_logger = logging_config.configure()
    root_logger = get_logger()

    assert isinstance(root_logger, logging_module.Logger)

    handlers = root_logger.handlers  # type: ignore[attr-defined]
    if logging_module == logging:
        # pytest automatically configures some handlers
        for handler in handlers:
            assert isinstance(handler, (_LiveLoggingNullHandler, LogCaptureHandler))
    else:
        assert len(handlers) == 0


@pytest.mark.parametrize(
    "logging_module, configure_root_logger",
    [
        [logging, True],
        [logging, False],
        [picologging, True],
        [picologging, False],
    ],
)
def test_customizing_handler(
    logging_module: ModuleType, configure_root_logger: bool, capsys: "CaptureFixture[str]"
) -> None:
    log_format = "%(levelname)s :: %(name)s :: %(message)s"

    logging_config = LoggingConfig(
        logging_module=logging_module.__name__,
        formatters={
            "test_format": {"format": log_format},
        },
        handlers={
            "console_stdout": {
                "class": f"{logging_module.__name__}.StreamHandler",
                "stream": "ext://sys.stdout",
                "level": "DEBUG",
                "formatter": "test_format",
            },
        },
        loggers={
            "test_logger": {
                "level": "DEBUG",
                "handlers": ["console_stdout"],
                "propagate": False,
            },
            "litestar": {
                "level": "DEBUG",
                "handlers": ["console_stdout"],
                "propagate": False,
            },
        },
        configure_root_logger=configure_root_logger,
    )

    # picologging seems to be broken: https://github.com/microsoft/picologging/issues/205
    if logging_module == picologging:
        del logging_config.handlers["console_stdout"]["stream"]

    get_logger = logging_config.configure()

    root_logger = get_logger()
    if configure_root_logger is True:
        handlers_names = [handler.name for handler in root_logger.handlers]
        assert "queue_listener" in handlers_names
    else:
        # Root logger shouldn't be configured but pytest adds some handlers
        for handler in root_logger.handlers:
            assert isinstance(handler, (_LiveLoggingNullHandler, LogCaptureHandler))

    def assert_logger(logger):
        assert isinstance(logger, logging_module.Logger)
        assert logger.level == logging_module.DEBUG
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging_module.StreamHandler)  # type: ignore
        assert logger.handlers[0].name == "console_stdout"
        assert logger.handlers[0].formatter._fmt == log_format

        logger.info("Hello from '%s'", logging_module.__name__)
        if logging_module == picologging:
            log_output = capsys.readouterr().err.strip()
        else:
            log_output = capsys.readouterr().out.strip()
        assert log_output == f"INFO :: {logger.name} :: Hello from '{logging_module.__name__}'"

    assert_logger(get_logger("test_logger"))  # type: ignore
    assert_logger(get_logger("litestar"))  # type: ignore
