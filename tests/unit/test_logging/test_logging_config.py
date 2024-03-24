import logging
import sys
import time
from importlib.util import find_spec
from logging.handlers import QueueHandler
from queue import Queue
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional
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
from litestar.logging.standard import LoggingQueueListener
from litestar.logging.standard import QueueListenerHandler as StandardQueueListenerHandler
from litestar.status_codes import HTTP_200_OK
from litestar.testing import create_test_client
from tests.helpers import cleanup_logging_impl, getHandlerByName

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture


@pytest.fixture(autouse=True)
def cleanup_logging() -> Generator:
    with cleanup_logging_impl():
        yield


def test__get_default_logging_module() -> None:
    assert find_spec("picologging")  # picologging should be installed in the test environment, simply checking that
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


@pytest.mark.parametrize(
    "picologging_exists, expected_handlers",
    [
        [True, default_picologging_handlers],
        [False, default_handlers],
    ],
)
def test_correct_default_handlers_set(picologging_exists: bool, expected_handlers: Any) -> None:
    with patch("litestar.logging.config.find_spec") as find_spec_mock:
        find_spec_mock.return_value = picologging_exists
        log_config = LoggingConfig()
        assert log_config.handlers == expected_handlers


@pytest.mark.parametrize(
    "logging_module, expected_handlers",
    [
        ["logging", default_handlers],
        ["picologging", default_picologging_handlers],
    ],
)
def test_correct_default_handlers_set_logging_module(logging_module: str, expected_handlers: Any) -> None:
    log_config = LoggingConfig(logging_module=logging_module)
    assert log_config.handlers == expected_handlers


@pytest.mark.parametrize(
    "logging_module, dict_config_not_called",
    [
        ["logging", "picologging.config.dictConfig"],
        ["picologging", "logging.config.dictConfig"],
    ],
)
def test_dictconfig_on_startup(logging_module: str, dict_config_not_called: str) -> None:
    with patch(f"{logging_module}.config.dictConfig") as dict_config_mock:
        with patch(dict_config_not_called) as dict_config_not_called_mock:
            test_logger = LoggingConfig(logging_module=logging_module)
            with create_test_client([], on_startup=[test_logger.configure], logging_config=None):
                assert dict_config_mock.called
                assert dict_config_mock.call_count == 1
                assert dict_config_not_called_mock.call_count == 0


@pytest.mark.parametrize(
    "logging_module, expected_handler_class, expected_listener_class",
    [
        [
            logging,
            QueueHandler if sys.version_info >= (3, 12, 0) else StandardQueueListenerHandler,
            LoggingQueueListener,
        ],
        [
            picologging,
            PicologgingQueueListenerHandler,
            picologging.handlers.QueueListener,
        ],
    ],
)
def test_default_queue_listener_handler(
    logging_module: ModuleType, expected_handler_class: Any, expected_listener_class: Any, capsys: "CaptureFixture[str]"
) -> None:
    def wait_log_queue(queue: Any, sleep_time: float = 0.1, max_retries: int = 5) -> None:
        retry = 0
        while queue.qsize() > 0 and retry < max_retries:
            retry += 1
            time.sleep(sleep_time)

    def assert_log(queue: Any, expected: str, count: Optional[int] = None) -> None:
        wait_log_queue(queue)
        log_output = capsys.readouterr().err.strip()
        if count is not None:
            assert len(log_output.split("\n")) == count
        assert log_output == expected

    get_logger = LoggingConfig(
        logging_module=logging_module.__name__,
        formatters={"standard": {"format": "%(levelname)s :: %(name)s :: %(message)s"}},
        loggers={
            "test_logger": {
                "level": "INFO",
                "handlers": ["queue_listener"],
                "propagate": False,
            },
        },
    ).configure()

    logger = get_logger("test_logger")
    assert type(logger) is logging_module.Logger

    handler = logger.handlers[0]
    assert type(handler) is expected_handler_class
    assert type(handler.queue) is Queue

    assert type(handler.listener) is expected_listener_class
    assert type(handler.listener.handlers[0]) is logging_module.StreamHandler

    logger.info("Testing now!")
    assert_log(handler.queue, expected="INFO :: test_logger :: Testing now!", count=1)

    var = "test_var"
    logger.info("%s", var)
    assert_log(handler.queue, expected="INFO :: test_logger :: test_var", count=1)


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
def test_default_loggers(logging_module: ModuleType, expected_handler_class: Any) -> None:
    with create_test_client(logging_config=LoggingConfig(logging_module=logging_module.__name__)) as client:
        root_logger = client.app.get_logger()
        assert isinstance(root_logger, logging_module.Logger)
        assert root_logger.name == "root"
        assert type(root_logger.handlers[0]) is expected_handler_class

        litestar_logger = client.app.logger
        assert type(litestar_logger) is logging_module.Logger
        assert litestar_logger.name == "litestar"
        assert type(litestar_logger.handlers[0]) is expected_handler_class

        # same handler instance
        assert root_logger.handlers[0] is litestar_logger.handlers[0]


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
        return {"isinstance": isinstance(request.logger.handlers[0], expected_handler_class)}  # type: ignore[attr-defined]

    with create_test_client(
        route_handlers=[handler],
        logging_config=LoggingConfig(logging_module=logging_module),
    ) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.json()["isinstance"]


@pytest.mark.parametrize("logging_module", [logging, picologging])
def test_validation(logging_module: ModuleType) -> None:
    logging_config = LoggingConfig(
        logging_module=logging_module.__name__,
        formatters={},
        handlers={},
        loggers={},
    )
    assert logging_config.formatters["standard"]
    assert logging_config.handlers["queue_listener"] == _get_default_handlers(logging_module.__name__)["queue_listener"]
    assert logging_config.loggers["litestar"]


def test_validation_no_logging_module() -> None:
    logging_config = LoggingConfig(
        formatters={},
        handlers={},
        loggers={},
    )
    assert logging_config.logging_module == "picologging"
    assert logging_config.formatters["standard"]
    assert logging_config.handlers["queue_listener"] == _get_default_handlers("picologging")["queue_listener"]
    assert logging_config.loggers["litestar"]


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
    assert root_logger.name == "root"
    assert isinstance(root_logger.handlers[0], expected_handler_class)


@pytest.mark.parametrize("logging_module", [logging, picologging])
def test_root_logger_no_config(logging_module: ModuleType) -> None:
    logging_config = LoggingConfig(logging_module=logging_module.__name__, configure_root_logger=False)
    get_logger = logging_config.configure()
    root_logger = get_logger()

    assert isinstance(root_logger, logging_module.Logger)

    handlers = root_logger.handlers
    if logging_module == logging:
        # pytest automatically configures some handlers
        for handler in handlers:
            assert isinstance(handler, (_LiveLoggingNullHandler, LogCaptureHandler))
    else:
        assert len(handlers) == 0


@pytest.mark.parametrize(
    "logging_module, configure_root_logger, expected_root_logger_handler_class",
    [
        [logging, True, QueueHandler if sys.version_info >= (3, 12, 0) else StandardQueueListenerHandler],
        [logging, False, None],
        [picologging, True, PicologgingQueueListenerHandler],
        [picologging, False, None],
    ],
)
def test_customizing_handler(
    logging_module: ModuleType,
    configure_root_logger: bool,
    expected_root_logger_handler_class: Any,
    capsys: "CaptureFixture[str]",
) -> None:
    log_format = "%(levelname)s :: %(name)s :: %(message)s"

    logging_config = LoggingConfig(
        logging_module=logging_module.__name__,
        formatters={
            "standard": {"format": log_format},
        },
        handlers={
            "console_stdout": {
                "class": f"{logging_module.__name__}.StreamHandler",
                "stream": "ext://sys.stdout",
                "level": "DEBUG",
                "formatter": "standard",
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

    # picologging seems to be broken, cannot make it log on stdout?
    # https://github.com/microsoft/picologging/issues/205
    if logging_module == picologging:
        del logging_config.handlers["console_stdout"]["stream"]

    get_logger = logging_config.configure()
    root_logger = get_logger()

    if configure_root_logger is True:
        assert isinstance(root_logger, logging_module.Logger)
        assert root_logger.level == logging_module.INFO

        root_logger_handler = root_logger.handlers[0]  # type: ignore[attr-defined]
        assert root_logger_handler.name == "queue_listener"
        assert type(root_logger_handler) is expected_root_logger_handler_class

        if type(root_logger_handler) is QueueHandler:
            assert getHandlerByName("console") is not None
            formatter = root_logger_handler.listener.handlers[0].formatter  # type: ignore[attr-defined]
        else:
            assert getHandlerByName("console") is None
            formatter = root_logger_handler.formatter
        assert formatter._fmt == log_format
    else:
        # queue_listener handler shouldn't be configured (we only checks the `logging` module)
        assert getHandlerByName("queue_listener") is None

        # Root logger shouldn't be configured but pytest adds some handlers (for the standard `logging` module)
        for handler in root_logger.handlers:  # type: ignore[attr-defined]
            assert isinstance(handler, (_LiveLoggingNullHandler, LogCaptureHandler))

    def assert_logger(logger: Any) -> None:
        assert type(logger) is logging_module.Logger
        assert logger.level == logging_module.DEBUG
        assert len(logger.handlers) == 1
        assert type(logger.handlers[0]) is logging_module.StreamHandler
        assert logger.handlers[0].name == "console_stdout"
        assert logger.handlers[0].formatter._fmt == log_format

        logger.info("Hello from '%s'", logging_module.__name__)
        if logging_module == picologging:
            log_output = capsys.readouterr().err.strip()
        else:
            log_output = capsys.readouterr().out.strip()
        assert log_output == f"INFO :: {logger.name} :: Hello from '{logging_module.__name__}'"

    assert_logger(get_logger("litestar"))
    assert_logger(get_logger("test_logger"))
