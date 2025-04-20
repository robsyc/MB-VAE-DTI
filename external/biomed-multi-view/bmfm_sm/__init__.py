"""
Module provides all the functionality for the BMFM Small Molecules repository.
It includes functionality to load and process datasets, pretain and finetune models, and use these models through an API.
"""

# ruff: noqa

import logging
import os
import socket
import threading
import time

import sklearn  # Needed for fast-transformers issue in with MPS
import wrapt

logger = logging.getLogger(__name__)

node_rank = os.environ.get("NODE_RANK", "0")
local_rank = os.environ.get("LOCAL_RANK", "0")
hostname, thread = socket.gethostname(), str(threading.get_ident())

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(custom_attribute)s - %(message)s",
    level=logging.INFO,
)
root_logger = logging.getLogger()

old_factory = logging.getLogRecordFactory()


def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.custom_attribute = ":".join((hostname, thread, node_rank, local_rank))
    return record


logging.setLogRecordFactory(record_factory)


def log_execution_time(wrapped):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        start_time = time.time()
        result = wrapped(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info("Execution time: %s seconds", execution_time)
        return result

    return wrapper(wrapped)
