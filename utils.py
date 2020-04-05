import os
import logging
from torch import load, save


def save_model(model, path):
    save(model.state_dict(), path)


def load_model(model, path):
    state_dict = load(path)
    model.load_state_dict(state_dict)


def setup_logger(logger, level=logging.DEBUG, filename=None):
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
