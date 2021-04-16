import copy
import logging
import os
from importlib import reload

import torch
from torch.utils.data import TensorDataset


def create_dataset(dataset, seq_length=1, target_length=1, device="cuda:0"):
    dataX, dataY = [], []
    for i in range(len(dataset) - seq_length - target_length + 1):
        dataX.append(
            torch.from_numpy(
                copy.deepcopy(dataset[i:(i + seq_length), 0])).reshape(-1, seq_length)
        )
        dataY.append(
            torch.from_numpy(
                copy.deepcopy(dataset[(i + seq_length):(i + seq_length + target_length), 0])).reshape(-1, target_length)
        )

    dataX = torch.cat(dataX) if len(dataX) > 0 else torch.tensor([])
    dataY = torch.cat(dataY) if len(dataY) > 0 else torch.tensor([])
    return dataX.to(device), dataY.to(device)


def create_tensor_dataset(trainX, trainY):
    return TensorDataset(trainX, trainY)


def update_flat_dicts(root: dict, targets: list):
    my_root = copy.deepcopy(root)
    my_targets = [copy.deepcopy(target) for target in targets]

    for k, v in my_root.items():
        for target in my_targets:
            if k in target:
                target[k] = v

    return my_targets


def create_dirs(path: str):
    try:
        os.makedirs(path)
    except BaseException as exc:
        logging.debug("Could not create dir", exc_info=exc)


def init_logging(logging_level: str):
    """Initializes logging with specific settings.
    Parameters
    ----------
    logging_level : str
        The desired logging level
    """

    reload(logging)
    logging.basicConfig(
        level=getattr(logging, logging_level.upper()),
        format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s](%(name)s)__[L%(lineno)d] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    logging.info(f"Successfully initialized logging.")
