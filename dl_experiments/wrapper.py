import logging
from typing import Union, List, Any, Mapping

import torch
from ignite.engine import create_supervised_trainer
from torch.utils.data import DataLoader, TensorDataset

from dl_experiments.common import update_flat_dicts
from dl_experiments.config import MyGRUConfig, MyCNNConfig, BaseModelConfig, GeneralConfig
from dl_experiments.model import MyCNN, MyGRU
from dl_experiments.model import MyBaseModel as BaseModel


class BaseWrapper(object):
    def __init__(self, model_class: Union[MyCNN, MyGRU],
                 model_config: Union[MyCNNConfig, MyGRUConfig],
                 checkpoint: dict,
                 device: str = "cpu"):
        self.model_class: BaseModel = model_class
        self.model_config: BaseModelConfig = model_config
        self.checkpoint: dict = checkpoint
        self.device: str = device

        self.model_args, self.optimizer_args, self.loss_args = update_flat_dicts(checkpoint["best_trial_config"],
                                                                                 [model_config.model_args,
                                                                                  model_config.optimizer_args,
                                                                                  model_config.loss_args])

        self.model_args = {**self.model_args, "device": device}
        self.instance = None

    def __get_shallow_model_instance__(self, load_state: bool = True):
        model_state_dict = self.checkpoint.get("model_state_dict", None)

        model = self.model_class(**self.model_args).to(self.device).to(torch.double)
        if load_state and model_state_dict is not None:
            model.load_state_dict(model_state_dict)
        return model

    @staticmethod
    def __get_last_epoch__(state_dict: dict):
        epoch: int = -1
        if "iteration" in state_dict:
            iteration = state_dict["iteration"]
            epoch_length = state_dict.get("epoch_length", None)
            if epoch_length is not None:
                epoch = iteration // epoch_length
        elif "epoch" in state_dict:
            epoch = state_dict["epoch"]
        return epoch

    def tune(self, data: TensorDataset):
        self.instance = self.__get_shallow_model_instance__(load_state=False)
        optimizer = self.model_config.optimizer_class(
            filter(lambda p: p.requires_grad, self.instance.parameters()), **self.optimizer_args)
        loss = self.model_config.loss_class(**self.loss_args)

        # create trainer #####
        trainer = create_supervised_trainer(self.instance,
                                            optimizer,
                                            loss_fn=loss,
                                            device=self.device,
                                            non_blocking=True
                                            )
        # create data loader #####
        loader = DataLoader(data,
                            shuffle=True,
                            batch_size=GeneralConfig.batch_size)
        # start training
        max_epochs: int = BaseWrapper.__get_last_epoch__(self.checkpoint.get("trainer_state_dict", {}))
        if max_epochs == -1:
            raise ValueError("Found no trainer state-dict or epoch!")
        logging.info(f"Tune for {max_epochs} epochs...")
        trainer.run(loader, max_epochs=max_epochs)

    def predict(self, data: TensorDataset):
        if self.instance is None:
            self.instance = self.__get_shallow_model_instance__(load_state=True)

        self.instance.eval()
        # predict
        target_pred = []
        target_true = []
        with torch.no_grad():
            for features, targets in DataLoader(data, batch_size=GeneralConfig.batch_size, shuffle=False):
                features = features.to(self.device).to(torch.double)
                target_pred += [self.instance.forward_eval_single(features)]

                targets = targets.to(self.device).to(torch.double)
                target_true += [targets]
        if hasattr(self.instance, "h_previous"):
            self.instance.h_previous = None

        if not (len(target_pred) and len(target_true)):
            return torch.tensor([], dtype=torch.double).to(self.device), \
                   torch.tensor([], dtype=torch.double).to(self.device)
        else:
            return torch.cat(target_pred, 0), torch.cat(target_true, 0)
