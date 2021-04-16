import os

import torch
import logging
import time
import numpy as np
from functools import partial
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ignite.engine import Events, Engine, create_supervised_trainer, create_supervised_evaluator
from datetime import datetime

from torch.utils.data import DataLoader
from dl_experiments.common import update_flat_dicts, create_tensor_dataset, create_dataset
from dl_experiments.config import TuneConfig, GeneralConfig
from dl_experiments.search_config import get_search_space_config
from dl_experiments.metrics import *


class HyperOptimizer(object):
    def __init__(self, config, job_identifier: str, device: str):

        # define fields and their types
        self.config = config

        self.epochs: int = GeneralConfig.epochs
        self.job_identifier: str = job_identifier
        self.device: str = device
        self.batch_size: int = GeneralConfig.batch_size

        self.loss_args: dict = config.loss_args
        self.loss_class: callable = config.loss_class
        self.optimizer_args: dict = config.optimizer_args
        self.optimizer_class: callable = config.optimizer_class
        self.model_args: dict = {**config.model_args, "device": device}
        self.model_class: callable = config.model_class

        logging.info(
            f"Default-Model: {self.model_class}, Default-Args={self.model_args}")
        logging.info(
            f"Default-Optimizer: {self.optimizer_class}, Default-Args={self.optimizer_args}")
        logging.info(
            f"Default-Loss: {self.loss_class}, Default-Args={self.loss_args}")
        logging.info(
            f"Default-Config: Device={self.device}, BatchSize={self.batch_size}")

    @staticmethod
    def stopping_criterion(trial_id, result, threshold: float = 0, key="validation_loss", relation="lt"):
        """A function determining if the training shall be stopped."""
        return result[key] < threshold if relation == "lt" else result[key] > threshold

    @staticmethod
    def perform_optimization(hyperoptimizer_instance,
                             train_data: np.array,
                             val_data: np.array,
                             model_name: str,
                             sampling_rate: str,
                             resources_per_trial: dict):
        """Perform hyperparameter optimization."""

        # Extract optionally provided configurations #####
        scheduler_config: dict = TuneConfig.scheduler
        optuna_search_config: dict = TuneConfig.optuna_search
        concurrency_limiter_config: dict = TuneConfig.concurrency_limiter
        tune_run_config: dict = TuneConfig.tune_run
        stopping_criterion_config: dict = TuneConfig.stopping_criterion
        tune_best_trial_config: dict = TuneConfig.tune_best_trial
        ##################################################
        search_space_config: dict = get_search_space_config(model_name, sampling_rate)

        scheduler = ASHAScheduler(max_t=GeneralConfig.epochs, **scheduler_config)

        reporter = CLIReporter(parameter_columns=list(search_space_config.keys()),
                               metric_columns=["validation_loss", "mae", "mse", "rmse", "smape"])

        search_alg = OptunaSearch(**optuna_search_config)
        search_alg = ConcurrencyLimiter(
            search_alg, **concurrency_limiter_config)

        tune_run_name = f"{hyperoptimizer_instance.job_identifier}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        start = time.time()

        analysis = tune.run(tune.with_parameters(partial(hyperoptimizer_instance,
                                                         target_metric="validation_loss",
                                                         epochs=GeneralConfig.epochs),
                                                 train_data=train_data,
                                                 val_data=val_data),
                            name=tune_run_name,
                            scheduler=scheduler,
                            progress_reporter=reporter,
                            search_alg=search_alg,
                            config=search_space_config,
                            resources_per_trial=resources_per_trial,
                            stop=partial(
                                HyperOptimizer.stopping_criterion, **stopping_criterion_config),
                            **tune_run_config)

        time_taken = time.time() - start

        # get best trial
        best_trial = analysis.get_best_trial(**tune_best_trial_config)

        # get some information from best trial
        best_trial_val_loss = best_trial.metric_analysis["validation_loss"]["min"]

        logging.info(f"Time taken: {time_taken:.2f} seconds.")
        logging.info("Best trial config: {}".format(best_trial.config))
        logging.info("Best trial final validation loss: {}".format(
            best_trial_val_loss))

        # load best checkpoint of best trial
        best_checkpoint_dir = analysis.get_best_checkpoint(best_trial)

        model_state_dict, optimizer_state_dict, trainer_state_dict, evaluator_state_dict = torch.load(
            os.path.join(best_checkpoint_dir, "checkpoint"))

        return {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "trainer_state_dict": trainer_state_dict,
            "evaluator_state_dict": evaluator_state_dict,
            "best_trial_config": best_trial.config,
            "best_trial_val_loss": best_trial_val_loss,
            "time_taken": time_taken
        }

    def __call__(self, config,
                 checkpoint_dir: str = None,
                 epochs: int = None,
                 target_metric: str = None,
                 train_data: np.array = None,
                 val_data: np.array = None):
        """Called by 'tune.run' during hyperparameter optimization.
        Check: https://docs.ray.io/en/releases-1.1.0/tune/api_docs/trainable.html"""

        if val_data is None:
            raise ValueError("'val_data' must be specified.")
        if train_data is None:
            raise ValueError("'train_data' must be specified.")

        # extract and override #####
        batch_size = self.batch_size
        model_args, optimizer_args, loss_args = update_flat_dicts(
            config, [self.model_args, self.optimizer_args, self.loss_args])
        ############################
        
        # also use end of training values, in order to predict first validation values
        val_data = np.concatenate((train_data[-model_args["input_dim"]:], val_data), axis=0)

        model = self.model_class(**model_args).double()
        optimizer = self.optimizer_class(
            filter(lambda p: p.requires_grad, model.parameters()), **optimizer_args)
        loss = self.loss_class(**loss_args)

        # create torch datasets
        train_data = create_tensor_dataset(*create_dataset(train_data,
                                                           seq_length=model_args["input_dim"],
                                                           target_length=model_args["output_dim"],
                                                           device=self.device))
        val_data = create_tensor_dataset(*create_dataset(val_data,
                                                         seq_length=model_args["input_dim"],
                                                         target_length=model_args["output_dim"],
                                                         device=self.device))

        # create trainer #####
        trainer = create_supervised_trainer(model,
                                            optimizer,
                                            loss_fn=loss,
                                            device=self.device,
                                            non_blocking=True
                                            )
        ######################
        
        # create evaluator #####
        val_metrics: dict = {
            "validation_loss": Loss(loss),
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
            "rmse": RootMeanSquaredError(),
            "smape": SymmetricMeanAbsolutePercentageError()
        }
        evaluator = create_supervised_evaluator(model,
                                                device=self.device,
                                                metrics=val_metrics)
        ########################

        # restore if possible #####
        if checkpoint_dir:
            checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.load_state_dict(checkpoint["trainer_state_dict"])
            evaluator.load_state_dict(checkpoint["evaluator_state_dict"])
        ###########################
        model = model.to(self.device)

        # create data loaders #####
        train_loader = DataLoader(train_data,
                                  shuffle=True,
                                  batch_size=batch_size)

        val_loader = DataLoader(val_data,
                                shuffle=False,
                                batch_size=batch_size)

        ###########################

        @trainer.on(Events.EPOCH_COMPLETED)
        def post_epoch_actions(trainer_instance: Engine):

            # evaluate model on validation set
            evaluator.run(val_loader)
            state_val_metrics = evaluator.state.metrics

            current_epoch: int = trainer_instance.state.epoch

            with tune.checkpoint_dir(current_epoch) as local_checkpoint_dir:
                # save model, optimizer and trainer checkpoints
                path = os.path.join(local_checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict(),
                            trainer_instance.state_dict(), evaluator.state_dict()), path)

            # report validation scores to ray-tune
            report_dict: dict = {
                **state_val_metrics,
                "done": current_epoch == epochs
            }

            tune.report(**report_dict)

        # start training
        trainer.run(train_loader, max_epochs=epochs)
