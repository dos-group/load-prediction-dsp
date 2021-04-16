import argparse
import copy
import logging
import os
import sys
import time

import numpy as np
import torch
import pandas as pd

pd.options.mode.chained_assignment = None  # just for now
from sklearn.preprocessing import MinMaxScaler

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

from dl_experiments.common import create_tensor_dataset, create_dataset, create_dirs, init_logging

init_logging("INFO")
from dl_experiments.optimization import HyperOptimizer
from dl_experiments.wrapper import BaseWrapper
from dl_experiments.config import MyGRUConfig, MyCNNConfig, GeneralConfig
from dl_experiments.model import MyGRU, MyCNN

parser = argparse.ArgumentParser()

parser.add_argument("-drd", "--data-root-dir", type=str,
                    required=True, help="Path to dataset root directory.")
parser.add_argument("-dn", "--dataset-name", type=str,
                    required=True, help="Name of the dataset, e.g. 'alibaba'.")
parser.add_argument("-dsr", "--dataset-sampling-rate", type=str, choices=["5min", "15min", "1h"],
                    required=True, help="Sampling rate of the dataset.")
parser.add_argument("-dtc", "--dataset-target-column", type=str,
                    default="messages", help="Dataset target column.")
parser.add_argument("-dd", "--dataset-delimiter", type=str,
                    default=",", help="Dataset delimiter.")

parser.add_argument("-m", "--model", required=True, type=str, choices=["GRU", "CNN"],
                    help="Model class to use.")

parser.add_argument("-d", "--device", type=str, required=True,
                    help="If available, will use this device for training.")

parser.add_argument("-ts", "--test-split", type=float, default=0.8,
                    help="Fraction of window used for testing.")

parser.add_argument("-vs", "--val-split", type=float, default=0.75,
                    help="Fraction of window used for validation.")

parser.add_argument("-mmn", "--min-max-normalization", action="store_true",
                    help="Specify if min-max normalization shall be applied.")

parser.add_argument("-ft", "--fine-tuning", action="store_true",
                    help="Specify if model should be retrained with optimal arguments on train + validation data.")

parser.add_argument("-cr", "--cpu-resources", type=int, required=True,
                    help="Number of cores per trial.")

parser.add_argument("-gr", "--gpu-resources", type=float, required=True,
                    help="Fraction of gpu to use per trial.")

args = parser.parse_args()

my_class = None
my_config = None
if args.model == "GRU":
    my_class = MyGRU
    my_config = MyGRUConfig
elif args.model == "CNN":
    my_class = MyCNN
    my_config = MyCNNConfig

dataset_name = args.dataset_name
dataset_sampling_rate = args.dataset_sampling_rate

job_identifier: str = f"{dataset_name}_{dataset_sampling_rate}_{args.model}_{args.device}"
normal_identifier: str = f"{dataset_name}_{dataset_sampling_rate}_{args.device}"

# read data
path_to_file = os.path.join(args.data_root_dir, f"{dataset_name}_{dataset_sampling_rate}.csv")
base_df = pd.read_csv(path_to_file, delimiter=args.dataset_delimiter, index_col=0, parse_dates=True)
base_df = base_df.dropna()
base_df = base_df.astype(np.int).astype(np.double)
# we will use this one later
test_df = copy.deepcopy(base_df.iloc[int(len(base_df) * args.test_split):])

orig_values = base_df[args.dataset_target_column].values.reshape(-1, 1)
# extract sub-arrays
train = orig_values[:int(len(base_df) * args.test_split)]
test = orig_values[int(len(base_df) * args.test_split):]

train_train = train[:int(len(train) * args.val_split)]
train_val = train[int(len(train) * args.val_split):]

logging.info(f"Train shape: {train_train.shape}")
logging.info(f"Validation shape: {train_val.shape}")
logging.info(f"Test shape: {test.shape}")

scaler = None
if args.min_max_normalization:
    logging.info(f"Fit scaler (using {len(train_train)} datapoints)...")
    scaler = MinMaxScaler()
    train_train = scaler.fit_transform(train_train)
    train_val = scaler.transform(train_val)
    test = scaler.transform(test)
    logging.info("Scaler fitted.")

resources_per_trial: dict = {
    "cpu": args.cpu_resources,  # how many cpu cores per trial?
    "gpu": args.gpu_resources  # needs to be "0" on cpu-only devices. You can also specify fractions
}

checkpoint: dict = None
if os.path.exists(os.path.join(GeneralConfig.best_checkpoint_dir, f"{job_identifier}_checkpoint.pt")):
    # load checkpoint
    checkpoint = torch.load(os.path.join(GeneralConfig.best_checkpoint_dir, f"{job_identifier}_checkpoint.pt"),
                            map_location=args.device)
else:
    # perform optimization
    optimizer_instance: HyperOptimizer = HyperOptimizer(my_config, job_identifier, args.device)
    checkpoint = optimizer_instance.perform_optimization(optimizer_instance,
                                                         train_train,
                                                         train_val,
                                                         args.model,
                                                         dataset_sampling_rate,
                                                         resources_per_trial)

    # save checkpoint
    create_dirs(GeneralConfig.best_checkpoint_dir)
    torch.save(checkpoint, os.path.join(GeneralConfig.best_checkpoint_dir, f"{job_identifier}_checkpoint.pt"))

# update specs with best config
wrapper = BaseWrapper(my_class, my_config, checkpoint, device=args.device)

ft_duration: float = 0.0
if args.fine_tuning:
    ft_samples = np.concatenate((train_train, train_val), axis=0)
    logging.info(f"Apply fine-tuning, use {ft_samples.shape} samples...")
    ft_data = create_tensor_dataset(*create_dataset(ft_samples,
                                                    seq_length=wrapper.model_args["input_dim"],
                                                    target_length=wrapper.model_args["output_dim"],
                                                    device=args.device))
    start_time = time.time()
    wrapper.tune(ft_data)
    end_time = time.time()
    ft_duration = end_time - start_time

# also use end of training values, in order to predict first test values
test = np.concatenate((train_val[-wrapper.model_args["input_dim"]:], test), axis=0)
logging.info(f"Corrected Test shape: {test.shape}")
# create test dataset tensor
test_data = create_tensor_dataset(*create_dataset(test,
                                                  seq_length=wrapper.model_args["input_dim"],
                                                  target_length=wrapper.model_args["output_dim"],
                                                  device=args.device))
# predict / test
start_time = time.time()
pred_values, _ = wrapper.predict(test_data)
pred_values = pred_values.tolist()
end_time = time.time()
inference_duration = (end_time - start_time) / len(pred_values)

test_df["t"] = test_df.messages
test_df = test_df.drop(["messages"], axis=1)

# results
create_dirs(GeneralConfig.result_dir)
try:
    results_df = pd.read_csv(os.path.join(GeneralConfig.result_dir,
                                          f"{normal_identifier}_results.csv"), index_col=0, parse_dates=True)
except:
    results_df = test_df.t.to_frame()

results_df[args.model] = 0
horizon: int = 12 if dataset_sampling_rate == "1h" else (4 if dataset_sampling_rate == "15min" else 1)
i: int = 0
while i < len(pred_values):
    values = np.array(pred_values[i]).reshape(-1, 1)
    if scaler is not None:
        values = scaler.inverse_transform(values)
    results_df[args.model].iloc[i:i + horizon] += values.flatten()
    i += 1

great_divider = list(range(1, len(test_df) + 1))
great_divider = list(map(lambda x: min(x, horizon), great_divider))
great_divider = [min(abs(idx - len(great_divider)), x) for idx, x in enumerate(great_divider)]
logging.info(f"Great Divider: {great_divider}")
results_df[args.model] /= great_divider
results_df.to_csv(os.path.join(GeneralConfig.result_dir, f"{normal_identifier}_results.csv"))

# durations
create_dirs(GeneralConfig.result_dir)
try:
    durations_df = pd.read_csv(os.path.join(GeneralConfig.result_dir, "durations.csv"))
except:
    durations_df = pd.DataFrame.from_dict({
        "dataset": [dataset_name],
        "sampling_rate": [dataset_sampling_rate],
        "device": [args.device]
    })

dict_keys = [f"{args.model}_pred", f"{args.model}_train"]
dict_values = [inference_duration, checkpoint["time_taken"] + ft_duration]

if len(durations_df[(durations_df.dataset == dataset_name) &
                    (durations_df.sampling_rate == dataset_sampling_rate) &
                    (durations_df.device == args.device)]):

    durations_df.loc[(durations_df.dataset == dataset_name) &
                     (durations_df.sampling_rate == dataset_sampling_rate) &
                     (durations_df.device == args.device),
                     dict_keys] = dict_values
else:
    new_row = {k: v for k, v in zip(["dataset", "sampling_rate", "device"] + dict_keys,
                                    [dataset_name, dataset_sampling_rate, args.device] + dict_values)}
    # append row to the dataframe
    durations_df = durations_df.append(new_row, ignore_index=True)

durations_df = durations_df.sort_values(by=['dataset', 'sampling_rate', 'device'])
durations_df.to_csv(os.path.join(GeneralConfig.result_dir, "durations.csv"), index=False)
