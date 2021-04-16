from ray import tune

all_search_configs: dict = {
    "GRU_5min": {
        "input_dim": tune.choice([48, 96, 192]),
        "hidden_dim": tune.choice([2, 4, 8, 16, 32, 64]),
        "output_dim": tune.choice([1]),
        "dropout": tune.choice([0.0, 0.5]),
        "num_layers": tune.choice([1, 2]),
        "bidirectional": tune.choice([False, True])
    },
    "GRU_15min": {
        "input_dim": tune.choice([24, 48, 96]),
        "hidden_dim": tune.choice([2, 4, 8, 16, 32, 64]),
        "output_dim": tune.choice([4]),
        "dropout": tune.choice([0.0, 0.5]),
        "num_layers": tune.choice([1, 2]),
        "bidirectional": tune.choice([False, True])
    },
    "GRU_1h": {
        "input_dim": tune.choice([12, 24, 48]),
        "hidden_dim": tune.choice([2, 4, 8, 16, 32, 64]),
        "output_dim": tune.choice([12]),
        "dropout": tune.choice([0.0, 0.5]),
        "num_layers": tune.choice([1, 2]),
        "bidirectional": tune.choice([False, True])
    },
    "CNN_5min": {
        "input_dim": tune.choice([48, 96, 192]),
        "output_dim": tune.choice([1]),

        "num_layers": tune.choice([1, 2, 3, 4]),

        "dropout_1": tune.choice([0.0, 0.2, 0.5]),
        "dropout_2": tune.choice([0.0, 0.2, 0.5]),
        "dropout_3": tune.choice([0.0, 0.2, 0.5]),
        "dropout_4": tune.choice([0.0, 0.2, 0.5]),

        "num_conv_kernels_1": tune.choice([8, 16, 32, 64]),
        "num_conv_kernels_2": tune.choice([16, 32, 64]),
        "num_conv_kernels_3": tune.choice([32, 64, 128]),
        "num_conv_kernels_4": tune.choice([64, 128]),

        "conv_kernel_size_1": tune.choice([1, 3, 5, 7, 9]),
        "conv_kernel_size_2": tune.choice([1, 3, 5, 7]),
        "conv_kernel_size_3": tune.choice([1, 3, 5]),
        "conv_kernel_size_4": tune.choice([1, 3]),

        "pool_kernel_size": tune.choice([2, 3]),
        "pool_function": tune.choice(["max", "avg"])
        },
    "CNN_15min": {
        "input_dim": tune.choice([24, 48, 96]),
        "output_dim": tune.choice([4]),

        "num_layers": tune.choice([1, 2, 3, 4]),

        "dropout_1": tune.choice([0.0, 0.2, 0.5]),
        "dropout_2": tune.choice([0.0, 0.2, 0.5]),
        "dropout_3": tune.choice([0.0, 0.2, 0.5]),
        "dropout_4": tune.choice([0.0, 0.2, 0.5]),

        "num_conv_kernels_1": tune.choice([8, 16, 32, 64]),
        "num_conv_kernels_2": tune.choice([16, 32, 64]),
        "num_conv_kernels_3": tune.choice([32, 64, 128]),
        "num_conv_kernels_4": tune.choice([64, 128]),

        "conv_kernel_size_1": tune.choice([1, 3, 5, 7, 9]),
        "conv_kernel_size_2": tune.choice([1, 3, 5, 7]),
        "conv_kernel_size_3": tune.choice([1, 3, 5]),
        "conv_kernel_size_4": tune.choice([1, 3]),

        "pool_kernel_size": tune.choice([2, 3]),
        "pool_function": tune.choice(["max", "avg"])
    },
    "CNN_1h": {
        "input_dim": tune.choice([12, 24, 48]),
        "output_dim": tune.choice([12]),

        "num_layers": tune.choice([1, 2, 3, 4]),

        "dropout_1": tune.choice([0.0, 0.2, 0.5]),
        "dropout_2": tune.choice([0.0, 0.2, 0.5]),
        "dropout_3": tune.choice([0.0, 0.2, 0.5]),
        "dropout_4": tune.choice([0.0, 0.2, 0.5]),

        "num_conv_kernels_1": tune.choice([8, 16, 32, 64]),
        "num_conv_kernels_2": tune.choice([16, 32, 64]),
        "num_conv_kernels_3": tune.choice([32, 64, 128]),
        "num_conv_kernels_4": tune.choice([64, 128]),

        "conv_kernel_size_1": tune.choice([1, 3, 5, 7, 9]),
        "conv_kernel_size_2": tune.choice([1, 3, 5, 7]),
        "conv_kernel_size_3": tune.choice([1, 3, 5]),
        "conv_kernel_size_4": tune.choice([1, 3]),

        "pool_kernel_size": tune.choice([2, 3]),
        "pool_function": tune.choice(["max", "avg"])
    }
}


def get_search_space_config(model_name: str, sampling_rate: str):
    base_config: dict = {
        "lr": tune.choice([0.1, 0.01, 0.001]),
        "weight_decay": tune.choice([0.01, 0.001, 0.0001]),
    }
    add_config = all_search_configs.get(f"{model_name}_{sampling_rate}", {})

    search_space_config: dict = {**base_config, **add_config}
    return search_space_config
