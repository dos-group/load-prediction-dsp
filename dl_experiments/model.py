import torch
from torch.nn import functional as F
import torch.nn as nn


class MyBaseModel(nn.Module):

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def forward_eval_single(self, *args, **kwargs):
        raise NotImplementedError

    def __reset_parameters__(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MyCNN(MyBaseModel):
    def __init__(self, *args, **kwargs):
        super(MyCNN, self).__init__()

        # define fields and their types
        self.input_dim: int
        self.output_dim: int

        self.num_layers: int

        self.dropout_1: float
        self.dropout_2: float
        self.dropout_3: float
        self.dropout_4: float

        self.num_conv_kernels_1: int
        self.num_conv_kernels_2: int
        self.num_conv_kernels_3: int
        self.num_conv_kernels_4: int

        self.conv_kernel_size_1: int
        self.conv_kernel_size_2: int
        self.conv_kernel_size_3: int
        self.conv_kernel_size_4: int

        self.pool_kernel_size: int
        self.pool_function: str

        self.__dict__.update(kwargs)

        def get_pool_layer():
            pool_class = nn.AvgPool1d if self.pool_function == "avg" else nn.MaxPool1d
            return pool_class(self.pool_kernel_size, stride=1)

        self.block_1 = nn.Sequential(
            nn.Conv1d(1,
                      self.num_conv_kernels_1,
                      self.conv_kernel_size_1,
                      padding=int(self.conv_kernel_size_1 / 2)),
            nn.ReLU(),
            get_pool_layer(),
            nn.Dropout(p=self.dropout_1)
        )
        self.fc_in_dim: int = self.num_conv_kernels_1 * (
                self.input_dim - (self.num_layers * (self.pool_kernel_size - 1)))

        self.block_2 = None
        if self.num_layers >= 2:
            self.block_2 = nn.Sequential(
                nn.Conv1d(self.num_conv_kernels_1,
                          self.num_conv_kernels_2,
                          self.conv_kernel_size_2,
                          padding=int(self.conv_kernel_size_2 / 2)),
                nn.ReLU(),
                get_pool_layer(),
                nn.Dropout(p=self.dropout_2)
            )
            self.fc_in_dim: int = self.num_conv_kernels_2 * (
                        self.input_dim - (self.num_layers * (self.pool_kernel_size - 1)))

        self.block_3 = None
        if self.num_layers >= 3:
            self.block_3 = nn.Sequential(
                nn.Conv1d(self.num_conv_kernels_2,
                          self.num_conv_kernels_3,
                          self.conv_kernel_size_3,
                          padding=int(self.conv_kernel_size_3 / 2)),
                nn.ReLU(),
                get_pool_layer(),
                nn.Dropout(p=self.dropout_3)
            )
            self.fc_in_dim: int = self.num_conv_kernels_3 * (
                    self.input_dim - (self.num_layers * (self.pool_kernel_size - 1)))

        self.block_4 = None
        if self.num_layers >= 4:
            self.block_4 = nn.Sequential(
                nn.Conv1d(self.num_conv_kernels_3,
                          self.num_conv_kernels_4,
                          self.conv_kernel_size_4,
                          padding=int(self.conv_kernel_size_4 / 2)),
                nn.ReLU(),
                get_pool_layer(),
                nn.Dropout(p=self.dropout_4)
            )
            self.fc_in_dim: int = self.num_conv_kernels_4 * (
                    self.input_dim - (self.num_layers * (self.pool_kernel_size - 1)))

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.fc_in_dim, self.output_dim)

        self.__reset_parameters__()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block_1(x)

        if self.block_2 is not None:
            x = self.block_2(x)

        if self.block_3 is not None:
            x = self.block_3(x)

        if self.block_4 is not None:
            x = self.block_4(x)

        x = self.flatten(x)
        return self.fc(x)

    # Just for compatibility #
    def forward_eval_single(self, x):
        return self.forward(x)

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__


class MyGRU(MyBaseModel):
    def __init__(self, *args, **kwargs):
        super(MyGRU, self).__init__()

        # define fields and their types
        self.input_dim: int
        self.hidden_dim: int
        self.output_dim: int
        self.dropout: float
        self.num_layers: int
        self.bidirectional: bool
        self.device: str

        self.__dict__.update(kwargs)

        # Used to memorize hidden state when doing online sequence predictions
        self.h_previous = None

        fc_input_dim = 2 * self.hidden_dim if self.bidirectional else self.hidden_dim

        self.gru = nn.GRU(
            self.input_dim,
            self.hidden_dim,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            num_layers=self.num_layers)

        self.fc = nn.Linear(fc_input_dim, self.output_dim)

        self.__reset_parameters__()

    def forward(self, x):

        x = x.unsqueeze(0)
        num_directions = 2 if self.bidirectional else 1

        h_0 = self.__init_hidden__(num_directions, x)
        output_gru, _ = self.gru(x, h_0)
        output_gru = output_gru.squeeze(0)

        return self.fc(output_gru)

    def forward_eval_single(self, x_t, reset=False):

        x_t = x_t.unsqueeze(0)
        num_directions = 2 if self.bidirectional else 1

        # Hidden state in first seq of the GRU
        if reset or self.h_previous is None:
            h_0 = self.__init_hidden__(num_directions, x_t)
        else:
            h_0 = self.h_previous

        output_gru, h_n = self.gru(x_t, h_0)
        self.h_previous = h_n
        output_gru = output_gru.squeeze(0)

        return self.fc(output_gru)

    def __init_hidden__(self, num_directions, x):
        h0 = torch.zeros((num_directions * self.num_layers, x.size(0), self.hidden_dim), dtype=torch.float64,
                         device=self.device)
        return h0

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__
