import torch
import torch.nn as nn


class FCBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_channels: int,
        activation: torch.nn.Module = torch.nn.ReLU,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            activation(inplace=True),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(n_channels),
        )

    def forward(self, x):
        return self.fc(x)


class DenseEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_channels: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
        activation: torch.nn.Module = torch.nn.ReLU,
    ):

        """
        Dense embedding network that takes a tensor of shape
        (batch, n_ifos, samples) and returns a tensor of shape
        (batch, n_ifos * out_features).

        Args:
            in_features:
                Number of input features. In our context, this
                represents the number of samples in the time series.
            out_features:
                Number of output features (per channel). For example,
                if we have 2 ifos and out_features = 2, then the output
                will be of shape (batch, 2 * 2) = (batch, 4).
            n_channels:
                Number of channels. In our context, this represents the
                number of ifos.
            hidden_layer_size:
                Number of neurons in the hidden layers.
            num_hidden_layers:
                Number of hidden layers.
        """
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(in_features, hidden_layer_size),
            nn.BatchNorm1d(n_channels),
            activation(inplace=True),
        )

        self.hidden_layers = nn.Sequential(
            *[
                FCBlock(
                    hidden_layer_size,
                    hidden_layer_size,
                    n_channels,
                    activation,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_layer_size, out_features),
            nn.BatchNorm1d(n_channels),
        )

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.hidden_layers(x)
        x = self.final_layer(x)

        batch_size, n_ifos, n_features = x.shape
        x = x.view(batch_size, n_ifos * n_features)
        return x


class IncoherentDenseEmbedding(nn.Module):
    """
    DenseEmbedding for N channels that first passes each channel
    separately through a :meth: `DenseEmbedding` (i.e. incoherently).
    A final Linear layer acts on the stacked features
    from each individual embedding.

    Expected input shape is (num_batch, N, in_shape). Output shape
    is (num_batch, 1, out_shape)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_channels: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
        activation: torch.nn.Module = torch.nn.ReLU,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                DenseEmbedding(
                    in_features,
                    out_features,
                    1,
                    hidden_layer_size,
                    num_hidden_layers,
                )
                for _ in range(n_channels)
            ]
        )
        self.final_layer = nn.Sequential(
            nn.Linear(n_channels * out_features, n_channels * out_features),
            nn.BatchNorm1d(n_channels * out_features),
        )

    def forward(self, x):
        # TODO: is there a torch like way to avoid this for loop?
        embedded_vals = []
        for channel_num, embedding in enumerate(self.embeddings):
            embedded_vals.append(embedding(x[:, channel_num, :].unsqueeze(1)))

        x_concat = torch.concat(embedded_vals, dim=1)
        x_concat = self.final_layer(x_concat)

        return x_concat
