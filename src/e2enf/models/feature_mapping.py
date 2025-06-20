import torch
from packaging import version
from torch import nn

from e2enf.layers.residual_block import ResSkipBlock

torch_is_ge_210 = version.parse(torch.__version__) >= version.parse("2.1.0")


class FeatureMapping(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,  # 入力のチャネル数
        out_channels: int = 80,  # 出力のチャネル数
        layers: int = 6,  # レイヤー数
        stacks: int = 3,  # 畳み込みブロックの数
        residual_channels: int = 1024,  # 残差結合のチャネル数
        gate_channels: int = 128,  # ゲートのチャネル数
        skip_out_channels: int = 1024,  # スキップ接続のチャネル数
        kernel_size: int = 3,  # 1 次元畳み込みのカーネルサイズ
    ):
        super().__init__()
        self.out_channels = out_channels

        self.input_x_conv = nn.Conv1d(in_channels, residual_channels, kernel_size=1, stride=1, bias=True)

        self.resblocks = nn.ModuleList()
        layers_per_stack = layers // stacks
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResSkipBlock(
                residual_channels,
                gate_channels,
                kernel_size,
                skip_out_channels,
                dilation=dilation,
            )
            self.resblocks.append(conv)

        self.postnet = nn.ModuleList(
            [
                nn.ReLU(),
                nn.Conv1d(skip_out_channels, skip_out_channels, kernel_size=1, stride=1, bias=True),
                nn.ReLU(),
                nn.Conv1d(skip_out_channels, out_channels, kernel_size=1, stride=1, bias=True),
            ]
        )

        self.apply_weight_norm()

    def forward(self, x):
        """Forward step

        Args:
            x: the acoustic features

        Returns:
            torch.Tensor: the output waveform
        """
        x = self.input_x_conv(x)
        skips = 0
        for f in self.resblocks:
            x, h = f(x)
            skips += h

        # スキップ接続の和を入力として、出力を計算
        x = skips
        for f in self.postnet:
            x = f(x)

        return x

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            try:
                if torch_is_ge_210:
                    nn.utils.parametrizations.weight_norm(m)
                else:
                    nn.utils.weight_norm(m)
            except ValueError:
                return

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization of the model"""

        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)
