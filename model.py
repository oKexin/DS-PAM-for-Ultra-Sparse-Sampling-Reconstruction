import torch
import torch.nn as nn

class _ResidualBlock(nn.Module):
    def __init__(self, channels, growth_channels):
        super(_ResidualBlock, self).__init__()
        self.rb = nn.Sequential(
            nn.Conv2d(channels, growth_channels, 3, 1, 1),
            nn.ReLU(True),
        )

    def forward(self, x):
        identity = x
        out = self.rb(x)
        out = torch.cat([identity, out], 1)
        return out

class _ResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth_channels, layers, reduction=16):
        super(_ResidualDenseBlock, self).__init__()
        rdb = []
        for index in range(layers):
            rdb.append(_ResidualBlock(channels + index * growth_channels, growth_channels))
        self.rdb = nn.Sequential(*rdb)

        # Local Feature Fusion layer
        self.local_feature_fusion = nn.Conv2d(channels + layers * growth_channels, channels, 1, 1, 0)
        self.se = SELayer(channels, reduction)

    def forward(self, x):
        identity = x

        out = self.rdb(x)
        out = self.local_feature_fusion(out)
        out = self.se(out)
        out = torch.add(out, identity)

        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))

class _UpsampleBlock(nn.Module):
    def __init__(self, channels, scale_factor):
        super(_UpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        base_layers = [
            nn.Conv2d(channels, channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
        ]
        if self.scale_factor == 4:
            num_blocks = 2
        elif self.scale_factor == 2:
            num_blocks = 1
        elif self.scale_factor == 6:
            num_blocks = 2
        elif self.scale_factor == 12:
            num_blocks = 3
        elif self.scale_factor == 16:
            num_blocks = 4
        else:  # scale_factor == 8
            num_blocks = 3
        upsample_layers = []
        for i in range(num_blocks):
            if (scale_factor == 6 or scale_factor == 12) and i == num_blocks - 1:
                upsample_layers += [
                    nn.Conv2d(channels, channels * 9, 3, 1, 1),
                    nn.PixelShuffle(3),
                ]
            else:
                upsample_layers += base_layers.copy()
        self.upsampling = nn.Sequential(*upsample_layers)
    def forward(self, x):
        return self.upsampling(x)

class MRDN(nn.Module):
    def __init__(
            self,
            scale_factor,
            in_channels = 1,
            out_channels = 1,
            channels = 64,
            num_rdb = 16,
            num_rb = 8,
            growth_channels = 64,
    ):
        super(MRDN, self).__init__()
        self.num_rdb = num_rdb

        # First layer
        self.conv1 = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.fusion = GateFusion(channels)
        # Second layer
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

        # Residual Dense Blocks
        trunk = []
        for _ in range(num_rdb):
            trunk.append(_ResidualDenseBlock(channels, growth_channels, num_rb))
        self.trunk = nn.Sequential(*trunk)

        # Global Feature Fusion
        self.global_feature_fusion = nn.Sequential(
            nn.Conv2d(int(num_rdb * channels), channels, 1, 1, 0),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        # Upscale block
        self.upsampling = _UpsampleBlock(channels, scale_factor)

        # Output layer
        self.conv3 = nn.Conv2d(channels, out_channels, 3, 1, 1)

    # Support torch.script function
    def forward(self, lor, ar):
        f_lor = self.conv1(lor)
        f_ear  = self.conv1(ar)
        fused = self.fusion(f_lor, f_ear)
        out  = self.conv2(fused)
        outs = []
        for i in range(self.num_rdb):
            out = self.trunk[i](out)
            outs.append(out)

        out = torch.cat(outs, 1)

        out = self.global_feature_fusion(out)

        out = torch.add(fused, out)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = torch.clamp_(out, 0.0, 1.0)
        return out

class GateFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # x: Key features (from OR)
        # y: Auxiliary prior (from AR)
        gate = self.gate(torch.cat([x, y], dim=1))  # [B, C, H, W]
        return x * gate + y * (1 - gate)

class Discriminator(nn.Module):
    def __init__(self, __patch_size__):
        super(Discriminator, self).__init__()
        # features
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.patch_size = __patch_size__ // 8
        self.conv1 = nn.Conv2d(1, 64, 3, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        # classifier
        self.linear0 = nn.Linear(256 * self.patch_size * self.patch_size, 100)
        self.linear1 = nn.Linear(100, 1)

    def forward(self, x):
        out = self.lrelu(self.bn1(self.conv1(x)))
        out = self.lrelu(self.bn2(self.conv2(out)))
        out = self.lrelu(self.bn3(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = self.lrelu(self.linear0(out))
        out = self.linear1(out)
        return out

class RDN(nn.Module):
    def __init__(
            self,
            scale_factor,
            in_channels = 1,
            out_channels = 1,
            channels = 64,
            num_rdb = 16,
            num_rb = 8,
            growth_channels = 64,
    ):
        super(RDN, self).__init__()
        self.num_rdb = num_rdb

        # First layer
        self.conv1 = nn.Conv2d(in_channels, channels, 3, 1, 1)

        # Second layer
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

        # Residual Dense Blocks
        trunk = []
        for _ in range(num_rdb):
            trunk.append(_ResidualDenseBlock(channels, growth_channels, num_rb))
        self.trunk = nn.Sequential(*trunk)

        # Global Feature Fusion
        self.global_feature_fusion = nn.Sequential(
            nn.Conv2d(int(num_rdb * channels), channels, 1, 1, 0),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

        # Upscale block
        self.upsampling = _UpsampleBlock(channels, scale_factor)

        # Output layer
        self.conv3 = nn.Conv2d(channels, out_channels, 3, 1, 1)

    # Support torch.script function
    def forward(self, lor):
        out1 = self.conv1(lor)
        out = self.conv2(out1)
        outs = []
        for i in range(self.num_rdb):
            out = self.trunk[i](out)
            outs.append(out)

        out = torch.cat(outs, 1)

        out = self.global_feature_fusion(out)
        out = torch.add(out1, out)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = torch.clamp_(out, 0.0, 1.0)
        return out


