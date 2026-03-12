"""
ResNet 模型实现

包含 BasicBlock、Bottleneck 块和完整的 ResNet 网络结构
"""

import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    """
    ResNet 基础块 (用于 ResNet-18 和 ResNet-34)

    Attributes:
        expansion (int): 输出通道扩展倍数,固定为 1
    """
    expansion = 1

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 downsample=None,
                 **kwargs):
        """
        初始化 BasicBlock

        Args:
            in_channel (int): 输入通道数
            out_channel (int): 输出通道数
            stride (int): 卷积步长,默认为 1
            downsample (nn.Module): 下采样层,用于调整输入维度,默认为 None
            **kwargs: 其他参数
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        """
        前向传播

        Args:
            x (Tensor): 输入张量

        Returns:
            Tensor: 输出张量
        """
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    ResNet 瓶颈块 (用于 ResNet-50、ResNet-101 等)

    Attributes:
        expansion (int): 输出通道扩展倍数,固定为 4
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 downsample=None,
                 groups=1,
                 width_per_group=64):
        """
        初始化 Bottleneck

        Args:
            in_channel (int): 输入通道数
            out_channel (int): 输出通道数
            stride (int): 卷积步长,默认为 1
            downsample (nn.Module): 下采样层,用于调整输入维度,默认为 None
            groups (int): 分组卷积的组数,默认为 1
            width_per_group (int): 每组的宽度,默认为 64
        """
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=width,
                               kernel_size=1,
                               stride=1,
                               bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width,
                               out_channels=width,
                               groups=groups,
                               kernel_size=3,
                               stride=stride,
                               bias=False,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width,
                               out_channels=out_channel * self.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """
        前向传播

        Args:
            x (Tensor): 输入张量

        Returns:
            Tensor: 输出张量
        """
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet 网络

    ResNet (残差网络) 主类,支持不同深度和变体的配置。

    Attributes:
        include_top (bool): 是否包含顶部的全连接层
        in_channel (int): 当前输入通道数
    """

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        """
        初始化 ResNet

        Args:
            block: 基础块类型 (BasicBlock 或 Bottleneck)
            blocks_num (list): 每层的块数列表,例如 [3, 4, 6, 3]
            num_classes (int): 分类类别数,默认为 1000
            include_top (bool): 是否包含顶部的全连接层,默认为 True
            groups (int): 分组卷积的组数,默认为 1
            width_per_group (int): 每组的宽度,默认为 64
        """
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3,
                               self.in_channel,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        """
        构建网络层

        Args:
            block: 基础块类型
            channel (int): 输出通道数
            block_num (int): 该层的块数
            stride (int): 第一个块的步长,默认为 1

        Returns:
            nn.Sequential: 构建好的网络层
        """
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,
                          channel * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(
            block(self.in_channel,
                  channel,
                  downsample=downsample,
                  stride=stride,
                  groups=self.groups,
                  width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(self.in_channel,
                      channel,
                      groups=self.groups,
                      width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x (Tensor): 输入张量

        Returns:
            Tensor: 输出张量
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    """
    构建 ResNet-34 模型

    Args:
        num_classes (int): 分类类别数,默认为 1000
        include_top (bool): 是否包含顶部的全连接层,默认为 True

    Returns:
        ResNet: ResNet-34 模型

    Note:
        预训练权重下载地址: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    """
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    """
    构建 ResNet-50 模型

    Args:
        num_classes (int): 分类类别数,默认为 1000
        include_top (bool): 是否包含顶部的全连接层,默认为 True

    Returns:
        ResNet: ResNet-50 模型

    Note:
        预训练权重下载地址: https://download.pytorch.org/models/resnet50-19c8e357.pth
    """
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    """
    构建 ResNet-101 模型

    Args:
        num_classes (int): 分类类别数,默认为 1000
        include_top (bool): 是否包含顶部的全连接层,默认为 True

    Returns:
        ResNet: ResNet-101 模型

    Note:
        预训练权重下载地址: https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    """
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    """
    构建 ResNeXt-50 32x4d 模型

    Args:
        num_classes (int): 分类类别数,默认为 1000
        include_top (bool): 是否包含顶部的全连接层,默认为 True

    Returns:
        ResNet: ResNeXt-50 32x4d 模型

    Note:
        预训练权重下载地址: https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    """
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    """
    构建 ResNeXt-101 32x8d 模型

    Args:
        num_classes (int): 分类类别数,默认为 1000
        include_top (bool): 是否包含顶部的全连接层,默认为 True

    Returns:
        ResNet: ResNeXt-101 32x8d 模型

    Note:
        预训练权重下载地址: https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    """
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
