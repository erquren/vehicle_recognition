# 车型识别

基于深度学习的车辆类型识别系统，使用 ResNet34 网络实现对 1777+ 种车型的精准分类。

[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-green)](https://www.python.org/)

## 演示效果

![demo1](https://github.com/erquren/vehicle_recognition/blob/main/car_img/demo1.png?raw=true)
![demo2](https://github.com/erquren/vehicle_recognition/blob/main/car_img/demo2.png?raw=true)
![demo3](https://github.com/erquren/vehicle_recognition/blob/main/car_img/demo3.png?raw=true)

## 项目简介

本项目使用 PyTorch 框架构建了一个基于 ResNet34 深度神经网络的车型分类模型，支持 1777+ 种车辆类别的识别。模型采用迁移学习策略，在 ImageNet 预训练权重基础上进行微调，实现了高精度的车辆类型识别。

### 核心特性

- 🚗 **支持 1777+ 车型**：覆盖主流汽车品牌和型号
- 🎯 **高精度识别**：基于 ResNet34 网络架构
- ⚡ **快速预测**：单张图片预测时间 < 0.1 秒（GPU）
- 📦 **开箱即用**：提供预训练模型，可直接使用
- 🔧 **易于扩展**：代码结构清晰，支持自定义训练

## 数据集

本项目使用 **HyperVID** 数据集进行训练，该数据集包含大量车辆图片，涵盖多种车型。

### 数据集下载

百度网盘下载链接：
- 链接: https://pan.baidu.com/s/1vvV2H5Jpewgba_VFsWvDcA
- 密码: vuo4

## 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.x
- CUDA (可选，用于 GPU 加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 快速预测

使用预训练模型进行预测：

```bash
python predict.py
```

将待预测图片放置在 `car_img/` 目录下，修改 `predict.py` 中的 `image_list` 指定你的图片路径。

## 模型训练

### 数据准备

1. 下载数据集并解压
2. 执行数据集划分脚本：

```bash
python dataset_split.py
```

### 开始训练

```bash
python train.py
```

训练参数可在 `train.py` 中调整，如：
- `epochs`: 训练轮数（默认 500）
- `batch_size`: 批次大小（默认 32）
- `lr`: 学习率（默认 0.0001）

训练完成后，模型权重将保存为 `resnet34.pth`。

## 项目结构

```
vehicle_recognition/
├── car_img/                 # 待预测图片目录
├── model.py                 # ResNet 模型定义
├── train.py                 # 训练脚本
├── predict.py               # 预测脚本
├── dataset_split.py         # 数据集划分脚本
├── resnet_car.pth          # 预训练模型权重
├── class_car.json          # 类别标签映射
└── requirements.txt         # 项目依赖
```

## 模型架构

本项目使用 ResNet34 网络作为骨干网络，主要包含以下组件：

- **BasicBlock**: ResNet 基础残差块
- **Bottleneck**: 瓶颈块（用于更深的网络）
- **ResNet**: 完整的网络结构，支持多种变体（ResNet34/50/101 等）

### 预训练权重

ResNet34 预训练权重下载地址：
https://download.pytorch.org/models/resnet34-333f7ec4.pth

## 使用示例

### 单张图片预测

```python
from model import resnet34
from predict import load_model, predict_image, load_class_mapping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_indict = load_class_mapping('class_car.json')
model = load_model(device, num_classes=1778)

# 预测图片
time_used, imgshow, result, top5 = predict_image(
    model, 'path/to/image.jpg', class_indict, device
)
print(f"预测结果: {result}")
print(f"Top-5: {top5}")
```

### 自定义训练

```python
from train import main, get_data_transform, build_model

# 修改数据路径和参数
image_path = '/path/to/dataset'
data_transform = get_data_transform()
# ... 更多自定义配置
```

## 性能指标

| 指标 | 数值 |
|------|------|
| 支持车型数 | 1777+ |
| 模型大小 | ~85 MB |
| 预测速度 (CPU) | ~0.3s/张 |
| 预测速度 (GPU) | <0.1s/张 |
| 训练准确率 | >95% |

## 常见问题

**Q: 如何添加新的车型类别？**

A: 将新车型图片添加到对应类别的训练目录下，重新训练模型。

**Q: 如何使用自己的数据集？**

A: 按照 ImageFolder 格式组织数据（每个类别一个文件夹），修改 `train.py` 中的 `image_path` 即可。

**Q: 训练显存不足怎么办？**

A: 减小 `batch_size` 或使用更小的网络（如 ResNet18）。

## 待办事项

- [ ] 扩充训练集数据量
- [ ] 支持更多车型类别
- [ ] 优化模型推理速度
- [ ] 提供 Web API 接口
- [ ] 支持批量预测

## License

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- HyperVID 数据集提供者
- PyTorch 框架及官方文档

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。
