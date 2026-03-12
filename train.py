"""
车辆类型识别模型训练脚本

使用 ResNet34 网络进行车辆类型的分类训练,支持预训练权重加载和模型保存。
"""

import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34


def get_data_transform():
    """
    获取训练和验证的数据预处理变换

    Returns:
        dict: 包含训练和验证数据变换的字典
    """
    return {
        "train":
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val":
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


def save_class_mapping(dataset, save_path='class_index.json'):
    """
    保存类别索引映射到 JSON 文件

    Args:
        dataset: ImageFolder 数据集对象
        save_path (str): JSON 文件保存路径,默认为 'class_index.json'
    """
    car_list = dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in car_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open(save_path, 'w') as json_file:
        json_file.write(json_str)


def create_dataloaders(image_path,
                       data_transform,
                       batch_size=32,
                       num_workers=8):
    """
    创建训练和验证的数据加载器

    Args:
        image_path (str): 数据集根目录路径
        data_transform (dict): 数据预处理变换字典
        batch_size (int): 批次大小,默认为 32
        num_workers (int): 数据加载线程数,默认为 8

    Returns:
        tuple: (train_loader, validate_loader, train_num, val_num)
    """
    train_dataset = datasets.ImageFolder(root=os.path.join(
        image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    save_class_mapping(train_dataset)

    validate_dataset = datasets.ImageFolder(root=os.path.join(
        image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)

    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, num_workers])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(
        train_num, val_num))

    return train_loader, validate_loader, train_num, val_num


def build_model(device,
                num_classes=1777,
                pretrained_path="./resnet34-333f7ec4.pth"):
    """
    构建并初始化模型

    Args:
        device: PyTorch 设备对象
        num_classes (int): 分类类别数,默认为 1777
        pretrained_path (str): 预训练权重路径,默认为 "./resnet34-333f7ec4.pth"

    Returns:
        ResNet: 初始化后的模型
    """
    net = resnet34()
    assert os.path.exists(pretrained_path), "file {} does not exist.".format(
        pretrained_path)
    net.load_state_dict(torch.load(pretrained_path, map_location=device))

    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, num_classes)
    net.to(device)

    return net


def train_one_epoch(net, train_loader, loss_function, optimizer, device, epoch,
                    epochs):
    """
    训练一个 epoch

    Args:
        net: 模型对象
        train_loader: 训练数据加载器
        loss_function: 损失函数
        optimizer: 优化器
        device: PyTorch 设备对象
        epoch (int): 当前 epoch 编号
        epochs (int): 总 epoch 数

    Returns:
        float: 平均训练损失
    """
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)

    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
            epoch + 1, epochs, loss)

    return running_loss / len(train_loader)


def validate(net, validate_loader, device, epoch, epochs):
    """
    验证模型性能

    Args:
        net: 模型对象
        validate_loader: 验证数据加载器
        device: PyTorch 设备对象
        epoch (int): 当前 epoch 编号
        epochs (int): 总 epoch 数

    Returns:
        float: 验证准确率
    """
    net.eval()
    acc = 0.0
    val_num = 0

    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_images, val_labels in val_bar:
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            val_num += val_labels.size(0)

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

    return acc / val_num


def main():
    """
    主训练函数

    执行完整的训练流程,包括数据加载、模型构建、训练循环和模型保存。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = get_data_transform()

    image_path = '/media/HyperVID-Dataset'
    assert os.path.exists(image_path), "{} path does not exist.".format(
        image_path)

    batch_size = 32
    train_loader, validate_loader, train_num, val_num = create_dataloaders(
        image_path, data_transform, batch_size=batch_size)

    net = build_model(device, num_classes=1777)

    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 500
    best_acc = 0.0
    save_path = './resnet34.pth'
    train_steps = len(train_loader)

    for epoch in range(epochs):
        train_loss = train_one_epoch(net, train_loader, loss_function,
                                     optimizer, device, epoch, epochs)
        val_accurate = validate(net, validate_loader, device, epoch, epochs)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, train_loss, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
