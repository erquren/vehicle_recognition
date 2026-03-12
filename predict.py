"""
车辆类型识别预测脚本

使用训练好的 ResNet 模型对车辆图片进行分类预测
"""

import os
import json
import time
import numpy as np
import torch
from torchvision import transforms
import cv2 as cv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from model import resnet34
from PIL import Image, ImageDraw, ImageFont

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def cvImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    """
    在 OpenCV 图片上添加中文文本

    Args:
        img: 图片对象 (OpenCV 或 PIL 格式)
        text (str): 要添加的文本内容
        left (int): 文本左边界距离
        top (int): 文本上边界距离
        textColor (tuple): 文本颜色 (B, G, R),默认为 (0, 255, 0)
        textSize (int): 字体大小,默认为 20

    Returns:
        np.ndarray: 添加了文本的图片 (OpenCV 格式)
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("NotoSansCJK-Black.ttc",
                                  textSize,
                                  encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)


def convert_to_input(image_path):
    """
    将图片路径转换为模型输入张量

    Args:
        image_path (str): 图片文件路径

    Returns:
        Tensor: 模型输入张量 [1, C, H, W]
    """
    assert os.path.exists(image_path), "file: '{}' dose not exist.".format(
        image_path)
    img = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    return img


def load_class_mapping(json_path):
    """
    加载类别索引映射

    Args:
        json_path (str): JSON 文件路径

    Returns:
        dict: 类别索引映射字典
    """
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(
        json_path)
    with open(json_path, "r", encoding='utf-8') as json_file:
        class_indict = json.load(json_file)
    return class_indict


def load_model(device, num_classes=1778, weights_path="resnet_car.pth"):
    """
    加载模型和权重

    Args:
        device: PyTorch 设备对象
        num_classes (int): 分类类别数,默认为 1778
        weights_path (str): 模型权重路径,默认为 "resnet_car.pth"

    Returns:
        ResNet: 加载了权重的模型
    """
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(
        weights_path)
    model = resnet34(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


def predict_image(model, image_path, class_indict, device):
    """
    对单张图片进行预测并返回结果

    Args:
        model: 模型对象
        image_path (str): 图片路径
        class_indict (dict): 类别索引映射
        device: PyTorch 设备对象

    Returns:
        tuple: (预测时间, 显示图片, 预测结果字符串, Top5 预测列表)
    """
    img = convert_to_input(image_path)
    imgshow = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)

    model.eval()
    start_time = time.time()

    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        predict_max5 = torch.topk(predict, 5)

    inference_time = time.time() - start_time
    print_res = "class: {}   prob: {:.3}".format(
        class_indict[str(predict_cla)], predict[predict_cla].numpy())

    top5_results = []
    for idx, ii in enumerate(predict_max5[1]):
        class_name = class_indict[str(ii.numpy())]
        probability = predict[ii].numpy()
        top5_results.append((class_name, probability))
        print(f"{class_name}: {probability:.4f}")

        imgshow = cvImgAddText(imgshow, f"{class_name}  得分:{probability:.4f}",
                               10, (idx + 1) * 25, (255, 255, 0), 20)

    return inference_time, imgshow, print_res, top5_results


def main():
    """
    主预测函数

    执行车辆类型预测,支持批量图片预测和结果展示
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_indict = load_class_mapping('class_car.json')
    model = load_model(device, num_classes=1778, weights_path="resnet_car.pth")

    image_list = [
        './car_img/car_1.png', './car_img/car_2.png', './car_img/car_3.png'
    ]
    inference_times = []

    for image_path in image_list:
        print(f"\n预测图片: {image_path}")
        inference_time, imgshow, print_res, _ = predict_image(
            model, image_path, class_indict, device)

        print(print_res)
        print(f'预测时间: {inference_time:.4f} 秒')

        cv.imshow('window', imgshow)
        cv.waitKey(5000)
        inference_times.append(inference_time)

    print(f'\n平均预测时间: {np.mean(inference_times):.4f} 秒')


if __name__ == '__main__':
    main()
