# 车型识别

![demo1](https://github.com/erquren/vehicle_recognition/blob/main/car_img/demo1.png?raw=true)
![demo2](https://github.com/erquren/vehicle_recognition/blob/main/car_img/demo2.png?raw=true)
![demo3](https://github.com/erquren/vehicle_recognition/blob/main/car_img/demo3.png?raw=true)

### 简介

使用pytorch做的车型分类模型，采用resnet网络，可以支持1777个类别

数据集使用的是hyperVID 下载地址为 链接: https://pan.baidu.com/s/1vvV2H5Jpewgba_VFsWvDcA   密码: vuo4


resnet_car.pth 为检测模型

predict.py 为预测程序

### 使用方法


pip install -r requirements.txt

### 数据预测

python predict.py



### 开始训练

下载数据集解压后执行 python dataset_split.py

分好后执行训练程序 python train.py

### TODO

增加训练集
