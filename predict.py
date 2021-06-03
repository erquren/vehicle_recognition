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

data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def cvImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("NotoSansCJK-Black.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)


def convert_to_input(image_path):
    assert os.path.exists(image_path), "file: '{}' dose not exist.".format(image_path)
    img = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    # [N, C, H, W]
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    return img


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # read class_indict
    json_path = 'class_car.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r", encoding='utf-8')
    class_indict = json.load(json_file)
    # create model
    model = resnet34(num_classes=1778).to(device)
    # load model weights
    weights_path = "resnet_car.pth"  # 200
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    abs_image_list = ['./car_img/car_1.png', './car_img/car_2.png', './car_img/car_3.png']
    temp_list = []
    for j in range(len(abs_image_list)):
        img = convert_to_input(abs_image_list[j])
        imgshow = cv.imdecode(np.fromfile(abs_image_list[j], dtype=np.uint8), 1)
        # prediction
        model.eval()
        t1 = time.time()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            predict_max5 = torch.topk(predict, 5)
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
        temp = 0
        for ii in predict_max5[1]:
            print(class_indict[str(ii.numpy())] + str(predict[ii].numpy()))
            temp = temp + 1
            imgshow = cvImgAddText(imgshow, class_indict[str(ii.numpy())] + '  得分:' + str(predict[ii].numpy()), 10,
                                   temp * 25, (255, 255, 0), 20)

        print(print_res)
        print('res_time=' + str(time.time() - t1))
        cv.imshow('window', imgshow)
        cv.waitKey(5000)
        temp_list.append(time.time() - t1)
    print('平均时间为')
    print(np.mean(temp_list))

if __name__ == '__main__':
    main()
