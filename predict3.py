import face_recognition
import cv2
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import GoogLeNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# age
json_path_age = './class_indices_age.json'
assert os.path.exists(json_path_age), "file: '{}' dose not exist.".format(json_path_age)

with open(json_path_age, "r") as f:
    class_indict_age = json.load(f)

model_age = GoogLeNet(num_classes=10, aux_logits=False).to(device)

weights_path_age = "./googleNet_age_158_0.6034.pth"
assert os.path.exists(weights_path_age), "file: '{}' dose not exist.".format(weights_path_age)
missing_keys, unexpected_keys = model_age.load_state_dict(torch.load(weights_path_age, map_location=device),
                                                          strict=False)
model_age.eval()

# gender
json_path_gender = './class_indices_gender.json'
assert os.path.exists(json_path_gender), "file: '{}' dose not exist.".format(json_path_gender)

with open(json_path_gender, "r") as f:
    class_indict_gender = json.load(f)

model_gender = GoogLeNet(num_classes=2, aux_logits=False).to(device)

weights_path_gender = "./googleNet_gender_126_0.9139.pth"
assert os.path.exists(weights_path_gender), "file: '{}' dose not exist.".format(weights_path_gender)
missing_keys, unexpected_keys = model_gender.load_state_dict(torch.load(weights_path_gender, map_location=device),
                                                             strict=False)
model_gender.eval()


path = './test2/'
testList = os.listdir(path)
path_save = './save2/'

all = len(testList)
num = 1

for file in testList:

    img_path = path + file
    frame = cv2.imread(img_path)

    height, width = frame.shape[:2]  # 获取原图像的水平方向尺寸和垂直方向尺寸
    frame = cv2.resize(frame, (width * 6, height * 6), interpolation=cv2.INTER_CUBIC)

    #
    # faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将图像转化为rgb颜色通道

    # # Detect faces
    # face_locations = faceCascade.detectMultiScale(
    #     rgb_frame,
    #     scaleFactor=1.5,
    #     minNeighbors=5,
    #     flags=cv2.CASCADE_SCALE_IMAGE
    # )
    # For each face
    #
    # for (x, y, w, h) in face_locations:
        # Draw rectangle around the face
        # cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 3)
        # left, top = x,y
        # right, bottom = x + w, y + h


    face_locations = face_recognition.face_locations(rgb_frame)  # 获得所有人脸位置


    # 将捕捉到的人脸显示出来
    for top, right, bottom, left in face_locations:

        # top -= 20
        # right += 10
        # bottom += 10
        # left -= 10

        # # 确保四个坐标不超出图像范围
        # top = max(0, top)
        # right = min(frame.shape[1], right)
        # bottom = min(frame.shape[0], bottom)
        # left = max(0, left)


        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)  # 画人脸矩形框

        img = rgb_frame[top:bottom, left:right,:]
        #
        img = Image.fromarray(img)

        # img1 = Image.open(img_path)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)


        import gc
        import objgraph

        # 进行垃圾回收
        gc.collect()

        # 获取对象数目最多的50个类型的信息
        obj_types = objgraph.most_common_types(limit=50)

        # 将对象数目输出到txt文件
        with open("object_count.txt", "a") as f:
            for obj_type, count in obj_types:
                f.write(f"{obj_type}: {count}\n")

            f.write("\n")

        # 文件将在with语句块结束后自动关闭

        with torch.no_grad():
            output = torch.squeeze(model_age(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        with torch.no_grad():
            output1 = torch.squeeze(model_gender(img.to(device))).cpu()
            predict1 = torch.softmax(output1, dim=0)
            predict_cla1 = torch.argmax(predict1).numpy()


        s = class_indict_gender[str(predict_cla1)] + '  ' + class_indict_age[str(predict_cla)]
        # 加上人名标签
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, s, (left + 3, bottom - 3), font, 1.0, (255, 255, 255), 1)


        print('{}/{}  Image Name: {}, predict: {}'.format(num,all,file, s))

    num += 1

    img_save = path_save + file
    cv2.imwrite(img_save, frame)


