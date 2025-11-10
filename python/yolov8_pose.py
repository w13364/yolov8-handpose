import os
import sys
import urllib
import urllib.request
import time
import numpy as np
import argparse
import cv2,math
from math import ceil

# 使用rknnlite
from rknnlite.api import RKNNLite

CLASSES = ['hand']

nmsThresh = 0.4
objectThresh = 0.5

# 定义21个关键点的手部骨架连接
hand_skeleton = [
    [0, 1], [1, 2], [2, 3], [3, 4],  # 拇指
    [0, 5], [5, 6], [6, 7], [7, 8],  # 食指
    [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
    [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
    [0, 17], [17, 18], [18, 19], [19, 20]   # 小指
]

# 为21个关键点定义颜色
hand_pose_palette = np.array([
    [0, 0, 255],     # 腕部 (蓝色)
    [0, 50, 255],    # 拇指根
    [0, 100, 255],   # 拇指中
    [0, 150, 255],   # 拇指近
    [0, 200, 255],   # 拇指尖
    [50, 0, 255],    # 食指根
    [100, 0, 255],   # 食指中
    [150, 0, 255],   # 食指近
    [200, 0, 255],   # 食指尖
    [0, 255, 0],     # 中指根
    [50, 255, 0],    # 中指中
    [100, 255, 0],   # 中指近
    [150, 255, 0],   # 中指尖
    [255, 0, 0],     # 无名指根
    [255, 50, 0],    # 无名指中
    [255, 100, 0],   # 无名指近
    [255, 150, 0],   # 无名指尖
    [255, 255, 0],   # 小指根
    [255, 255, 50],  # 小指中
    [255, 255, 100], # 小指近
    [255, 255, 150]  # 小指尖
], dtype=np.uint8)

# 为21个关键点分配颜色
hand_kpt_color = hand_pose_palette

# 为骨架连接分配颜色
hand_limb_color = np.array([
    [0, 0, 255], [0, 50, 255], [0, 100, 255], [0, 150, 255],  # 拇指
    [50, 0, 255], [100, 0, 255], [150, 0, 255], [200, 0, 255],  # 食指
    [0, 255, 0], [50, 255, 0], [100, 255, 0], [150, 255, 0],  # 中指
    [255, 0, 0], [255, 50, 0], [255, 100, 0], [255, 150, 0],  # 无名指
    [255, 255, 0], [255, 255, 50], [255, 255, 100], [255, 255, 150]  # 小指
], dtype=np.uint8)

def letterbox_resize(image, size, bg_color):
    """
    letterbox_resize the image according to the specified size
    :param image: input image, which can be a NumPy array or file path
    :param size: target size (width, height)
    :param bg_color: background filling data 
    :return: processed image
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    target_width, target_height = size
    image_height, image_width, _ = image.shape

    # Calculate the adjusted image size
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    # Use cv2.resize() for proportional scaling
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new canvas and fill it
    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
    return result_image, aspect_ratio, offset_x, offset_y


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint

def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    # 将输入向量减去最大值以提高数值稳定性
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def process(out,keypoints,index,model_w,model_h,stride,scale_w=1,scale_h=1):
    xywh=out[:,:64,:]
    conf=sigmoid(out[:,64:,:])
    out=[]
    for h in range(model_h):
        for w in range(model_w):
            for c in range(len(CLASSES)):
                if conf[0,c,(h*model_w)+w]>objectThresh:
                    xywh_=xywh[0,:,(h*model_w)+w] #[1,64,1]
                    xywh_=xywh_.reshape(1,4,16,1)
                    data=np.array([i for i in range(16)]).reshape(1,1,16,1)
                    xywh_=softmax(xywh_,2)
                    xywh_ = np.multiply(data, xywh_)
                    xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)

                    xywh_temp=xywh_.copy()
                    xywh_temp[0]=(w+0.5)-xywh_[0]
                    xywh_temp[1]=(h+0.5)-xywh_[1]
                    xywh_temp[2]=(w+0.5)+xywh_[2]
                    xywh_temp[3]=(h+0.5)+xywh_[3]

                    xywh_[0]=((xywh_temp[0]+xywh_temp[2])/2)
                    xywh_[1]=((xywh_temp[1]+xywh_temp[3])/2)
                    xywh_[2]=(xywh_temp[2]-xywh_temp[0])
                    xywh_[3]=(xywh_temp[3]-xywh_temp[1])
                    xywh_=xywh_*stride

                    xmin=(xywh_[0] - xywh_[2] / 2) * scale_w
                    ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
                    xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
                    ymax = (xywh_[1] + xywh_[3] / 2) * scale_h
                    # 关键点索引可能需要调整，确保正确获取21个关键点
                    keypoint=keypoints[...,(h*model_w)+w+index] 
                    keypoint[...,0:2]=keypoint[...,0:2]//1
                    box = DetectBox(c,conf[0,c,(h*model_w)+w], xmin, ymin, xmax, ymax,keypoint)
                    out.append(box)

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yolov8 Pose Python Demo', add_help=True)
    # basic params
    parser.add_argument('--model_path', type=str, required=True,
                        help='model path, could be .rknn file')
    parser.add_argument('--target', type=str,
                        default='rk3566', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str,
                        default=None, help='device id')
    args = parser.parse_args()

    # ---------- RKNN 初始化 ---------- 
    print("Initializing RKNNLite...") 
    rknn_lite = RKNNLite() 
    
    # 加载 RKNN 模型
    print(f"Loading RKNN model: {args.model_path}") 
    ret = rknn_lite.load_rknn(args.model_path) 
    if ret != 0: 
        print("Failed to load RKNN model!") 
        sys.exit(1) 
    print('done')

    print("Initializing RKNN runtime environment...")
    ret = rknn_lite.init_runtime()
    if ret != 0: 
        print("Failed to initialize RKNN runtime environment!") 
        sys.exit(1)
    print('done')

    # 设置输入 - 使用正确的输入尺寸 224x224
    img = cv2.imread('../model/11.jpg')  # 使用手部图片

    # 根据错误日志计算，模型期望的输入尺寸应该是224x224
    # 150528 ÷ 3 = 50176，√50176 ≈ 224
    input_size = (224, 224)  # 改为模型实际期望的尺寸
    letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(img, input_size, 56)  # letterbox缩放
    infer_img = letterbox_img[..., ::-1]  # BGR2RGB

    # 推理 - 添加批处理维度
    print('--> Running model')
    infer_img = np.expand_dims(infer_img, axis=0)  # 添加批处理维度，将(224,224,3)变为(1,224,224,3)
    results = rknn_lite.inference(inputs=[infer_img])
    if results is None:
        print("Inference failed, results is None")
        sys.exit(1)

    outputs=[]
    if len(results) >= 4:
        keypoints=results[3]
        for x in results[:3]:
            index,stride=0,0
            if hasattr(x, 'shape') and len(x.shape) >= 3:
                # 根据新的输入尺寸调整stride和index的计算
                if x.shape[2]==7:  # 224/32 = 7
                    stride=32
                    index=7*4*7*4+7*2*7*2  # 相应调整
                elif x.shape[2]==14:  # 224/16 = 14
                    stride=16
                    index=7*4*7*4  # 相应调整
                elif x.shape[2]==28:  # 224/8 = 28
                    stride=8
                    index=0
                else:
                    # 如果尺寸不符合预期，尝试使用原始逻辑但输出警告
                    print(f"Warning: Unexpected feature map shape: {x.shape}")
                    if x.shape[2]==20:
                        stride=32
                        index=20*4*20*4+20*2*20*2
                    elif x.shape[2]==40:
                        stride=16
                        index=20*4*20*4
                    elif x.shape[2]==80:
                        stride=8
                        index=0
                    else:
                        print(f"Error: Cannot determine stride for shape {x.shape}")
                        continue
                         
                feature=x.reshape(1,65,-1)
                output=process(feature,keypoints,index,x.shape[3],x.shape[2],stride)
                outputs=outputs+output
        predbox = NMS(outputs)

        for i in range(len(predbox)):
            xmin = int((predbox[i].xmin-offset_x)/aspect_ratio)
            ymin = int((predbox[i].ymin-offset_y)/aspect_ratio)
            xmax = int((predbox[i].xmax-offset_x)/aspect_ratio)
            ymax = int((predbox[i].ymax-offset_y)/aspect_ratio)
            classId = predbox[i].classId
            score = predbox[i].score
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            ptext = (xmin, ymin)
            title= CLASSES[classId] + "%.2f" % score

            cv2.putText(img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            # 处理21个关键点
            keypoints = predbox[i].keypoint.reshape(-1, 3)  # keypoint [x, y, conf]
            keypoints[...,0]=(keypoints[...,0]-offset_x)/aspect_ratio
            keypoints[...,1]=(keypoints[...,1]-offset_y)/aspect_ratio

            # 绘制21个关键点
            for k, keypoint in enumerate(keypoints):
                x, y, conf = keypoint
                if k < len(hand_kpt_color):
                    color_k = [int(x) for x in hand_kpt_color[k]]
                    if x != 0 and y != 0:
                        cv2.circle(img, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)
            # 绘制手部骨架连接
            for k, sk in enumerate(hand_skeleton):
                pos1 = (int(keypoints[sk[0], 0]), int(keypoints[sk[0], 1]))
                pos2 = (int(keypoints[sk[1], 0]), int(keypoints[sk[1], 1]))

                conf1 = keypoints[sk[0], 2]
                conf2 = keypoints[sk[1], 2]
                if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                    continue
                if k < len(hand_limb_color):
                    cv2.line(img, pos1, pos2, [int(x) for x in hand_limb_color[k]], thickness=2, lineType=cv2.LINE_AA)
        
        cv2.imwrite("./result.jpg", img)
        print("save image in ./result.jpg")
    else:
        print(f"Unexpected results structure, got {len(results)} outputs")
    # 释放资源
    rknn_lite.release()
