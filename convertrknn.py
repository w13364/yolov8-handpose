import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
from math import exp

ONNX_MODEL = './best.onnx'
RKNN_MODEL = './best.rknn'
DATASET = './dataset.txt'

QUANTIZE_ON = True


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
# 修改骨架连接配置为手掌关键点连接
# skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
#             [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

skeleton = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)   # 小指
]

CLASSES = ['hand']

meshgrid = []

class_num = len(CLASSES)
headNum = 3
# 修改关键点数量为21（手掌通常有21个关键点）
# keypoint_num = 17
keypoint_num = 21

strides = [8, 16, 32]
mapSize = [[80, 80], [40, 40], [20, 20]]
nmsThresh = 0.55
# 在文件开头的常量定义部分修改阈值
# 修改前：objectThresh = 0.5
# 修改后：将阈值提高到0.7或更高
objectThresh = 0.7

# 此外，我们可以在postprocess函数中添加一些额外的过滤
# 比如过滤掉过小的边界框
# 修改postprocess函数中的边界框创建部分：

# 创建检测框对象并添加到结果列表
# 添加额外的过滤条件，比如边界框的最小宽度和高度
if (xmax - xmin) > 10 and (ymax - ymin) > 10:  # 过滤掉过小的边界框
    box = DetectBox(0, cls_score, xmin, ymin, xmax, ymax, poseResult)
    detectResult.append(box)

# input_imgH = 640
# input_imgW = 640

# 修改输入尺寸为模型实际期望的224x224
input_imgH = 224
input_imgW = 224


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, pose):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.pose = pose


def GenerateMeshgrid():
    for index in range(headNum):
        for i in range(mapSize[index][0]):
            for j in range(mapSize[index][1]):
                meshgrid.append(j + 0.5)
                meshgrid.append(i + 0.5)


# 修复IOU函数中的除以零问题
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
    
    # 修复除以零的问题
    if total == 0:
        return 0
    
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
    return 1 / (1 + exp(-x))


# 修改后的postprocess函数
def postprocess(out, img_h, img_w):
    print('postprocess ... ')
    detectResult = []

    # 根据实际情况，out[0] 是形状为 [1, 68, 1029] 的张量
    if len(out) == 0:
        return []
    
    # 重塑输出张量为 [68, 1029]
    output_tensor = out[0].reshape(68, 1029)
    
    # 提取 cls, reg 和 pose 组件
    # 注意：这里的提取方式需要根据模型的实际输出格式进行调整
    # 以下是基于您提供的 68=1+4+63 信息的假设实现
    cls_data = output_tensor[0:1, :]  # 前1个通道是分类数据
    reg_data = output_tensor[1:5, :]  # 接下来的4个通道是回归数据
    pose_data = output_tensor[5:68, :]  # 最后63个通道是姿态数据
    
    # 转换到合适的格式
    cls_data = cls_data.reshape(-1)  # 展平为一维数组
    reg_data = reg_data.reshape(-1)
    pose_data = pose_data.reshape(-1)
    
    scale_h = img_h / input_imgH
    scale_w = img_w / input_imgW
    
    # 由于模型结构变化，我们需要重新实现检测逻辑
    # 这里提供一个简化的实现，您可能需要根据实际模型输出格式进一步调整
    for i in range(1029):  # 遍历1029个可能的检测结果
        # 获取置信度并应用sigmoid激活函数
        cls_score = sigmoid(cls_data[i])
        
        # 如果置信度大于阈值，则进行处理
        if cls_score > objectThresh:
            # 获取回归坐标信息
            x_center = reg_data[i * 4 + 0]
            y_center = reg_data[i * 4 + 1]
            width = reg_data[i * 4 + 2]
            height = reg_data[i * 4 + 3]
            
            # 计算边界框坐标
            xmin = (x_center - width / 2) * scale_w
            ymin = (y_center - height / 2) * scale_h
            xmax = (x_center + width / 2) * scale_w
            ymax = (y_center + height / 2) * scale_h
            
            # 边界检查
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_w, xmax)
            ymax = min(img_h, ymax)
            
            # 提取姿态关键点信息
            poseResult = []
            for kc in range(keypoint_num):
                # 获取关键点坐标和可见性得分
                px = pose_data[i * 63 + kc * 3 + 0]
                py = pose_data[i * 63 + kc * 3 + 1]
                vs = sigmoid(pose_data[i * 63 + kc * 3 + 2])
                
                # 转换到原始图像坐标系
                x = px * scale_w
                y = py * scale_h
                
                poseResult.append(vs)
                poseResult.append(x)
                poseResult.append(y)
            
            # 创建检测框对象并添加到结果列表
            box = DetectBox(0, cls_score, xmin, ymin, xmax, ymax, poseResult)
            detectResult.append(box)
    
    # 应用非最大抑制
    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)
    
    return predBox

def export_rknn_inference(img):
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], quantized_algorithm='normal', quantized_method='channel', target_platform='rk3568')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    # 使用正确的输出节点名称 'output0'
    ret = rknn.load_onnx(model=ONNX_MODEL, outputs=['output0'])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime(target='rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    rknn.release()
    print('done')

    return outputs


if __name__ == '__main__':
    print('This is main ...')
    GenerateMeshgrid()
    img_path = '11.jpg'
    orig_img = cv2.imread(img_path)
    img_h, img_w = orig_img.shape[:2]
    
    # 图像预处理
    origimg = cv2.resize(orig_img, (input_imgW, input_imgH), interpolation=cv2.INTER_LINEAR)
    origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)
    
    # 确保数据格式正确
    # 如果模型需要的是NHWC格式(1, 224, 224, 3)，则当前代码是正确的
    # 如果模型需要的是NCHW格式(1, 3, 224, 224)，则需要添加下面这行
    # origimg = origimg.transpose(2, 0, 1)  # 将HWC转换为CHW
    
    img = np.expand_dims(origimg, 0)

    outputs = export_rknn_inference(img)

    out = []
    for i in range(len(outputs)):
        out.append(outputs[i])
    predbox = postprocess(out, img_h, img_w)

    print('obj num is :', len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        xmax = int(predbox[i].xmax)
        ymax = int(predbox[i].ymax)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (128, 128, 0), 2)
        ptext = (xmin, ymin)
        title = CLASSES[classId] + "%.2f" % score
        cv2.putText(orig_img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # 手掌关键点渲染
        pose = predbox[i].pose
        for i in range(0, keypoint_num):
            if pose[i * 3] > 0.5:
                # 根据手指分组设置颜色
                if i == 0:  # 手腕点
                    color = color_list[0]  # 红色
                elif 1 <= i <= 4:  # 拇指
                    color = color_list[0]  # 红色
                elif 5 <= i <= 8:  # 食指
                    color = color_list[1]  # 绿色
                elif 9 <= i <= 12:  # 中指
                    color = color_list[2]  # 蓝色
                elif 13 <= i <= 16:  # 无名指
                    color = (255, 255, 0)  # 黄色
                else:  # 小指
                    color = (255, 0, 255)  # 紫色
                cv2.circle(orig_img, (int(pose[i * 3 + 1]), int(pose[i * 3 + 2])), 2, color, 5)

        # 渲染骨架连接线
        for i, sk in enumerate(skeleton):
            if pose[sk[0] * 3] > 0.5 and pose[sk[1] * 3] > 0.5:  # 确保两个点都有效
                pos1 = (int(pose[sk[0] * 3 + 1]), int(pose[sk[0] * 3 + 2]))
                pos2 = (int(pose[sk[1] * 3 + 1]), int(pose[sk[1] * 3 + 2]))
                # 根据骨骼所属手指设置颜色
                if sk[0] == 0:  # 连接手腕的骨骼
                    # 根据连接的手指确定颜色
                    if sk[1] == 1:  # 拇指
                        color = color_list[0]
                    elif sk[1] == 5:  # 食指
                        color = color_list[1]
                    elif sk[1] == 9:  # 中指
                        color = color_list[2]
                    elif sk[1] == 13:  # 无名指
                        color = (255, 255, 0)
                    else:  # 小指
                        color = (255, 0, 255)
                else:
                    # 根据起点确定手指颜色
                    if 1 <= sk[0] <= 3:  # 拇指
                        color = color_list[0]
                    elif 5 <= sk[0] <= 7:  # 食指
                        color = color_list[1]
                    elif 9 <= sk[0] <= 11:  # 中指
                        color = color_list[2]
                    elif 13 <= sk[0] <= 15:  # 无名指
                        color = (255, 255, 0)
                    else:  # 小指
                        color = (255, 0, 255)
                cv2.line(orig_img, pos1, pos2, color, thickness=2, lineType=cv2.LINE_AA)

    cv2.imwrite('./test_rknn_result.jpg', orig_img)
    # cv2.imshow("test", orig_img)
    # cv2.waitKey(0)
