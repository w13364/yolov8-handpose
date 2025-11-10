import cv2
import torch
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend

# 定义输入大小为224×224
INPUT_WIDTH = 224
INPUT_HEIGHT = 224

# 手部关键点连接关系
hand_skeleton = [
    [0, 1], [1, 2], [2, 3], [3, 4],  # 拇指
    [0, 5], [5, 6], [6, 7], [7, 8],  # 食指
    [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
    [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
    [0, 17], [17, 18], [18, 19], [19, 20]   # 小指
]

# 手部关键点颜色
keypoint_colors = [
    (255, 255, 255),  # 关键点颜色（白色）
] * 21

# 手指连接线颜色
connection_colors = [
    (0, 0, 255),    # 拇指 - 红色
    (0, 128, 255),  # 食指 - 橙色
    (0, 255, 255),  # 中指 - 黄色
    (0, 255, 0),    # 无名指 - 绿色
    (255, 0, 0)     # 小指 - 蓝色
]

def preprocess_letterbox(image):
    letterbox = LetterBox(new_shape=(INPUT_HEIGHT, INPUT_WIDTH), stride=32, auto=True)
    image = letterbox(image=image)
    image = (image[..., ::-1] / 255.0).astype(np.float32) # BGR to RGB, 0 - 255 to 0.0 - 1.0
    image = image.transpose(2, 0, 1)[None]  # BHWC to BCHW (n, 3, h, w)
    image = torch.from_numpy(image)
    return image

def preprocess_warpAffine(image):
    scale = min((INPUT_WIDTH / image.shape[1], INPUT_HEIGHT / image.shape[0]))
    ox = (INPUT_WIDTH  - scale * image.shape[1]) / 2
    oy = (INPUT_HEIGHT - scale * image.shape[0]) / 2
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
    ], dtype=np.float32)
    
    img_pre = cv2.warpAffine(image, M, (INPUT_WIDTH, INPUT_HEIGHT), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    IM = cv2.invertAffineTransform(M)
 
    img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]  # BHWC to BCHW (n, 3, h, w)
    img_pre = torch.from_numpy(img_pre)
    return img_pre, IM

def iou(box1, box2):
    def area_box(box):
        return (box[2] - box[0]) * (box[3] - box[1])
 
    left   = max(box1[0], box2[0])
    top    = max(box1[1], box2[1])
    right  = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])
    cross  = max((right-left), 0) * max((bottom-top), 0)
    union  = area_box(box1) + area_box(box2) - cross
    if cross == 0 or union == 0:
        return 0
    return cross / union

def NMS(boxes, iou_thres):
 
    remove_flags = [False] * len(boxes)
 
    keep_boxes = []
    for i, ibox in enumerate(boxes):
        if remove_flags[i]:
            continue
 
        keep_boxes.append(ibox)
        for j in range(i + 1, len(boxes)):
            if remove_flags[j]:
                continue
 
            jbox = boxes[j]
            if iou(ibox, jbox) > iou_thres:
                remove_flags[j] = True
    return keep_boxes

def postprocess(pred, IM=[], conf_thres=0.5, iou_thres=0.2):
    # 输出形状为 [1, 68, 1029]，我们需要转换为适用于后续处理的格式
    # 首先将输出转换为[1, 8400, 68]的形状以便处理
    if pred.shape == (1, 68, 1029):
        pred = pred.transpose(1, 2)  # 转换为[1, 1029, 68]
        # 我们只需要前8400个检测结果
        if pred.shape[1] > 8400:
            pred = pred[:, :8400, :]
    
    boxes = []
    # 手部检测只有1个类别，所以简化处理
    for img_id, box_id in zip(*np.where(pred[...,4] > conf_thres)):
        item = pred[img_id, box_id]
        cx, cy, w, h, conf = item[:5]
        left    = cx - w * 0.5
        top     = cy - h * 0.5
        right   = cx + w * 0.5
        bottom  = cy + h * 0.5
        # 手部关键点数量为21，每个关键点包含3个值(x, y, confidence)
        # 确保关键点数量为21*3=63
        keypoints = item[5:5+63].reshape(-1, 3)
        if len(keypoints) < 21:
            # 如果关键点数量不足，补充到21个
            padding = np.zeros((21 - len(keypoints), 3), dtype=np.float32)
            keypoints = np.vstack((keypoints, padding))
        
        # 坐标转换
        if len(IM) > 0:
            keypoints[:, 0] = keypoints[:, 0] * IM[0][0] + IM[0][2]
            keypoints[:, 1] = keypoints[:, 1] * IM[1][1] + IM[1][2]
        
        boxes.append([left, top, right, bottom, conf, *keypoints.reshape(-1).tolist()])
 
    if len(boxes) > 0:
        boxes = np.array(boxes)
        if len(IM) > 0:
            lr = boxes[:,[0, 2]]
            tb = boxes[:,[1, 3]]
            boxes[:,[0,2]] = IM[0][0] * lr + IM[0][2]
            boxes[:,[1,3]] = IM[1][1] * tb + IM[1][2]
        
        boxes = sorted(boxes.tolist(), key=lambda x:x[4], reverse=True)
        return NMS(boxes, iou_thres)
    else:
        return []

def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    r, g, b = 0, 0, 0
 
    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q
 
    return int(b * 255), int(g * 255), int(r * 255)

def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)

if __name__ == "__main__":
    
    # 读取测试图像
    img = cv2.imread("8.jpg")  # 使用手部图像进行测试
    if img is None:
        print("无法读取图像，请检查路径")
        exit()
    
    # 预处理图像
    img_pre, IM = preprocess_warpAffine(img)
    
    # 加载手部检测模型
    try:
        # 修复：直接传入模型路径作为位置参数，而不是使用weights=参数
        model = AutoBackend("best.pt")
        names = model.names
        
        # 模型推理
        result = model(img_pre)[0]
        
        # 确保输出形状符合 [1, 68, 1029]
        print(f"模型输出形状: {result.shape}")
        
        # 后处理
        boxes = postprocess(result, IM)
        
        # 绘制结果
        for box in boxes:
            left, top, right, bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            confidence = box[4]
            label = 0  # 手部检测只有一个类别
            
            # 绘制边界框
            color = (255, 255, 0)  # 青色检测框
            cv2.rectangle(img, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
            caption = f"hand {confidence:.2f}"
            w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
            cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
            cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)
            
            # 绘制关键点
            keypoints = box[5:]
            keypoints = np.array(keypoints).reshape(-1, 3)
            
            # 存储有效的关键点用于绘制连接线
            valid_keypoints = []
            for i, keypoint in enumerate(keypoints):
                x, y, conf = keypoint
                if conf > 0.5 and x != 0 and y != 0:
                    cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                    valid_keypoints.append((int(x), int(y)))
                else:
                    valid_keypoints.append((-1, -1))
            
            # 绘制关键点连接线
            color_index = 0
            connection_count = 0
            for i, sk in enumerate(hand_skeleton):
                pos1 = valid_keypoints[sk[0]]
                pos2 = valid_keypoints[sk[1]]
                
                if pos1[0] != -1 and pos1[1] != -1 and pos2[0] != -1 and pos2[1] != -1:
                    cv2.line(img, pos1, pos2, connection_colors[color_index], thickness=2, lineType=cv2.LINE_AA)
                
                # 每4个连接为一组，对应一根手指
                connection_count += 1
                if connection_count % 4 == 0:
                    color_index = min(color_index + 1, len(connection_colors) - 1)
        
        # 保存结果
        cv2.imwrite("hand_result.jpg", img)
        print("保存检测结果完成")
    except Exception as e:
        print(f"处理过程中出错: {e}")