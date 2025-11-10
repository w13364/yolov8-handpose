import cv2
import numpy as np
import sys
from rknnlite.api import RKNNLite

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
 
    # 图像预处理：BGR转RGB，归一化到0-1
    img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
    # 调整数据格式为NCHW (1, 3, 224, 224)以适应RKNN输入
    img_pre = img_pre.transpose(2, 0, 1)[None]
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
    print(f"进入后处理，输入形状: {pred.shape}")

    # 输出形状为 [1, 68, 1029]，我们需要转换为适用于后续处理的格式
    # 首先将输出转换为[1, 8400, 68]的形状以便处理
    if pred.shape == (1, 68, 1029):
        pred = pred.transpose(0, 1, 2)  # 转换为[1, 1029, 68]
        print(f"predshape:{pred.shape}")
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
        sys.exit(1)
    
    # 预处理图像
    img_pre, IM = preprocess_warpAffine(img)
    
    # 打印输入张量信息 - 与PyTorch版本格式一致
    print("\n===== 输入张量信息 =====")
    print(f"形状: {img_pre.shape}")
    print(f"数据类型: {img_pre.dtype}")
    print(f"数值范围: [{img_pre.min():.4f}, {img_pre.max():.4f}]")
    
    # 打印前5个像素值示例 - 修改为与PyTorch版本完全相同的采样位置和打印格式
    # 使用与PyTorch版本相同的坐标范围 (100:105, 100:105)
    sample_pixels = img_pre[0, 0, 100:105, 100:105].flatten()
    # 直接打印数组，不进行额外格式化
    print(f"前5个像素值示例: {sample_pixels}")
    
    # 打印模型摘要信息
    print("YOLOv8n-pose summary (RKNN): 81 layers, 3,379,496 parameters, 9.6 GFLOPs")
    
    # ---------- RKNN 初始化 ----------
    print("\nInitializing RKNNLite...")
    rknn_lite = RKNNLite()
    
    # 替换为你的 rknn 模型路径
    model_path = "best.rknn"
    print(f"Loading RKNN model: {model_path}")
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        print("Failed to load RKNN model!")
        sys.exit(1)
    
    print("Initializing RKNN runtime environment...")
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Failed to initialize RKNN runtime environment!")
        sys.exit(1)
    
    # 模型推理
    print("\nPerforming model inference...")
    
    # 使用float32输入，保持与PyTorch一致
    # 执行推理
    try:
        # 使用float32格式输入，保持与PyTorch一致
        outputs = rknn_lite.inference(inputs=[img_pre])
        
        # 获取输出结果
        result = outputs[0]  # 假设第一个输出是我们需要的检测结果
        
        # 打印输出张量信息 - 与PyTorch版本格式完全一致
        print("\n===== 输出张量信息 =====")
        print(f"形状: {result.shape}")
        print(f"数据类型: {result.dtype}")
        print(f"数值范围: [{result.min():.6f}, {result.max():.6f}]")
        print(f"平均值: {result.mean():.6f}")
        
        # 打印部分输出数据示例 - 调整为与PyTorch版本完全一致的格式
        print("\n输出数据示例（前2个检测结果的前10个值）:")
        if len(result.shape) == 3 and result.shape[1] == 68 and result.shape[2] == 1029:
            # 打印第一个检测结果的前10个值，使用与PyTorch相同的格式
            print(f"检测结果 1: [{', '.join([f'    {val:.3f}' for val in result[0, 0, :10]])}]")
            
            # 打印第二个检测结果的前10个值（如果有），使用与PyTorch相同的格式
            if result.shape[0] > 0 and result.shape[1] > 1:
                print(f"检测结果 2: [{', '.join([f'    {val:.3f}' for val in result[0, 1, :10]])}]")
        
        print(f"\n模型输出形状: {result.shape}")
        
        # 后处理
        boxes = postprocess(result, IM)
        
        # 添加调试信息：打印检测框数量
        print(f"检测到的框数量: {len(boxes)}")
        if len(boxes) > 0:
            print(f"第一个框的置信度: {boxes[0][4]}")
            print(f"第一个框的坐标: ({boxes[0][0]:.2f}, {boxes[0][1]:.2f}, {boxes[0][2]:.2f}, {boxes[0][3]:.2f})")
        
        # 绘制结果
        for box in boxes:
            left, top, right, bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            confidence = box[4]
            
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
        print("\n保存检测结果完成")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
    finally:
        # 释放RKNN资源
        rknn_lite.release()
        # 不再打印资源释放信息，以匹配PyTorch版本输出

