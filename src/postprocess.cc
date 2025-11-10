#include "postprocess.h"
#include <algorithm>
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

static inline float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (12102203.1616540672 * x + 1064807160.56887296);
    return v.f;
}

static inline float IOU(float XMin1, float YMin1, float XMax1, float YMax1, float XMin2, float YMin2, float XMax2, float YMax2)
{
    float Inter = 0;
    float Total = 0;
    float XMin = 0;
    float YMin = 0;
    float XMax = 0;
    float YMax = 0;
    float Area1 = 0;
    float Area2 = 0;
    float InterWidth = 0;
    float InterHeight = 0;

    XMin = ZQ_MAX(XMin1, XMin2);
    YMin = ZQ_MAX(YMin1, YMin2);
    XMax = ZQ_MIN(XMax1, XMax2);
    YMax = ZQ_MIN(YMax1, YMax2);

    InterWidth = XMax - XMin;
    InterHeight = YMax - YMin;

    InterWidth = (InterWidth >= 0) ? InterWidth : 0;
    InterHeight = (InterHeight >= 0) ? InterHeight : 0;

    Inter = InterWidth * InterHeight;

    Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
    Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);

    Total = Area1 + Area2 - Inter;

    return float(Inter) / float(Total);
}

static float DeQnt2F32(int8_t qnt, int zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

// 仿射变换矩阵结构
struct AffineMatrix {
    float i2d[6];       // 图像到网络输入的变换矩阵
    float d2i[6];       // 网络输入到图像的逆变换矩阵

    // 移除重复的AffineMatrix结构体定义
    // 因为它已经在postprocess.h头文件中定义了
    // 仿射变换矩阵方法实现
    void AffineMatrix::compute(const cv::Size& from, const cv::Size& to) {
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;
        float scale = std::min(scale_x, scale_y);
    
        // 计算仿射变换矩阵
        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;
        
        // 计算逆变换矩阵
        cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
        cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
    }

    void AffineMatrix::affine_project(float x, float y, float* ox, float* oy) {
        *ox = d2i[0] * x + d2i[1] * y + d2i[2];
        *oy = d2i[3] * x + d2i[4] * y + d2i[5];
    }
};

// 删除重复的NMS方法枚举定义

/****** v8hand ****/
GetResultRectYolov8::GetResultRectYolov8()
{
    keypoint_num = 21;  // 手部关键点数量为21
    nmsThresh = 0.2;
    objectThresh = 0.5;  // 降低置信度阈值以增加检测灵敏度
    input_w = 224;
    input_h = 224;
    max_objects = 1024;
    nms_method = NMSMethod::CPU;
}

GetResultRectYolov8::~GetResultRectYolov8()
{
}

float GetResultRectYolov8::sigmoid(float x)
{
    return 1 / (1 + fast_exp(-x));
}

// 仿射变换预处理函数
int GetResultRectYolov8::preprocess_warpAffine(const cv::Mat& image, cv::Mat& img_pre, AffineMatrix& affine_matrix)
{
    // 计算仿射变换矩阵
    cv::Size from_size = image.size();
    cv::Size to_size(input_w, input_h);
    affine_matrix.compute(from_size, to_size);
    
    // 应用仿射变换
    cv::warpAffine(image, img_pre, cv::Mat(2, 3, CV_32F, affine_matrix.i2d), to_size, 
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    // 转换图像格式: BGR -> RGB, 0-255 -> 0.0-1.0
    cv::cvtColor(img_pre, img_pre, cv::COLOR_BGR2RGB);
    img_pre.convertTo(img_pre, CV_32F, 1.0 / 255.0);
    
    // 调整维度顺序: HWC -> CHW
    std::vector<cv::Mat> channels;
    cv::split(img_pre, channels);
    cv::Mat merged(cv::Size(input_w, input_h), CV_32FC3);
    
    for (int c = 0; c < 3; c++) {
        channels[c].copyTo(merged(cv::Rect(0, 0, input_w, input_h)));
    }
    
    img_pre = merged.reshape(1, 1); // 转换为1行，3*H*W列
    
    return 0;
}

// 修复GenerateMeshgrid中的未使用变量警告
int GetResultRectYolov8::GenerateMeshgrid()
{
    meshgrid.clear();
    
    // 为三个不同分辨率的特征图生成网格
    // 移除未使用的strides变量
    int sizes[] = {7, 14, 28};    // 三个特征图的尺寸
    
    for (int i = 0; i < 3; i++) {
        int size = sizes[i];
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                // 存储网格中心坐标 (x+0.5, y+0.5)
                meshgrid.push_back((x + 0.5f) / size);
                meshgrid.push_back((y + 0.5f) / size);
            }
        }
    }
    
    printf("=== yolov8 Meshgrid Generate success! Total grid points: %d\n", (int)meshgrid.size() / 2);
    return 0;
}

// CPU版本的NMS实现
std::vector<DetectRect> cpu_nms(std::vector<DetectRect>& boxes, float threshold) {
    std::sort(boxes.begin(), boxes.end(), [](const DetectRect& a, const DetectRect& b) {
        return a.score > b.score;
    });

    std::vector<DetectRect> output;
    output.reserve(boxes.size());

    std::vector<bool> remove_flags(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (remove_flags[i]) continue;

        output.push_back(boxes[i]);

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (remove_flags[j]) continue;
            
            float iou_value = IOU(boxes[i].xmin, boxes[i].ymin, boxes[i].xmax, boxes[i].ymax,
                                  boxes[j].xmin, boxes[j].ymin, boxes[j].xmax, boxes[j].ymax);
            if (iou_value >= threshold) {
                remove_flags[j] = true;
            }
        }
    }
    return output;
}

// 修改GetConvDetectionResult方法签名以匹配头文件
extern "C" int GetResultRectYolov8::GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects, std::vector<float> &DetectKeyPoints, AffineMatrix& affine_matrix)
{
    int ret = 0;
    
    // 修改：允许单个输出张量
    if (qnt_zp.empty() || qnt_scale.empty()) {
        std::cerr << "Error: No output tensors available" << std::endl;
        return -1;
    }

    std::vector<DetectRect> detectRects;
    
    // 生成网格
    GenerateMeshgrid();
    
    // 修改：获取单个输出张量的参数
    int8_t *output = (int8_t *)pBlob[0];
    int quant_zp = qnt_zp[0];
    float quant_scale = qnt_scale[0];
    
    // 特征图参数：7x7, 14x14, 28x28三个特征图拼接成1029个网格点
    const int total_grids = 1029;  // 总共1029个网格点
    const int channels = 68;       // 通道数为68
    
    // 处理输出张量中的每个网格点
    for (int grid_idx = 0; grid_idx < total_grids; grid_idx++) {
        // 获取网格中心坐标（使用之前生成的网格）
        if (static_cast<size_t>(grid_idx * 2 + 1) >= meshgrid.size()) {
            continue;
        }
        
        // 通道0通常是置信度
        float conf = sigmoid(DeQnt2F32(output[grid_idx], quant_zp, quant_scale));
        
        // 如果置信度高于阈值，处理这个检测
        if (conf > objectThresh) {
            // 通道1-4通常是边界框参数 (cx, cy, w, h)
            float dx = DeQnt2F32(output[grid_idx + total_grids * 1], quant_zp, quant_scale);
            float dy = DeQnt2F32(output[grid_idx + total_grids * 2], quant_zp, quant_scale);
            float dw = DeQnt2F32(output[grid_idx + total_grids * 3], quant_zp, quant_scale);
            float dh = DeQnt2F32(output[grid_idx + total_grids * 4], quant_zp, quant_scale);
            
            // 计算边界框中心和宽高
            float x_center = dx;
            float y_center = dy;
            float width = exp(dw);
            float height = exp(dh);
            
            // 计算边界框角点
            float xmin = (x_center - width / 2);
            float ymin = (y_center - height / 2);
            float xmax = (x_center + width / 2);
            float ymax = (y_center + height / 2);
            
            // 添加边界框大小过滤以避免非常小的框
            float box_width = (xmax - xmin);
            float box_height = (ymax - ymin);
            float min_box_area = 30 * 30;  // 最小框面积
            
            if (box_width * box_height > min_box_area) {
                DetectRect temp;
                temp.xmin = xmin;
                temp.ymin = ymin;
                temp.xmax = xmax;
                temp.ymax = ymax;
                temp.classId = 0;  // 手部检测只有一个类别
                temp.score = conf;
                
                // 提取关键点（通道5-67，共63个通道，对应21个关键点的x/y/score）
                std::vector<KeyPoint> keyPoints;
                for (int kc = 0; kc < keypoint_num; kc++) {
                    int base_channel = 5 + kc * 3;
                    
                    // 确保通道索引不超过范围
                    if (base_channel + 2 >= channels) {
                        // 填充默认值
                        KeyPoint Point;
                        Point.x = 0.0f;
                        Point.y = 0.0f;
                        Point.z = 0.0f;
                        Point.score = 0.0f;
                        keyPoints.push_back(Point);
                        continue;
                    }
                    
                    // 提取每个关键点的x, y坐标和score
                    float kx = DeQnt2F32(output[grid_idx + total_grids * base_channel], quant_zp, quant_scale);
                    float ky = DeQnt2F32(output[grid_idx + total_grids * (base_channel + 1)], quant_zp, quant_scale);
                    float kscore = sigmoid(DeQnt2F32(output[grid_idx + total_grids * (base_channel + 2)], quant_zp, quant_scale));
                    
                    KeyPoint Point;
                    Point.x = kx;
                    Point.y = ky;
                    Point.z = 0.0f;  // 没有z坐标信息
                    Point.score = kscore;
                    
                    keyPoints.push_back(Point);
                }
                
                temp.keyPoints = keyPoints;
                detectRects.push_back(temp);
            }
        }
    }

    // 应用逆变换将检测结果映射回原图
    for (auto& rect : detectRects) {
        float ox1, oy1, ox2, oy2;
        affine_matrix.affine_project(rect.xmin, rect.ymin, &ox1, &oy1);
        affine_matrix.affine_project(rect.xmax, rect.ymax, &ox2, &oy2);
        rect.xmin = ox1;
        rect.ymin = oy1;
        rect.xmax = ox2;
        rect.ymax = oy2;
        
        // 映射关键点回原图
        for (auto& kp : rect.keyPoints) {
            float ox, oy;
            affine_matrix.affine_project(kp.x, kp.y, &ox, &oy);
            kp.x = ox;
            kp.y = oy;
        }
    }

    // 应用非极大值抑制（NMS）
    if (nms_method == NMSMethod::CPU && !detectRects.empty()) {
        detectRects = cpu_nms(detectRects, nmsThresh);
    }
    
    // 处理检测结果
    for (size_t i = 0; i < detectRects.size(); ++i) {
        float xmin1 = detectRects[i].xmin;
        float ymin1 = detectRects[i].ymin;
        float xmax1 = detectRects[i].xmax;
        float ymax1 = detectRects[i].ymax;
        int classId = detectRects[i].classId;
        float score = detectRects[i].score;

        if (classId != -1) {
            // 以格式存储检测结果：classId, score, xmin1, ymin1, xmax1, ymax1
            DetectiontRects.push_back(float(classId));
            DetectiontRects.push_back(float(score));
            DetectiontRects.push_back(float(xmin1));
            DetectiontRects.push_back(float(ymin1));
            DetectiontRects.push_back(float(xmax1));
            DetectiontRects.push_back(float(ymax1));

            // 以格式存储关键点：每个21个关键点的(score, x, y)
            for(size_t kn = 0; kn < static_cast<size_t>(keypoint_num); kn++) {
                if (kn < detectRects[i].keyPoints.size()) {
                    DetectKeyPoints.push_back(float(detectRects[i].keyPoints[kn].score));
                    DetectKeyPoints.push_back(float(detectRects[i].keyPoints[kn].x));
                    DetectKeyPoints.push_back(float(detectRects[i].keyPoints[kn].y));
                } else {
                    // 关键点不足时的处理
                    DetectKeyPoints.push_back(0.0f);
                    DetectKeyPoints.push_back(0.0f);
                    DetectKeyPoints.push_back(0.0f);
                }
            }
        }
    }

    // 修复有符号和无符号整数比较的警告
    for (int grid_idx = 0; grid_idx < total_grids; grid_idx++) {
        // 获取网格中心坐标（使用之前生成的网格）
        if (static_cast<size_t>(grid_idx * 2 + 1) >= meshgrid.size()) {
            continue;
        }
        
        // 通道0通常是置信度
        float conf = sigmoid(DeQnt2F32(output[grid_idx], quant_zp, quant_scale));
        
        // 如果置信度高于阈值，处理这个检测
        if (conf > objectThresh) {
            // 通道1-4通常是边界框参数 (cx, cy, w, h)
            float dx = DeQnt2F32(output[grid_idx + total_grids * 1], quant_zp, quant_scale);
            float dy = DeQnt2F32(output[grid_idx + total_grids * 2], quant_zp, quant_scale);
            float dw = DeQnt2F32(output[grid_idx + total_grids * 3], quant_zp, quant_scale);
            float dh = DeQnt2F32(output[grid_idx + total_grids * 4], quant_zp, quant_scale);
            
            // 计算边界框中心和宽高
            float x_center = dx;
            float y_center = dy;
            float width = exp(dw);
            float height = exp(dh);
            
            // 计算边界框角点
            float xmin = (x_center - width / 2);
            float ymin = (y_center - height / 2);
            float xmax = (x_center + width / 2);
            float ymax = (y_center + height / 2);
            
            // 添加边界框大小过滤以避免非常小的框
            float box_width = (xmax - xmin);
            float box_height = (ymax - ymin);
            float min_box_area = 30 * 30;  // 最小框面积
            
            if (box_width * box_height > min_box_area) {
                DetectRect temp;
                temp.xmin = xmin;
                temp.ymin = ymin;
                temp.xmax = xmax;
                temp.ymax = ymax;
                temp.classId = 0;  // 手部检测只有一个类别
                temp.score = conf;
                
                // 提取关键点（通道5-67，共63个通道，对应21个关键点的x/y/score）
                std::vector<KeyPoint> keyPoints;
                for (int kc = 0; kc < keypoint_num; kc++) {
                    int base_channel = 5 + kc * 3;
                    
                    // 确保通道索引不超过范围
                    if (base_channel + 2 >= channels) {
                        // 填充默认值
                        KeyPoint Point;
                        Point.x = 0.0f;
                        Point.y = 0.0f;
                        Point.z = 0.0f;
                        Point.score = 0.0f;
                        keyPoints.push_back(Point);
                        continue;
                    }
                    
                    // 提取每个关键点的x, y坐标和score
                    float kx = DeQnt2F32(output[grid_idx + total_grids * base_channel], quant_zp, quant_scale);
                    float ky = DeQnt2F32(output[grid_idx + total_grids * (base_channel + 1)], quant_zp, quant_scale);
                    float kscore = sigmoid(DeQnt2F32(output[grid_idx + total_grids * (base_channel + 2)], quant_zp, quant_scale));
                    
                    KeyPoint Point;
                    Point.x = kx;
                    Point.y = ky;
                    Point.z = 0.0f;  // 没有z坐标信息
                    Point.score = kscore;
                    
                    keyPoints.push_back(Point);
                }
                
                temp.keyPoints = keyPoints;
                detectRects.push_back(temp);
            }
        }
    }

    // 修复有符号和无符号整数比较的警告
    for (size_t i = 0; i < detectRects.size(); ++i) {
        float xmin1 = detectRects[i].xmin;
        float ymin1 = detectRects[i].ymin;
        float xmax1 = detectRects[i].xmax;
        float ymax1 = detectRects[i].ymax;
        int classId = detectRects[i].classId;
        float score = detectRects[i].score;

        if (classId != -1) {
            // 以格式存储检测结果：classId, score, xmin1, ymin1, xmax1, ymax1
            DetectiontRects.push_back(float(classId));
            DetectiontRects.push_back(float(score));
            DetectiontRects.push_back(float(xmin1));
            DetectiontRects.push_back(float(ymin1));
            DetectiontRects.push_back(float(xmax1));
            DetectiontRects.push_back(float(ymax1));

            // 以格式存储关键点：每个21个关键点的(score, x, y)
            for(size_t kn = 0; kn < static_cast<size_t>(keypoint_num); kn++) {
                if (kn < detectRects[i].keyPoints.size()) {
                    DetectKeyPoints.push_back(float(detectRects[i].keyPoints[kn].score));
                    DetectKeyPoints.push_back(float(detectRects[i].keyPoints[kn].x));
                    DetectKeyPoints.push_back(float(detectRects[i].keyPoints[kn].y));
                } else {
                    // 关键点不足时的处理
                    DetectKeyPoints.push_back(0.0f);
                    DetectKeyPoints.push_back(0.0f);
                    DetectKeyPoints.push_back(0.0f);
                }
            }
        }
    }

    printf("== Total detected objects after filtering: %d \n", int(float(DetectiontRects.size()) / 6.0));

    return ret;
}