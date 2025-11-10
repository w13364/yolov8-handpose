#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <opencv2/opencv.hpp>

typedef signed char int8_t;
typedef unsigned int uint32_t;

// 仿射变换矩阵结构
struct AffineMatrix {
    float i2d[6];       // 图像到网络输入的变换矩阵
    float d2i[6];       // 网络输入到图像的逆变换矩阵

    void compute(const cv::Size& from, const cv::Size& to);
    void affine_project(float x, float y, float* ox, float* oy);
};

// 更新关键点结构，增加z坐标
typedef struct
{
    float x;
    float y;
    float z;
    float score;
} KeyPoint;

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int classId;
    std::vector<KeyPoint> keyPoints;
} DetectRect;

// NMS方法枚举
enum class NMSMethod : int {
    CPU = 0,         // 通用CPU版本，用于评估mAP
    FastGPU = 1      // 快速GPU版本，在极端情况下有小的精度损失
};

// yolov8
class GetResultRectYolov8
{
public:
    GetResultRectYolov8();

    ~GetResultRectYolov8();

    int GenerateMeshgrid();

    int GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects, std::vector<float> &DetectKeyPoints, AffineMatrix& affine_matrix);

    float sigmoid(float x);
    
    int preprocess_warpAffine(const cv::Mat& image, cv::Mat& img_pre, AffineMatrix& affine_matrix);

private:
    std::vector<float> meshgrid;

    const int class_num = 1;
    int headNum = 3;

    int input_w = 224;
    int input_h = 224;
    int strides[3] = {8, 16, 32};
    int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};
    // 关键点数量从17改为21
    int keypoint_num = 21;

    float nmsThresh = 0.2;  // 进一步降低NMS阈值，从0.25改为0.2
    float objectThresh = 0.7; // 显著提高置信度阈值，从0.6改为0.7
    int max_objects = 1024; // 最大检测目标数
    NMSMethod nms_method = NMSMethod::CPU; // NMS方法
};

#endif