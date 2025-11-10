// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// 注释掉RGA相关的头文件
// #include "RgaUtils.h"
// #include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
// #include "rga.h"
#include "rknn_api.h"
#include <dirent.h>

// 手部关键点连接关系和颜色定义
const std::vector<std::pair<int, int>> HAND_CONNECTIONS = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},  // 拇指
    {0, 5}, {5, 6}, {6, 7}, {7, 8},  // 食指
    {0, 9}, {9, 10}, {10, 11}, {11, 12},  // 中指
    {0, 13}, {13, 14}, {14, 15}, {15, 16},  // 无名指
    {0, 17}, {17, 18}, {18, 19}, {19, 20}   // 小指
};

const std::vector<cv::Scalar> CONNECTION_COLORS = {
    cv::Scalar(0, 0, 255),    // 拇指 - 红色
    cv::Scalar(0, 128, 255),  // 食指 - 橙色
    cv::Scalar(0, 255, 255),  // 中指 - 黄色
    cv::Scalar(0, 255, 0),    // 无名指 - 绿色
    cv::Scalar(255, 0, 0)     // 小指 - 蓝色
};

const cv::Scalar BOX_COLOR = cv::Scalar(255, 255, 0);  // 检测框颜色 - 青色

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

int detect(char *model_path, char *image_path, char *save_image_path)
{
    // 删除未使用的变量
    // int status = 0;
    // char *model_name = NULL;
    // size_t actual_size = 0;
    
    rknn_context ctx;
    size_t actual_size = 0;
    int img_width = 0;
    int img_height = 0;
    int img_channel = 0;
    struct timeval start_time, stop_time, stop_time1;
    int ret;

    printf("Read %s ...\n", image_path);
    cv::Mat src_image = cv::imread(image_path, 1);
    if (!src_image.data)
    {
        printf("cv::imread %s fail!\n", image_path);
        return -1;
    }
    cv::Mat img;
    cv::cvtColor(src_image, img, cv::COLOR_BGR2RGB);

    img_width = img.cols;
    img_height = img.rows;

    printf("img width = %d, img height = %d\n", img_width, img_height);

    /* Create the neural network */
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_path, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    // You may not need resize when src resulotion equals to dst resulotion
    void *resize_buf = nullptr;
    cv::Mat resized_img;

    // 在detect函数中，替换resize部分的代码
    // 从这部分开始修改
    if (img_width != width || img_height != height) {
    printf("resize with OpenCV using affine transformation!\n");
    
    // 使用仿射变换保持图像比例进行resize
    float scale = std::min((float)width / img_width, (float)height / img_height);
    int new_width = static_cast<int>(img_width * scale);
    int new_height = static_cast<int>(img_height * scale);
    
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    
    // 创建目标图像并居中放置调整后的图像
    cv::Mat padded_img(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
    int x_offset = (width - new_width) / 2;
    int y_offset = (height - new_height) / 2;
    resized_img.copyTo(padded_img(cv::Rect(x_offset, y_offset, new_width, new_height)));
    
    inputs[0].buf = (void *)padded_img.data;
} else {
    inputs[0].buf = (void *)img.data;
}
// 修改结束

    gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // 后处理部分
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    int8_t *pblob[9];
    for (int i = 0; i < io_num.n_output; ++i)
    {
        pblob[i] = (int8_t *)outputs[i].buf;
    }

    // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在vector<float>中
    GetResultRectYolov8 PostProcess;
    std::vector<float> DetectiontRects;

    // 将21个关键点按照每个点（score, x, y）的顺序存入
    std::vector<float> DetectKeyPoints;
    PostProcess.GetConvDetectionResult(pblob, out_zps, out_scales, DetectiontRects, DetectKeyPoints);

    gettimeofday(&stop_time1, NULL);
    printf("postprocess run use %f ms\n", (__get_us(stop_time1) - __get_us(stop_time)) / 1000);

    // 修改关键点数量为21
    int KeyPointsNum = 21;
    float pose_score = 0;
    int pose_x = 0, pose_y = 0;
    int NumIndex = 0, Temp = 0;
    
    // 存储每个检测框的关键点坐标，用于绘制连接线
    std::vector<std::vector<cv::Point>> allKeyPoints;
    std::vector<std::vector<float>> allScores;
    
    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] * float(img_width) + 0.5);
        int ymin = int(DetectiontRects[i + 3] * float(img_height) + 0.5);
        int xmax = int(DetectiontRects[i + 4] * float(img_width) + 0.5);
        int ymax = int(DetectiontRects[i + 5] * float(img_height) + 0.5);

        char text1[256];
        sprintf(text1, "%d:%.2f", classId, conf);
        // 使用青色绘制检测框
        rectangle(src_image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), BOX_COLOR, 2);
        putText(src_image, text1, cv::Point(xmin, ymin + 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        
        Temp = 0;
        // 存储当前检测框的关键点
        std::vector<cv::Point> keyPoints;
        std::vector<float> scores;
        
        // 关键点
        for (int k = NumIndex * KeyPointsNum * 3; k < (NumIndex + 1)* KeyPointsNum * 3 ; k += 3)
        {
            pose_score = DetectKeyPoints[k + 0];
            
            if(pose_score > 0.5)
            {
                pose_x = int(DetectKeyPoints[k + 1] * float(img_width) + 0.5);
                pose_y = int(DetectKeyPoints[k + 2] * float(img_height) + 0.5);
                
                // 绘制关键点（使用白色填充圆）
                cv::circle(src_image, cv::Point(pose_x, pose_y), 4, cv::Scalar(255, 255, 255), -1);
                
                // 记录有效的关键点
                keyPoints.push_back(cv::Point(pose_x, pose_y));
                scores.push_back(pose_score);
            } else {
                // 对于置信度低的点，记录为无效点
                keyPoints.push_back(cv::Point(-1, -1));
                scores.push_back(0);
            }
            Temp += 1;
        }
        
        // 保存当前检测框的关键点信息
        allKeyPoints.push_back(keyPoints);
        allScores.push_back(scores);
        NumIndex += 1;
    }
    
    // 绘制关键点连接线
    for (size_t boxIdx = 0; boxIdx < allKeyPoints.size(); ++boxIdx) {
        const auto& keyPoints = allKeyPoints[boxIdx];
        const auto& scores = allScores[boxIdx];
        
        // 为每根手指分配颜色索引
        int colorIndex = 0;
        int connectionCount = 0;
        
        for (const auto& conn : HAND_CONNECTIONS) {
            int startIdx = conn.first;
            int endIdx = conn.second;
            
            // 检查两个点是否都有效
            if (startIdx < keyPoints.size() && endIdx < keyPoints.size() && 
                keyPoints[startIdx].x != -1 && keyPoints[startIdx].y != -1 &&
                keyPoints[endIdx].x != -1 && keyPoints[endIdx].y != -1) {
                
                // 绘制连接线
                cv::line(src_image, keyPoints[startIdx], keyPoints[endIdx], 
                         CONNECTION_COLORS[colorIndex], 2);
            }
            
            // 更新颜色索引（每4个连接为一组，对应一根手指）
            connectionCount++;
            if (connectionCount % 4 == 0) {
                colorIndex = std::min(colorIndex + 1, (int)CONNECTION_COLORS.size() - 1);
            }
        }
    }
    
    imwrite(save_image_path, src_image);

    printf("== obj: %d \n", int(float(DetectiontRects.size()) / 6.0));

    // release
    ret = rknn_destroy(ctx);

    if (model_data) {
        free(model_data);
    }

    if (resize_buf) {
        free(resize_buf);
    }

    return 0;
}

int main(int argc, char **argv)
{
    // 修改为手势检测模型路径
    char model_path[256] = "rk3568/best.rknn";
    char image_path[256] = "8.jpg";
    char save_image_path[256] = "hand_result.jpg";

    detect(model_path, image_path, save_image_path);
    return 0;
}