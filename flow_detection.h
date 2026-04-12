#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <semaphore.h>
#include <string>
#include <cfloat>  // 添加这个头文件用于DBL_MAX
#include <cmath>   // 添加这个头文件用于isnan, isinf


const char* get_type_name(rknn_tensor_type type);

const char* get_format_name(rknn_tensor_format fmt);

size_t get_element_size(rknn_tensor_type type);

unsigned char* load_model_file(const char* model_path, int* model_size);

void __f32_to_f16(uint16_t* f16, float* f32, int num);
void __f16_to_f32(float* f32, uint16_t* f16, int num);

void flow_detection(const char* model_path);

// RKNN模型推理接口
struct RKNNModel {
    rknn_context ctx;
    unsigned char* model_data;
    int model_size;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    bool initialized;
    
    RKNNModel() : ctx(0), model_data(nullptr), model_size(0), 
                  input_attrs(nullptr), output_attrs(nullptr), initialized(false) {
        memset(&io_num, 0, sizeof(io_num));
    }
};

// 加载RKNN模型
// 返回0表示成功，负数表示失败
int load_rknn_model(RKNNModel* model, const char* model_path);

// 释放RKNN模型资源
void release_rknn_model(RKNNModel* model);

// 执行推理
// input_data: 输入数据数组，每个元素对应一个输入节点
// input_sizes: 每个输入节点的元素数量数组
// output_data: 输出数据数组（由函数分配内存，调用者需要释放）
// output_sizes: 每个输出节点的元素数量数组（由函数分配，调用者需要释放）
// 返回0表示成功，负数表示失败
int rknn_inference(RKNNModel* model, 
                   const float** input_data, 
                   const int* input_sizes,
                   float*** output_data, 
                   int** output_sizes);

// 示例函数：演示如何使用RKNN模型进行推理
void example_rknn_inference(const char* model_path);