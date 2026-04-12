#include "flow_detection.h"


// 数据类型转字符串
const char* get_type_name(rknn_tensor_type type) {
    switch(type) {
        case RKNN_TENSOR_FLOAT32: return "FLOAT32";
        case RKNN_TENSOR_FLOAT16: return "FLOAT16";
        case RKNN_TENSOR_INT8: return "INT8";
        case RKNN_TENSOR_UINT8: return "UINT8";
        case RKNN_TENSOR_INT16: return "INT16";
        case RKNN_TENSOR_UINT16: return "UINT16";
        case RKNN_TENSOR_INT32: return "INT32";
        case RKNN_TENSOR_UINT32: return "UINT32";
        case RKNN_TENSOR_INT64: return "INT64";
        default: return "UNKNOWN";
    }
}

// 数据格式转字符串
const char* get_format_name(rknn_tensor_format fmt) {
    switch(fmt) {
        case RKNN_TENSOR_NCHW: return "NCHW";
        case RKNN_TENSOR_NHWC: return "NHWC";
        case RKNN_TENSOR_NC1HWC2: return "NC1HWC2";
        case RKNN_TENSOR_UNDEFINED: return "UNDEFINED";
        default: return "UNKNOWN";
    }
}

// 根据数据类型获取元素大小
size_t get_element_size(rknn_tensor_type type) {
    switch(type) {
        case RKNN_TENSOR_FLOAT32: return sizeof(float);
        case RKNN_TENSOR_FLOAT16: return sizeof(uint16_t);
        case RKNN_TENSOR_INT8: return sizeof(int8_t);
        case RKNN_TENSOR_UINT8: return sizeof(uint8_t);
        case RKNN_TENSOR_INT16: return sizeof(int16_t);
        case RKNN_TENSOR_UINT16: return sizeof(uint16_t);
        case RKNN_TENSOR_INT32: return sizeof(int32_t);
        case RKNN_TENSOR_UINT32: return sizeof(uint32_t);
        case RKNN_TENSOR_INT64: return sizeof(int64_t);
        default: return sizeof(uint8_t);
    }
}

// 加载模型文件
unsigned char* load_model_file(const char* model_path, int* model_size) {
    FILE* file = fopen(model_path, "rb");
    if (!file) {
        printf("无法打开模型文件: %s\n", model_path);
        return nullptr;
    }
    
    fseek(file, 0, SEEK_END);
    *model_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    unsigned char* model_data = (unsigned char*)malloc(*model_size);
    if (!model_data) {
        printf("内存分配失败\n");
        fclose(file);
        return nullptr;
    }
    
    fread(model_data, 1, *model_size, file);
    fclose(file);
    
    printf("成功加载模型: %s (大小: %d bytes)\n", model_path, *model_size);
    return model_data;
}
void __f32_to_f16(uint16_t* f16, float* f32, int num)
{
    float* src = f32;
    uint16_t* dst = f16;
    int i = 0;

    for (; i < num; i++) {
        float in = *src;

        uint32_t fp32 = *((uint32_t *) &in);
        uint32_t t1 = (fp32 & 0x80000000u) >> 16;  /* sign bit. */
        uint32_t t2 = (fp32 & 0x7F800000u) >> 13;  /* Exponent bits */
        uint32_t t3 = (fp32 & 0x007FE000u) >> 13;  /* Mantissa bits, no rounding */
        uint32_t fp16 = 0u;

        if( t2 >= 0x023c00u )
        {
            fp16 = t1 | 0x7BFF;     /* Don't round to infinity. */
        }
        else if( t2 <= 0x01c000u )
        {
            fp16 = t1;
        }
        else
        {
            t2 -= 0x01c000u;
            fp16 = t1 | t2 | t3;
        }

        *dst = (uint16_t) fp16;

        src ++;
        dst ++;
    }
}

void __f16_to_f32(float* f32, uint16_t* f16, int num)
{
    uint16_t* src = f16;
    float* dst = f32;
    int i = 0;

    for (; i < num; i++) {
        uint16_t in = *src;

        int32_t t1;
        int32_t t2;
        int32_t t3;
        float out;

        t1 = in & 0x7fff;         // Non-sign bits
        t2 = in & 0x8000;         // Sign bit
        t3 = in & 0x7c00;         // Exponent

        t1 <<= 13;                // Align mantissa on MSB
        t2 <<= 16;                // Shift sign bit into position

        t1 += 0x38000000;         // Adjust bias

        t1 = (t3 == 0 ? 0 : t1);  // Denormals-as-zero

        t1 |= t2;                 // Re-insert sign bit

        *((uint32_t*)&out) = t1;

        *dst = out;

        src ++;
        dst ++;
    }
}

using namespace std;


const string ADDRESS = "192.168.137.34:1883";  // 订阅方的IP和端口
const string CLIENT_ID = "publisher_client_192.168.137.2";  // 发布方的客户端ID
const string TOPIC = "test/topic";
const string PAYLOAD1 = "Hello from 192.168.137.2!";
const char* PAYLOAD2 = "Second message from 192.168.137.2!";
const int QOS = 1;

// 简单的回调类
// class callback : public virtual mqtt::callback {
// public:
//     void connection_lost(const string& cause) override {
//         cout << "Connection lost: " << cause << endl;
//     }
    
//     void message_arrived(mqtt::const_message_ptr msg) override {
//         cout << "Message arrived: " << msg->get_payload_str() << endl;
//     }
    
//     void connect_complete(const mqtt::token& tok) {
//         cout << "Connection completed successfully!" << endl;
//     }
// };

// int main(int argc, char** argv) {
//     int ret = 0;
//     rknn_context ctx = 0;
//     unsigned char* model_data = nullptr;
//     int model_size = 0;
    
//     // 预声明变量
//     rknn_tensor_attr* input_attrs = nullptr;
//     rknn_tensor_attr* output_attrs = nullptr;
//     rknn_input* inputs = nullptr;
//     rknn_output* outputs = nullptr;
//     uint16_t** input_buffers = nullptr;
//     bool all_match = true;
//     mytest::SubTest msg;
//     sem_t *semaphore = nullptr;
//     sem_t *semaphore_empty = nullptr;
//     int fd = -1;
//     void *mapped_memory = nullptr;
//     string input_string;
    
//     int sample_count = 0;

//     if (argc != 2) {
//         printf("用法: %s <rknn_model_path>\n", argv[0]);
//         printf("示例: %s ./transformer_model.rknn\n", argv[0]);
//         return -1;
//     }

//     // mqtt::client cli(ADDRESS, CLIENT_ID);

//     // callback cb;
//     // cli.set_callback(cb);

//     // auto connOpts = mqtt::connect_options_builder() 
//     //     .keep_alive_interval(chrono::seconds(20))
//     //     .clean_session()
//     //     .finalize();
    
//     // try {
//     //     cli.connect(connOpts);
//     // }
//     // catch (const mqtt::exception& exc) {
//     //     cerr << "Error: " << exc.what() << " ["
//     //         << exc.get_reason_code() << "]" << endl;
//     //     return 1;
//     // }
//     char* model_path = argv[1];
//     printf("========== RKNN输入格式验证程序 ==========\n");

//     // 1. 加载并初始化模型
//     model_data = load_model_file(model_path, &model_size);
//     if (!model_data) return -1;

//     ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
//     if (ret < 0) {
//         printf("rknn_init 失败! ret=%d\n", ret);
//         free(model_data);
//         return -1;
//     }
//     printf("RKNN上下文初始化成功\n");

//     // 2. 查询输入输出节点数量
//     rknn_input_output_num io_num;
//     ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
//     if (ret < 0) {
//         printf("查询输入输出数量失败: %d\n", ret);
//         goto cleanup;
//     }
//     printf("模型有 %d 个输入节点, %d 个输出节点\n", io_num.n_input, io_num.n_output);

//     // 3. 查询输入节点属性
//     input_attrs = new rknn_tensor_attr[io_num.n_input];
//     for (uint32_t i = 0; i < io_num.n_input; i++) {
//         input_attrs[i].index = i;
//         ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
//         if (ret < 0) {
//             printf("查询输入节点 %d 属性失败: %d\n", i, ret);
//             continue;
//         }
        
//         printf("输入节点 %d: 类型=%s, 格式=%s, 维度=[", 
//                i, get_type_name(input_attrs[i].type), get_format_name(input_attrs[i].fmt));
//         for(uint32_t j = 0; j < input_attrs[i].n_dims; j++) {
//             printf("%d", input_attrs[i].dims[j]);
//             if(j < input_attrs[i].n_dims - 1) printf(", ");
//         }
//         printf("], 大小=%d bytes\n", input_attrs[i].size);
//     }

//     // 4. 查询输出节点属性
//     output_attrs = new rknn_tensor_attr[io_num.n_output];
//     for (uint32_t i = 0; i < io_num.n_output; i++) {
//         output_attrs[i].index = i;
//         ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
//         if (ret < 0) {
//             printf("查询输出节点 %d 属性失败: %d\n", i, ret);
//             continue;
//         }
        
//         printf("输出节点 %d: 类型=%s, 格式=%s, 维度=[", 
//                i, get_type_name(output_attrs[i].type), get_format_name(output_attrs[i].fmt));
//         for(uint32_t j = 0; j < output_attrs[i].n_dims; j++) {
//             printf("%d", output_attrs[i].dims[j]);
//             if(j < output_attrs[i].n_dims - 1) printf(", ");
//         }
//         printf("], 大小=%d bytes\n", output_attrs[i].size);
//     }


//     // 5. 输入数据内存初始化
//     inputs = new rknn_input[io_num.n_input];
//     input_buffers = new uint16_t*[io_num.n_input];
    
//     for (uint32_t i = 0; i < io_num.n_input; i++) {
//         input_buffers[i] = nullptr;
//     }


//     char buf[8192];
//     fd = open("/dev/shm/test_mmap",O_RDWR);
//     if(fd < 0) 
//     {
//         cout << "no file" << endl;
//         sem_close(semaphore);
//         return 0;
//     }

//     mapped_memory = (void *)mmap(NULL,8192,PROT_READ,MAP_SHARED,fd,0);
//     if(mapped_memory == MAP_FAILED) {
//         cout << "mmap failed" << endl;
//         close(fd);
//         sem_close(semaphore);
//         return 1;
//     }

//     // 打开信号量
//     semaphore = sem_open("/test_semaphore", 0);
//     semaphore_empty = sem_open("/test_semaphore_empty", 0);
//     if(semaphore == SEM_FAILED) {
//         cout << "无法打开信号量  semaphore" << endl;
//         return 1;
//     }
//     if(semaphore_empty == SEM_FAILED) {
//         cout << "无法打开信号量  semaphore_empty" << endl;
//         return 1;
//     }



//     cout << "信号量打开成功，循环读取数据" << endl;
    
//     while(1)
//     {
//         sample_count++;
        
//         // 清理之前的输入缓冲区，避免内存泄漏
//         if (input_buffers) {
//             for (uint32_t i = 0; i < io_num.n_input; i++) {
//                 if (input_buffers[i]) {
//                     delete[] input_buffers[i];
//                     input_buffers[i] = nullptr;
//                 }
//             }
//         }
        
//         // 等待发送进程的信号
//         if(sem_wait(semaphore) == -1) {
//             cout << "等待信号量失败" << endl;
//             goto cleanup;
//         }
//         cout << "收到信号，开始读取数据..." << endl;

//         memcpy(buf,mapped_memory,5114);
//         sem_post(semaphore_empty);
//         input_string = string(buf,5114);
        
//         if(!msg.ParseFromString(input_string))
//         {
//             cout << "parse failed" << endl;
//             goto cleanup;
//         }





//         for (uint32_t i = 0; i < io_num.n_input; i++) {
//             size_t element_size = get_element_size(input_attrs[i].type);
//             size_t buffer_size = input_attrs[i].n_elems * element_size;
            
//             // 根据实际数据类型分配正确的缓冲区
//             if (input_attrs[i].type == RKNN_TENSOR_FLOAT16) {
//                 input_buffers[i] = new uint16_t[buffer_size / sizeof(uint16_t)];
//             } else if (input_attrs[i].type == RKNN_TENSOR_FLOAT32) {
//                 input_buffers[i] = new uint16_t[buffer_size / sizeof(uint16_t)];
//             } else {
//                 input_buffers[i] = new uint16_t[buffer_size / sizeof(uint16_t)];
//             }
            
//             if (!input_buffers[i]) {
//                 printf("为输入节点 %d 分配内存失败\n", i);
//                 goto cleanup;
//             }
            
//             // 清零缓冲区，确保没有随机数据
//             memset(input_buffers[i], 0, buffer_size);
            
//             // 安全检查：确保索引在有效范围内
//             if(i < (uint32_t)msg.arr_size())
//             {
//                 if (input_attrs[i].type == RKNN_TENSOR_FLOAT16) {
//                     // 准备float32数组用于批量转换
//                     float* temp_float_array = new float[input_attrs[i].n_elems];
//                     memset(temp_float_array, 0, input_attrs[i].n_elems * sizeof(float));
                    
//                     // 填充数据并检查有效性
//                     for(uint32_t j = 0; j < input_attrs[i].n_elems && j < (uint32_t)msg.arr(i).value_size(); j++)
//                     {
//                         float float_val = msg.arr(i).value(j);
                        
//                         // 检查数据有效性
//                         if (isnan(float_val) || isinf(float_val)) {
//                             printf("警告：输入节点 %d 位置 %d 包含无效值: %f，使用0.0替代\n", i, j, float_val);
//                             float_val = 0.0f;
//                         }
                        
//                         temp_float_array[j] = float_val;
//                     }
                    
//                     // 使用新的批量转换函数
//                     uint16_t* buffer = (uint16_t*)input_buffers[i];
//                     __f32_to_f16(buffer, temp_float_array, input_attrs[i].n_elems);
                    
//                     // 释放临时数组
//                     delete[] temp_float_array;
//                 } else {
//                     // 处理非FLOAT16类型
//                     for(uint32_t j = 0; j < input_attrs[i].n_elems && j < (uint32_t)msg.arr(i).value_size(); j++)
//                     {
//                         float float_val = msg.arr(i).value(j);
                        
//                         // 检查数据有效性
//                         if (isnan(float_val) || isinf(float_val)) {
//                             printf("警告：输入节点 %d 位置 %d 包含无效值: %f，使用0.0替代\n", i, j, float_val);
//                             float_val = 0.0f;
//                         }
                        
//                         // 直接拷贝FLOAT32数据
//                         memcpy((char*)input_buffers[i] + j * element_size, &float_val, element_size);
//                     }
//                 }
//             }
//             else if(i - (uint32_t)msg.arr_size() < (uint32_t)msg.matrix_size())
//             {
//                 int matrix_index = i - msg.arr_size();
                
//                 if (input_attrs[i].type == RKNN_TENSOR_FLOAT16) {
//                     // 准备float32数组用于批量转换
//                     float* temp_float_array = new float[input_attrs[i].n_elems];
//                     memset(temp_float_array, 0, input_attrs[i].n_elems * sizeof(float));
                    
//                     // 填充数据并检查有效性
//                     for(uint32_t j = 0; j < input_attrs[i].n_elems && j < (uint32_t)msg.matrix(matrix_index).value_size(); j++)
//                     {
//                         float float_val = msg.matrix(matrix_index).value(j);
                        
//                         // 检查数据有效性
//                         if (isnan(float_val) || isinf(float_val)) {
//                             printf("警告：输入节点 %d 位置 %d 包含无效值: %f，使用0.0替代\n", i, j, float_val);
//                             float_val = 0.0f;
//                         }
                        
//                         temp_float_array[j] = float_val;
//                     }
                    
//                     // 使用新的批量转换函数
//                     uint16_t* buffer = (uint16_t*)input_buffers[i];
//                     __f32_to_f16(buffer, temp_float_array, input_attrs[i].n_elems);
                    
//                     // 释放临时数组
//                     delete[] temp_float_array;
//                 } else {
//                     // 处理非FLOAT16类型
//                     for(uint32_t j = 0; j < input_attrs[i].n_elems && j < (uint32_t)msg.matrix(matrix_index).value_size(); j++)
//                     {
//                         float float_val = msg.matrix(matrix_index).value(j);
                        
//                         // 检查数据有效性
//                         if (isnan(float_val) || isinf(float_val)) {
//                             printf("警告：输入节点 %d 位置 %d 包含无效值: %f，使用0.0替代\n", i, j, float_val);
//                             float_val = 0.0f;
//                         }
                        
//                         // 直接拷贝FLOAT32数据
//                         memcpy((char*)input_buffers[i] + j * element_size, &float_val, element_size);
//                     }
//                 }
//             }
//             else {
//                 // 对于超出数据范围的输入节点，使用默认值0
//                 printf("警告：输入节点 %d 超出数据范围，使用默认值0\n", i);
//                 if (input_attrs[i].type == RKNN_TENSOR_FLOAT16) {
//                     // 准备全0的float32数组用于批量转换
//                     float* temp_float_array = new float[input_attrs[i].n_elems];
//                     memset(temp_float_array, 0, input_attrs[i].n_elems * sizeof(float));
                    
//                     // 使用新的批量转换函数
//                     uint16_t* buffer = (uint16_t*)input_buffers[i];
//                     __f32_to_f16(buffer, temp_float_array, input_attrs[i].n_elems);
                    
//                     // 释放临时数组
//                     delete[] temp_float_array;
//                 } else {
//                     memset(input_buffers[i], 0, buffer_size);
//                 }
//             }
    
//             inputs[i].index = i;
//             inputs[i].buf = input_buffers[i];
//             inputs[i].size = (uint32_t)buffer_size;
//             inputs[i].pass_through = 0;
//             inputs[i].type = input_attrs[i].type;
//             inputs[i].fmt = input_attrs[i].fmt;
//         }

//         // 添加调试信息：显示输入数据
//         printf("调试信息 - 输入数据样本 (样本 #%d):\n", sample_count);
//         for (uint32_t i = 0; i < min(3u, io_num.n_input); i++) {  // 显示前3个输入节点
//             printf("  输入节点 %d: ", i);
//             if (input_attrs[i].type == RKNN_TENSOR_FLOAT16) {
//                 uint16_t* buffer = (uint16_t*)input_buffers[i];
//                 // 使用新的转换函数进行调试显示
//                 float* temp_debug_array = new float[min(5u, input_attrs[i].n_elems)];
//                 __f16_to_f32(temp_debug_array, buffer, min(5u, input_attrs[i].n_elems));
//                 for(uint32_t j = 0; j < min(5u, input_attrs[i].n_elems); j++) {
//                     printf("%.6f ", temp_debug_array[j]);
//                 }
//                 delete[] temp_debug_array;
//             } else {
//                 float* buffer = (float*)input_buffers[i];
//                 for(uint32_t j = 0; j < min(5u, input_attrs[i].n_elems); j++) {
//                     printf("%.6f ", buffer[j]);
//                 }
//             }
//             printf("\n");
//         }
        
//         // 显示原始protobuf数据信息
//         printf("原始数据信息: arr_size=%d, matrix_size=%d\n", msg.arr_size(), msg.matrix_size());
//         if (msg.arr_size() > 0) {
//             printf("  arr[0] 前5个值: ");
//             for(int j = 0; j < min(5, msg.arr(0).value_size()); j++) {
//                 printf("%.6f ", msg.arr(0).value(j));
//             }
//             printf("\n");
//         }
//         if (msg.matrix_size() > 0) {
//             printf("  matrix[0] 前5个值: ");
//             for(int j = 0; j < min(5, msg.matrix(0).value_size()); j++) {
//                 printf("%.6f ", msg.matrix(0).value(j));
//             }
//             printf("\n");
//         }

//         // 7. 设置输入并执行推理
//         ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
//         if (ret < 0) {
//             printf("设置输入失败! ret=%d\n", ret);
//             goto cleanup;
//         }
//         printf("输入设置成功\n");

//         ret = rknn_run(ctx, NULL);
//         if (ret < 0) {
//             printf("推理执行失败! ret=%d\n", ret);
//             goto cleanup;
//         }
//         printf("推理执行成功\n");

//         // 8. 获取输出结果
//         outputs = new rknn_output[io_num.n_output];
//         for (uint32_t i = 0; i < io_num.n_output; i++) {
//             outputs[i].index = i;
//             outputs[i].want_float = 1;  // 获取浮点数据
//             outputs[i].is_prealloc = 0; // 让RKNN分配内存
//         }

//         // 逐个获取输出
//         for (uint32_t i = 0; i < io_num.n_output; i++) {
//             ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
//             if (ret < 0) {
//                 printf("获取输出节点 %d 失败! ret=%d\n", i, ret);
//                 goto cleanup;
//             }
//         }
        
//         // 9. 输出结果
//         printf("\n========== 模型输出结果 ==========\n");
//         for (uint32_t i = 0; i < io_num.n_output; i++) {
//             printf("输出节点 %d:\n", i);
//             printf("  类型: %s\n", get_type_name(output_attrs[i].type));
//             printf("  格式: %s\n", get_format_name(output_attrs[i].fmt));
//             printf("  维度: [");
//             for(uint32_t j = 0; j < output_attrs[i].n_dims; j++) {
//                 printf("%d", output_attrs[i].dims[j]);
//                 if(j < output_attrs[i].n_dims - 1) printf(", ");
//             }
//             printf("]\n");
//             printf("  大小: %d bytes\n", outputs[i].size);
//             printf("  数据: ");
            
//             // 打印所有元素（对于小尺寸输出）或前几个元素（对于大尺寸输出）
//             int print_count = (output_attrs[i].n_elems <= 10) ? output_attrs[i].n_elems : 5;
//             if (output_attrs[i].type == RKNN_TENSOR_FLOAT32) {
//                 float* data = (float*)outputs[i].buf;
//                 for (uint32_t j = 0; j < print_count && j < output_attrs[i].n_elems; j++) {
//                     printf("%.6f", data[j]);
//                     if (j < print_count - 1 && j < output_attrs[i].n_elems - 1) printf(", ");
//                 }
//             } else if (output_attrs[i].type == RKNN_TENSOR_FLOAT16) {
//                 // 检查输出设置和实际数据大小
//                 printf("调试: 输出节点%d - want_float=%d, 期望大小=%d bytes, 实际大小=%d bytes\n", 
//                        i, outputs[i].want_float, output_attrs[i].n_elems * 4, outputs[i].size);
                
//                 if (outputs[i].want_float == 1) {
//                     // RKNN应该已经转换为float32，检查实际数据大小
//                     if (outputs[i].size == output_attrs[i].n_elems * 4) {
//                         // 确实是float32数据
//                         printf("调试: RKNN已转换为FLOAT32\n");
//                         float* data = (float*)outputs[i].buf;
//                         for (uint32_t j = 0; j < print_count && j < output_attrs[i].n_elems; j++) {
//                             printf("%.6f", data[j]);
//                             if (j < print_count - 1 && j < output_attrs[i].n_elems - 1) printf(", ");
//                         }
//                     } else {
//                         // 仍然是float16数据，手动转换
//                         printf("调试: 仍是FLOAT16数据，手动转换\n");
//                         uint16_t* f16_data = (uint16_t*)outputs[i].buf;
//                         float* f32_data = new float[print_count];
//                         __f16_to_f32(f32_data, f16_data, print_count);
                        
//                         for (uint32_t j = 0; j < print_count && j < output_attrs[i].n_elems; j++) {
//                             printf("%.6f", f32_data[j]);
//                             if (j < print_count - 1 && j < output_attrs[i].n_elems - 1) printf(", ");
//                         }
                        
//                         delete[] f32_data;
//                     }
//                 } else {
//                     // 原始FLOAT16数据，需要手动转换
//                     printf("调试: want_float=0，手动转换FLOAT16\n");
//                     uint16_t* f16_data = (uint16_t*)outputs[i].buf;
//                     float* f32_data = new float[print_count];
//                     __f16_to_f32(f32_data, f16_data, print_count);
                    
//                     for (uint32_t j = 0; j < print_count && j < output_attrs[i].n_elems; j++) {
//                         printf("%.6f", f32_data[j]);
//                         if (j < print_count - 1 && j < output_attrs[i].n_elems - 1) printf(", ");
//                     }
                    
//                     delete[] f32_data;
//                 }
//             }
//             if (output_attrs[i].n_elems > print_count) {
//                 printf(" ... (共 %d 个元素)", output_attrs[i].n_elems);
//             }
//             printf("\n\n");
//         }


        
//         // 重要：释放输出内存，避免内存复用问题
//         if (outputs) {
//             rknn_outputs_release(ctx, io_num.n_output, outputs);
//             delete[] outputs;
//             outputs = nullptr;  // 重置指针
//         }
//     }
    


    


//     // // 6. 验证输入一致性
//     // all_match = true;
//     // for (uint32_t i = 0; i < io_num.n_input; i++) {
//     //     if (inputs[i].type != input_attrs[i].type || 
//     //         inputs[i].fmt != input_attrs[i].fmt ||
//     //         inputs[i].size != input_attrs[i].n_elems * get_element_size(input_attrs[i].type)) {
//     //         printf("输入节点 %d 参数不匹配\n", i);
//     //         all_match = false;
//     //     }
//     // }
    
//     // if (all_match) {
//     //     printf("所有输入参数匹配\n");
//     // } else {
//     //     printf("发现输入参数不匹配\n");
//     // }


// cleanup:
//     // 清理资源
    
//     if (mapped_memory != nullptr && mapped_memory != MAP_FAILED) {
//         munmap(mapped_memory, 8192);
//     }
//     if (fd >= 0) {
//         close(fd);
//     }
//     if (semaphore != nullptr) {
//         sem_close(semaphore);
//     }
    
//     if (outputs) {
//         rknn_outputs_release(ctx, io_num.n_output, outputs);
//         delete[] outputs;
//     }
    
//     if (input_buffers) {
//         for (uint32_t i = 0; i < io_num.n_input; i++) {
//             if (input_buffers[i]) delete[] input_buffers[i];
//         }
//         delete[] input_buffers;
//     }
    
//     if (inputs) delete[] inputs;
//     if (input_attrs) delete[] input_attrs;
//     if (output_attrs) delete[] output_attrs;
//     if (ctx) rknn_destroy(ctx);
//     if (model_data) free(model_data);
    
//     printf("========== 程序结束 ==========\n");
//     return ret >= 0 ? 0 : -1;
// }

// ========== RKNN模型推理接口实现 ==========

int load_rknn_model(RKNNModel* model, const char* model_path) {
    if (!model || !model_path) {
        printf("错误: 无效的参数\n");
        return -1;
    }
    
    if (model->initialized) {
        printf("警告: 模型已经初始化，先释放旧模型\n");
        release_rknn_model(model);
    }
    
    // 1. 加载模型文件
    model->model_data = load_model_file(model_path, &model->model_size);
    if (!model->model_data) {
        return -1;
    }
    
    // 2. 初始化RKNN上下文
    int ret = rknn_init(&model->ctx, model->model_data, model->model_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init 失败! ret=%d\n", ret);
        free(model->model_data);
        model->model_data = nullptr;
        return -1;
    }
    printf("RKNN上下文初始化成功\n");
    
    // 2. 查询输入输出节点数量
    ret = rknn_query(model->ctx, RKNN_QUERY_IN_OUT_NUM, &model->io_num, sizeof(model->io_num));
    if (ret < 0) {
        printf("查询输入输出数量失败: %d\n", ret);
        rknn_destroy(model->ctx);
        free(model->model_data);
        model->model_data = nullptr;
        return -1;
    }
    printf("模型有 %d 个输入节点, %d 个输出节点\n", model->io_num.n_input, model->io_num.n_output);
    
    // 3. 查询输入节点属性
    model->input_attrs = new rknn_tensor_attr[model->io_num.n_input];
    for (uint32_t i = 0; i < model->io_num.n_input; i++) {
        model->input_attrs[i].index = i;
        ret = rknn_query(model->ctx, RKNN_QUERY_INPUT_ATTR, &model->input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("查询输入节点 %d 属性失败: %d\n", i, ret);
            delete[] model->input_attrs;
            rknn_destroy(model->ctx);
            free(model->model_data);
            model->model_data = nullptr;
            return -1;
        }
        
        printf("输入节点 %d: 类型=%s, 格式=%s, 维度=[", 
               i, get_type_name(model->input_attrs[i].type), get_format_name(model->input_attrs[i].fmt));
        for(uint32_t j = 0; j < model->input_attrs[i].n_dims; j++) {
            printf("%d", model->input_attrs[i].dims[j]);
            if(j < model->input_attrs[i].n_dims - 1) printf(", ");
        }
        printf("], 大小=%d bytes, 元素数=%d\n", model->input_attrs[i].size, model->input_attrs[i].n_elems);
    }
    
    // 4. 查询输出节点属性
    model->output_attrs = new rknn_tensor_attr[model->io_num.n_output];
    for (uint32_t i = 0; i < model->io_num.n_output; i++) {
        model->output_attrs[i].index = i;
        ret = rknn_query(model->ctx, RKNN_QUERY_OUTPUT_ATTR, &model->output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("查询输出节点 %d 属性失败: %d\n", i, ret);
            delete[] model->input_attrs;
            delete[] model->output_attrs;
            rknn_destroy(model->ctx);
            free(model->model_data);
            model->model_data = nullptr;
            return -1;
        }
        
        printf("输出节点 %d: 类型=%s, 格式=%s, 维度=[", 
               i, get_type_name(model->output_attrs[i].type), get_format_name(model->output_attrs[i].fmt));
        for(uint32_t j = 0; j < model->output_attrs[i].n_dims; j++) {
            printf("%d", model->output_attrs[i].dims[j]);
            if(j < model->output_attrs[i].n_dims - 1) printf(", ");
        }
        printf("], 大小=%d bytes, 元素数=%d\n", model->output_attrs[i].size, model->output_attrs[i].n_elems);
    }
    
    model->initialized = true;
    printf("模型加载完成\n");
    return 0;
}

void release_rknn_model(RKNNModel* model) {
    if (!model) return;
    
    if (model->ctx) {
        rknn_destroy(model->ctx);
        model->ctx = 0;
    }
    
    if (model->model_data) {
        free(model->model_data);
        model->model_data = nullptr;
    }
    
    if (model->input_attrs) {
        delete[] model->input_attrs;
        model->input_attrs = nullptr;
    }
    
    if (model->output_attrs) {
        delete[] model->output_attrs;
        model->output_attrs = nullptr;
    }
    
    model->model_size = 0;
    model->initialized = false;
    memset(&model->io_num, 0, sizeof(model->io_num));
}

int rknn_inference(RKNNModel* model, 
                   const float** input_data, 
                   const int* input_sizes,
                   float*** output_data, 
                   int** output_sizes) {
    if (!model || !model->initialized) {
        printf("错误: 模型未初始化\n");
        return -1;
    }
    
    if (!input_data || !input_sizes || !output_data || !output_sizes) {
        printf("错误: 无效的参数\n");
        return -1;
    }
    
    int ret = 0;
    rknn_input* inputs = nullptr;
    rknn_output* outputs = nullptr;
    uint16_t** input_buffers = nullptr;
    
    // 1. 准备输入数据
    inputs = new rknn_input[model->io_num.n_input];
    input_buffers = new uint16_t*[model->io_num.n_input];
    
    for (uint32_t i = 0; i < model->io_num.n_input; i++) {
        input_buffers[i] = nullptr;
        
        // 检查输入大小是否匹配
        if (input_sizes[i] != (int)model->input_attrs[i].n_elems) {
            printf("错误: 输入节点 %d 大小不匹配，期望 %d，实际 %d\n", 
                   i, model->input_attrs[i].n_elems, input_sizes[i]);
            ret = -1;
            goto cleanup;
        }
        
        size_t element_size = get_element_size(model->input_attrs[i].type);
        size_t buffer_size = model->input_attrs[i].n_elems * element_size;
        
        // 分配输入缓冲区
        input_buffers[i] = new uint16_t[(buffer_size + sizeof(uint16_t) - 1) / sizeof(uint16_t)];
        if (!input_buffers[i]) {
            printf("为输入节点 %d 分配内存失败\n", i);
            ret = -1;
            goto cleanup;
        }
        
        // 根据数据类型转换输入数据
        if (model->input_attrs[i].type == RKNN_TENSOR_FLOAT16) {
            // 转换为FLOAT16
            __f32_to_f16((uint16_t*)input_buffers[i], (float*)input_data[i], model->input_attrs[i].n_elems);
        } else if (model->input_attrs[i].type == RKNN_TENSOR_FLOAT32) {
            // 直接拷贝FLOAT32数据
            memcpy(input_buffers[i], input_data[i], buffer_size);
        } else {
            // 其他类型直接拷贝
            memcpy(input_buffers[i], input_data[i], buffer_size);
        }
        
        // 设置输入
        inputs[i].index = i;
        inputs[i].buf = input_buffers[i];
        inputs[i].size = (uint32_t)buffer_size;
        
        // 如果格式是 UNDEFINED，使用 pass_through 模式，让 RKNN 自动处理
        if (model->input_attrs[i].fmt == RKNN_TENSOR_UNDEFINED) {
            inputs[i].pass_through = 1;  // 直接传递，不进行格式转换
            // pass_through = 1 时，type 和 fmt 不需要设置
            inputs[i].type = model->input_attrs[i].type;  // 仍然设置 type 以防万一
            inputs[i].fmt = RKNN_TENSOR_UNDEFINED;  // 保持 UNDEFINED
        } else {
            inputs[i].pass_through = 0;  // 需要格式转换
            inputs[i].type = model->input_attrs[i].type;
            inputs[i].fmt = model->input_attrs[i].fmt;
        }
        
        printf("输入节点 %d: index=%d, buf=%p, size=%u, pass_through=%d, type=%d, fmt=%d\n",
               i, inputs[i].index, inputs[i].buf, inputs[i].size, 
               inputs[i].pass_through, inputs[i].type, inputs[i].fmt);
    }
    
    // 2. 设置输入并执行推理
    printf("设置输入数据...\n");
    ret = rknn_inputs_set(model->ctx, model->io_num.n_input, inputs);
    if (ret < 0) {
        printf("设置输入失败! ret=%d\n", ret);
        goto cleanup;
    }
    printf("输入设置成功\n");
    
    printf("执行推理...\n");
    ret = rknn_run(model->ctx, NULL);
    if (ret < 0) {
        printf("推理执行失败! ret=%d\n", ret);
        goto cleanup;
    }
    printf("推理执行成功\n");
    
    // 3. 获取输出结果
    printf("准备获取输出，输出节点数: %d\n", model->io_num.n_output);
    if (model->io_num.n_output == 0) {
        printf("错误: 输出节点数为0\n");
        ret = -1;
        goto cleanup;
    }
    
    outputs = new rknn_output[model->io_num.n_output];
    memset(outputs, 0, sizeof(rknn_output) * model->io_num.n_output);
    for (uint32_t i = 0; i < model->io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;  // 获取浮点数据
        outputs[i].is_prealloc = 0; // 让RKNN分配内存
        outputs[i].buf = nullptr;   // 初始化为null
        outputs[i].size = 0;         // 初始化为0
    }
    
    printf("调用 rknn_outputs_get...\n");
    ret = rknn_outputs_get(model->ctx, model->io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("获取输出失败! ret=%d\n", ret);
        // 在错误情况下，确保 outputs 被正确清理
        delete[] outputs;
        outputs = nullptr;
        goto cleanup;
    }
    printf("rknn_outputs_get 成功，开始处理输出数据...\n");
    
    // 4. 分配输出数据数组并转换数据
    printf("分配输出数据数组，输出节点数: %d\n", model->io_num.n_output);
    *output_data = new float*[model->io_num.n_output];
    *output_sizes = new int[model->io_num.n_output];
    
    // 初始化输出数据指针
    for (uint32_t i = 0; i < model->io_num.n_output; i++) {
        (*output_data)[i] = nullptr;
        (*output_sizes)[i] = 0;
    }
    
    printf("处理 %d 个输出节点...\n", model->io_num.n_output);
    for (uint32_t i = 0; i < model->io_num.n_output; i++) {
        printf("处理输出节点 %d: 元素数=%d, buf=%p, size=%d\n", 
               i, model->output_attrs[i].n_elems, outputs[i].buf, outputs[i].size);
        (*output_sizes)[i] = model->output_attrs[i].n_elems;
        
        // 分配输出缓冲区
        (*output_data)[i] = new float[model->output_attrs[i].n_elems];
        if (!(*output_data)[i]) {
            printf("为输出节点 %d 分配内存失败\n", i);
            ret = -1;
            // 清理已分配的输出数据
            for (uint32_t j = 0; j < i; j++) {
                if ((*output_data)[j]) {
                    delete[] (*output_data)[j];
                }
            }
            delete[] *output_data;
            delete[] *output_sizes;
            *output_data = nullptr;
            *output_sizes = nullptr;
            goto cleanup;
        }
        
        // 检查输出缓冲区是否有效
        if (!outputs[i].buf || outputs[i].size == 0) {
            printf("警告: 输出节点 %d 的缓冲区无效 (buf=%p, size=%d)\n", 
                   i, outputs[i].buf, outputs[i].size);
            // 使用零填充
            memset((*output_data)[i], 0, model->output_attrs[i].n_elems * sizeof(float));
        } else if (outputs[i].want_float == 1 && outputs[i].size == model->output_attrs[i].n_elems * sizeof(float)) {
            // RKNN已经转换为float32
            memcpy((*output_data)[i], outputs[i].buf, outputs[i].size);
        } else if (model->output_attrs[i].type == RKNN_TENSOR_FLOAT16) {
            // 需要从FLOAT16转换
            if (outputs[i].size >= model->output_attrs[i].n_elems * sizeof(uint16_t)) {
                __f16_to_f32((*output_data)[i], (uint16_t*)outputs[i].buf, model->output_attrs[i].n_elems);
            } else {
                printf("警告: 输出节点 %d 的FLOAT16数据大小不匹配\n", i);
                memset((*output_data)[i], 0, model->output_attrs[i].n_elems * sizeof(float));
            }
        } else if (model->output_attrs[i].type == RKNN_TENSOR_FLOAT32) {
            // 直接拷贝FLOAT32数据
            size_t expected_size = model->output_attrs[i].n_elems * sizeof(float);
            size_t copy_size = (outputs[i].size < expected_size) ? outputs[i].size : expected_size;
            memcpy((*output_data)[i], outputs[i].buf, copy_size);
        } else {
            // 其他类型，尝试直接拷贝
            size_t expected_size = model->output_attrs[i].n_elems * get_element_size(model->output_attrs[i].type);
            size_t copy_size = (outputs[i].size < expected_size) ? outputs[i].size : expected_size;
            memcpy((*output_data)[i], outputs[i].buf, copy_size);
        }
    }
    
    // 释放RKNN输出内存
    rknn_outputs_release(model->ctx, model->io_num.n_output, outputs);
    
cleanup:
    // 清理输入缓冲区
    if (input_buffers) {
        for (uint32_t i = 0; i < model->io_num.n_input; i++) {
            if (input_buffers[i]) {
                delete[] input_buffers[i];
            }
        }
        delete[] input_buffers;
    }
    
    if (inputs) {
        delete[] inputs;
    }
    
    // 注意：如果已经调用了 rknn_outputs_release，outputs 的内存已经被 RKNN 释放
    // 但我们仍然需要释放 outputs 数组本身
    if (outputs) {
        // 只有在没有调用 rknn_outputs_release 的情况下才需要清理
        // 如果 ret >= 0，说明已经调用了 rknn_outputs_release，只需要删除数组
        // 如果 ret < 0，可能需要清理 buf，但通常 RKNN 会处理
        delete[] outputs;
        outputs = nullptr;
    }
    
    return ret >= 0 ? 0 : -1;
}

// ========== 示例：使用RKNN模型进行推理 ==========
// 这个函数展示了如何加载模型并使用模拟数据进行推理
void example_rknn_inference(const char* model_path) {
    RKNNModel model;
    
    // 1. 加载模型
    printf("========== 加载RKNN模型 ==========\n");
    if (load_rknn_model(&model, model_path) != 0) {
        printf("模型加载失败\n");
        return;
    }
    
    // 2. 准备模拟输入数据
    printf("\n========== 准备输入数据 ==========\n");
    const float** input_data = new const float*[model.io_num.n_input];
    int* input_sizes = new int[model.io_num.n_input];
    
    for (uint32_t i = 0; i < model.io_num.n_input; i++) {
        int n_elems = model.input_attrs[i].n_elems;
        input_sizes[i] = n_elems;
        
        // 分配并初始化模拟数据（这里使用简单的递增序列作为示例）
        float* data = new float[n_elems];
        for (int j = 0; j < n_elems; j++) {
            data[j] = (float)j / 100.0f;  // 模拟数据：0.0, 0.01, 0.02, ...
        }
        input_data[i] = data;
        
        printf("输入节点 %d: %d 个元素\n", i, n_elems);
    }
    
    // 3. 执行推理
    printf("\n========== 执行推理 ==========\n");
    float** output_data = nullptr;
    int* output_sizes = nullptr;
    
    if (rknn_inference(&model, input_data, input_sizes, &output_data, &output_sizes) != 0) {
        printf("推理失败\n");
        goto cleanup_example;
    }
    
    // 4. 显示输出结果
    printf("\n========== 推理结果 ==========\n");
    if (!output_data || !output_sizes) {
        printf("错误: 输出数据为空\n");
        goto cleanup_example;
    }
    
    printf("输出节点数: %d\n", model.io_num.n_output);
    for (uint32_t i = 0; i < model.io_num.n_output; i++) {
        if (i >= (uint32_t)model.io_num.n_output || !output_data[i] || output_sizes[i] <= 0) {
            printf("错误: 输出节点 %d 数据无效\n", i);
            continue;
        }
        printf("输出节点 %d: %d 个元素\n", i, output_sizes[i]);
        printf("  前10个值: ");
        int print_count = (output_sizes[i] < 10) ? output_sizes[i] : 10;
        for (int j = 0; j < print_count; j++) {
            printf("%.6f ", output_data[i][j]);
        }
        if (output_sizes[i] > 10) {
            printf("... (共 %d 个元素)", output_sizes[i]);
        }
        printf("\n");
    }
    
cleanup_example:
    // 5. 清理资源
    printf("\n========== 清理资源 ==========\n");
    
    // 释放输入数据
    for (uint32_t i = 0; i < model.io_num.n_input; i++) {
        delete[] const_cast<float*>(input_data[i]);
    }
    delete[] input_data;
    delete[] input_sizes;
    
    // 释放输出数据
    if (output_data) {
        for (uint32_t i = 0; i < model.io_num.n_output; i++) {
            if (output_data[i]) {
                delete[] output_data[i];
            }
        }
        delete[] output_data;
    }
    if (output_sizes) {
        delete[] output_sizes;
    }
    
    // 释放模型
    release_rknn_model(&model);
    
    printf("示例完成\n");
}
