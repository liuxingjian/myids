#include <iostream>
#include "flow_detection.h"

int main(int argc, char** argv) {
    // 使用 models 文件夹中的模型文件
    const char* model_path = "models/ids_mlp_model.rknn";
    
    // 如果命令行提供了模型路径，使用命令行参数
    if (argc > 1) {
        model_path = argv[1];
    }
    
    std::cout << "========== RKNN 推理示例 ==========" << std::endl;
    std::cout << "模型路径: " << model_path << std::endl;
    std::cout << std::endl;
    
    // 调用示例推理函数
    example_rknn_inference(model_path);
    
    return 0;
}