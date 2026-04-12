import numpy as np
import time
import onnxruntime as ort

def main():
    # 模型路径
    model_path = "/home/lemon/zzu/ai-ids/models/DBGT_v1.onnx"
    
    print("=" * 60)
    print("ONNX Model Inference Test")
    print("=" * 60)
    
    # 1. 加载 ONNX 模型
    print("\n--> Loading ONNX model")
    print(f"Model path: {model_path}")
    
    try:
        # 创建推理会话
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # 2. 获取模型输入输出信息
    print("\n--> Model information")
    print(f"Inputs: {len(session.get_inputs())}")
    for i, input_meta in enumerate(session.get_inputs()):
        print(f"  Input {i}:")
        print(f"    Name: {input_meta.name}")
        print(f"    Shape: {input_meta.shape}")
        print(f"    Type: {input_meta.type}")
    
    print(f"\nOutputs: {len(session.get_outputs())}")
    for i, output_meta in enumerate(session.get_outputs()):
        print(f"  Output {i}:")
        print(f"    Name: {output_meta.name}")
        print(f"    Shape: {output_meta.shape}")
        print(f"    Type: {output_meta.type}")
    
    # 3. 准备输入数据
    print("\n--> Preparing input data")
    input_dict = {}
    
    for input_meta in session.get_inputs():
        input_name = input_meta.name
        input_shape = input_meta.shape
        
        # 处理动态维度（用 1 替换 None、-1 或字符串）
        processed_shape = []
        for dim in input_shape:
            if dim is None or dim == -1:
                processed_shape.append(1)
            elif isinstance(dim, str):
                # 字符串维度（动态维度），使用默认值 1
                processed_shape.append(1)
            elif isinstance(dim, int):
                processed_shape.append(dim)
            else:
                # 其他类型，尝试转换为整数，失败则使用 1
                try:
                    processed_shape.append(int(dim))
                except:
                    processed_shape.append(1)
        
        # 创建随机输入数据
        input_data = np.random.randn(*processed_shape).astype(np.float32)
        input_dict[input_name] = input_data
        
        print(f"  {input_name}: shape={input_data.shape}, dtype={input_data.dtype}, size={input_data.size}")
    
    # 4. 执行推理并记录时间
    print("\n--> Running inference")
    
    # 预热（可选，让模型初始化）
    print("Warming up...")
    try:
        _ = session.run(None, input_dict)
    except Exception as e:
        print(f"Warmup failed: {e}")
        return
    
    # 实际推理并计时
    num_runs = 10  # 运行多次取平均
    times = []
    
    print(f"Running {num_runs} inference(s)...")
    for i in range(num_runs):
        t1 = time.time()
        outputs = session.run(None, input_dict)
        t2 = time.time()
        inference_time = (t2 - t1) * 1000  # 转换为毫秒
        times.append(inference_time)
        if i == 0:
            print(f"  Run {i+1}: {inference_time:.4f} ms")
    
    # 5. 显示结果
    print("\n" + "=" * 60)
    print("Inference Results")
    print("=" * 60)
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    print(f"\nTiming Statistics (over {num_runs} runs):")
    print(f"  Average time: {avg_time:.4f} ms")
    print(f"  Min time:     {min_time:.4f} ms")
    print(f"  Max time:     {max_time:.4f} ms")
    print(f"  Std deviation: {std_time:.4f} ms")
    
    print(f"\nOutput information:")
    for i, output in enumerate(outputs):
        if isinstance(output, np.ndarray):
            print(f"  Output {i}:")
            print(f"    Shape: {output.shape}")
            print(f"    Dtype: {output.dtype}")
            print(f"    Min:   {output.min():.6f}")
            print(f"    Max:   {output.max():.6f}")
            print(f"    Mean:  {output.mean():.6f}")
        else:
            print(f"  Output {i}: {type(output)}")
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)


if __name__ == '__main__':
    main()

