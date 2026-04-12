import os
import onnx
from onnxsim import simplify

# 获取当前脚本所在目录，以便无论从哪里运行都能找到正确的文件
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../models/t.onnx")
output_path = os.path.join(script_dir, "../models/tp.onnx")

# 加载模型
model = onnx.load(model_path)

# 定义覆盖形状的字典 {输入名: 形状}
input_shapes = {
    "input_IN_PKTS": [1, 8, 1],
    "input_OUT_PKTS": [1, 8, 1],
    "input_FLOW_DURATION_MILLISECONDS": [1, 8, 1],
    "input_IN_BYTES": [1, 8, 1],
    "input_OUT_BYTES": [1, 8, 1],
    "input_L4_SRC_PORT": [1, 8, 32],
    "input_L4_DST_PORT": [1, 8, 32],
    "input_PROTOCOL": [1, 8, 3],
    "input_L7_PROTO": [1, 8, 19],
    "input_TCP_FLAGS": [1, 8, 9]
}

# 简化并固定形状
model_simp, check = simplify(model, overwrite_input_shapes=input_shapes)

assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path)
print("模型已固定形状并保存！")