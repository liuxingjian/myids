import onnx
import onnx_graphsurgeon as gs
model = onnx.load("../models/t_multi_class.onnx")
graph = gs.import_onnx(model)
for node in graph.nodes:
    if node.name == "model/block_0_transformer_encoder/block_0_multi_head_attn/attention_output/einsum/Einsum_MatMul":
        for i, inp in enumerate(node.inputs):
            print(f"Input {i}: shape={inp.shape}")
        for i, out in enumerate(node.outputs):
            print(f"Output {i}: shape={out.shape}")
