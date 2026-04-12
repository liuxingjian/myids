import onnx
import onnx_graphsurgeon as gs
model = onnx.load("../models/t_multi_class.onnx")
graph = gs.import_onnx(model)
for node in graph.nodes:
    if "MatMul" in node.name and "einsum" in node.name:
        print(node.name)
        for i, inp in enumerate(node.inputs):
            print(f"Input {i}: name={inp.name} shape={inp.shape}")
