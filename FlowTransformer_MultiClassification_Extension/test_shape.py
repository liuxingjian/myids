import onnx
import onnx_graphsurgeon as gs

m = onnx.load("t.onnx")
g = gs.import_onnx(m)

for n in g.nodes:
    if n.op == "MatMul" and "Einsum" in n.name:
        print(n.name)
        for i, inp in enumerate(n.inputs):
            print(f"  Inp {i}: {inp.name} shape={inp.shape}")
        
