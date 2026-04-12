import onnx
import onnx_graphsurgeon as gs
import numpy as np

def convert_einsum_to_matmul(input_path, output_path):
    model = onnx.load(input_path)
    model.opset_import[0].version = 13 
    graph = gs.import_onnx(model)
    einsum_count = 0

    for node in graph.nodes:
        if node.op == "Einsum":
            equation = node.attrs.get("equation", b"")
            if isinstance(equation, bytes): equation = equation.decode("utf-8")
            print(f"Found Einsum '{node.name}' eq: '{equation}'")
            
            inp_A = node.inputs[0]
            inp_B = node.inputs[1]
            out_C = node.outputs[0]

            if equation.replace(" ", "") == "abcd,cde->abe":
                # Special case: Reshape A to 3D and B to 2D
                # Reshape A to [0, 0, -1]
                shape_A = gs.Constant(name=f"{node.name}_shapeA", values=np.array([0, 0, -1], dtype=np.int64))
                reshaped_A = gs.Variable(name=f"{inp_A.name}_reshaped", dtype=inp_A.dtype)
                reshape_node_A = gs.Node(op="Reshape", name=f"{node.name}_ReshapeA", inputs=[inp_A, shape_A], outputs=[reshaped_A])
                graph.nodes.append(reshape_node_A)
                
                # Reshape B if it is constant
                if isinstance(inp_B, gs.Constant):
                    new_b_shape = (-1, inp_B.values.shape[-1])
                    inp_B.values = inp_B.values.reshape(new_b_shape)
                    matmul_inp_B = inp_B
                else:
                    shape_B = gs.Constant(name=f"{node.name}_shapeB", values=np.array([-1, inp_B.shape[-1] if inp_B.shape else out_C.shape[-1]], dtype=np.int64))
                    reshaped_B = gs.Variable(name=f"{inp_B.name}_reshaped", dtype=inp_B.dtype)
                    reshape_node_B = gs.Node(op="Reshape", name=f"{node.name}_ReshapeB", inputs=[inp_B, shape_B], outputs=[reshaped_B])
                    graph.nodes.append(reshape_node_B)
                    matmul_inp_B = reshaped_B

                matmul_node = gs.Node(op="MatMul", name=f"{node.name}_MatMul", inputs=[reshaped_A, matmul_inp_B], outputs=[out_C])
                graph.nodes.append(matmul_node)
                node.outputs = []
                einsum_count += 1

            elif "cd" in equation.split(",")[0] and "ed" in equation.split(",")[1]:
                rank = len(inp_B.shape) if inp_B.shape else 4
                perm = list(range(rank))
                perm[-1], perm[-2] = perm[-2], perm[-1]
                trans_out = gs.Variable(name=f"{inp_B.name}_transposed", dtype=inp_B.dtype)
                trans_node = gs.Node(op="Transpose", name=f"{node.name}_Transpose", inputs=[inp_B], outputs=[trans_out], attrs={"perm": perm})
                graph.nodes.append(trans_node)
                matmul_node = gs.Node(op="MatMul", name=f"{node.name}_MatMul", inputs=[inp_A, trans_out], outputs=[out_C])
                graph.nodes.append(matmul_node)
                node.outputs = []
                einsum_count += 1
            elif "cd" in equation.split(",")[0] and "de" in equation.split(",")[1]:
                matmul_node = gs.Node(op="MatMul", name=f"{node.name}_MatMul", inputs=[inp_A, inp_B], outputs=[out_C])
                graph.nodes.append(matmul_node)
                node.outputs = []
                einsum_count += 1
            else:
                print(f"[!] Warning: Handing generic fallback for '{equation}'.")
                eq_in, eq_out = equation.split('->')
                in_a, in_b = eq_in.split(',')
                if in_a[-1] == in_b[-1]:
                    rank = len(inp_B.shape) if inp_B.shape else len(in_b)
                    perm = list(range(rank))
                    perm[-1], perm[-2] = perm[-2], perm[-1]
                    trans_out = gs.Variable(name=f"{inp_B.name}_transposed", dtype=inp_B.dtype)
                    trans_node = gs.Node(op="Transpose", name=f"{node.name}_Transpose", inputs=[inp_B], outputs=[trans_out], attrs={"perm": perm})
                    graph.nodes.append(trans_node)
                    matmul_node = gs.Node(op="MatMul", name=f"{node.name}_MatMul", inputs=[inp_A, trans_out], outputs=[out_C])
                    graph.nodes.append(matmul_node)
                    node.outputs = []
                    einsum_count += 1
                else:
                    matmul_node = gs.Node(op="MatMul", name=f"{node.name}_MatMul", inputs=[inp_A, inp_B], outputs=[out_C])
                    graph.nodes.append(matmul_node)
                    node.outputs = []
                    einsum_count += 1

    if einsum_count > 0:
        graph.cleanup().toposort()
        try:
            new_model = gs.export_onnx(graph)
            new_model = onnx.shape_inference.infer_shapes(new_model)
            onnx.save(new_model, output_path)
            print(f"Successfully replaced {einsum_count} Einsum nodes.")
        except Exception as e:
            print(f"Failed to export: {e}")

