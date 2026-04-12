import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import onnxruntime as ort
from rknnlite.api import RKNNLite

try:
    import onnx
    from onnx import numpy_helper
except Exception:
    onnx = None
    numpy_helper = None

# Reuse project preprocessor for realistic input generation from CSV.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
from preprocess_dataframe import DataFramePreprocessor


MODEL_INPUT_NAMES = [
    'input_OUT_PKTS',
    'input_IN_BYTES',
    'input_OUT_BYTES',
    'input_IN_PKTS',
    'input_FLOW_DURATION_MILLISECONDS',
    'input_L4_SRC_PORT',
    'input_L4_DST_PORT',
    'input_PROTOCOL',
    'input_L7_PROTO',
    'input_TCP_FLAGS',
]

MODEL_EXPECTED_DIMS = {
    'L4_SRC_PORT': 32,
    'L4_DST_PORT': 32,
    'PROTOCOL': 5,
    'L7_PROTO': 32,
    'TCP_FLAGS': 18,
}


def infer_input_shape(input_name: str, window_size: int) -> tuple:
    if any(k in input_name for k in ['IN_BYTES', 'OUT_BYTES', 'FLOW_DURATION', 'PKTS']):
        return (1, window_size, 1)
    if 'L4_SRC_PORT' in input_name or 'L4_DST_PORT' in input_name:
        return (1, window_size, 32)
    if 'PROTOCOL' in input_name:
        return (1, window_size, 5)
    if 'L7_PROTO' in input_name:
        return (1, window_size, 32)
    if 'TCP_FLAGS' in input_name:
        return (1, window_size, 18)
    return (1, window_size, 1)


def build_random_inputs(input_names: List[str], window_size: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = {}
    for name in input_names:
        shape = infer_input_shape(name, window_size)
        arr = rng.standard_normal(shape).astype(np.float32)
        inputs[name] = arr
    return inputs


def build_inputs_from_npz(npz_path: str, input_names: List[str]) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    inputs = {}
    missing = []
    for name in input_names:
        if name not in data:
            missing.append(name)
            continue
        inputs[name] = data[name].astype(np.float32)

    if missing:
        raise RuntimeError(f'Missing keys in npz: {missing}')
    return inputs


def build_inputs_from_csv(csv_path: str, config_path: str) -> Dict[str, np.ndarray]:
    preprocessor = DataFramePreprocessor.load_from_json(config_path)

    if not os.path.exists(csv_path):
        raise RuntimeError(f'CSV not found: {csv_path}')

    df = np_load_dataframe(csv_path)
    preprocessed_df = preprocessor.preprocess_dataframe(
        df,
        drop_original=False,
        keep_other_columns=False,
        verbose=False,
    )

    model_inputs = preprocessor.dataframe_to_model_inputs(
        preprocessed_df,
        window_size=preprocessor.window_size,
        input_name_prefix='input_',
        batch_size=1,
        model_expected_dims=MODEL_EXPECTED_DIMS,
    )

    missing = [name for name in MODEL_INPUT_NAMES if name not in model_inputs]
    if missing:
        raise RuntimeError(f'Missing model inputs after preprocessing: {missing}')

    return {name: model_inputs[name].astype(np.float32) for name in MODEL_INPUT_NAMES}


def np_load_dataframe(csv_path: str):
    # pandas import is local to keep script startup light.
    import pandas as pd

    return pd.read_csv(csv_path)


def print_input_stats(inputs: Dict[str, np.ndarray], input_names: List[str]) -> None:
    print('\nInput stats:')
    for name in input_names:
        arr = inputs[name]
        print(
            f'  {name}: shape={arr.shape}, min={arr.min():.6f}, max={arr.max():.6f}, '
            f'mean={arr.mean():.6f}, std={arr.std():.6f}, nz={np.count_nonzero(arr)}/{arr.size}'
        )


def run_onnx(onnx_path: str, inputs: Dict[str, np.ndarray]):
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_input_names = [m.name for m in sess.get_inputs()]

    missing = [name for name in onnx_input_names if name not in inputs]
    if missing:
        raise RuntimeError(f'ONNX expected inputs missing from prepared input dict: {missing}')

    feed = {name: inputs[name] for name in onnx_input_names}
    t1 = time.time()
    outputs = sess.run(None, feed)
    t2 = time.time()
    return outputs, (t2 - t1) * 1000.0, onnx_input_names


def resolve_core_mask(core_id: int):
    if core_id == 0:
        return RKNNLite.NPU_CORE_0
    if core_id == 1:
        return RKNNLite.NPU_CORE_1
    if core_id == 2:
        return RKNNLite.NPU_CORE_2
    return RKNNLite.NPU_CORE_0


def run_rknn(rknn_path: str, inputs: Dict[str, np.ndarray], input_order: List[str], core_id: int):
    rknn = RKNNLite()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f'load_rknn failed: {ret}')

    ret = rknn.init_runtime(core_mask=resolve_core_mask(core_id))
    if ret != 0:
        raise RuntimeError(f'init_runtime failed: {ret}')

    try:
        infer_list = [inputs[name] for name in input_order]
        t1 = time.time()
        outputs = rknn.inference(inputs=infer_list)
        t2 = time.time()
        return outputs, (t2 - t1) * 1000.0
    finally:
        rknn.release()


def first_output_array(outputs) -> np.ndarray:
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 0:
            raise RuntimeError('No output returned')
        out = outputs[0]
    else:
        out = outputs
    return np.asarray(out)


def compare_outputs(onnx_out: np.ndarray, rknn_out: np.ndarray) -> None:
    onnx_flat = onnx_out.reshape(-1).astype(np.float64)
    rknn_flat = rknn_out.reshape(-1).astype(np.float64)

    print('\nOutput stats:')
    print(
        f'  ONNX: shape={onnx_out.shape}, min={onnx_out.min():.8f}, '
        f'max={onnx_out.max():.8f}, mean={onnx_out.mean():.8f}, std={onnx_out.std():.8f}'
    )
    print(
        f'  RKNN: shape={rknn_out.shape}, min={rknn_out.min():.8f}, '
        f'max={rknn_out.max():.8f}, mean={rknn_out.mean():.8f}, std={rknn_out.std():.8f}'
    )

    onnx_finite = np.isfinite(onnx_flat).all()
    rknn_finite = np.isfinite(rknn_flat).all()
    if not onnx_finite or not rknn_finite:
        print('\nDiff metrics: skipped (non-finite output detected)')
        if not onnx_finite:
            print('  Diagnosis: ONNX output contains NaN/Inf -> source ONNX is invalid or numerically unstable.')
        if not rknn_finite:
            print('  Diagnosis: RKNN output contains NaN/Inf -> conversion/runtime instability.')
        return

    if onnx_flat.shape != rknn_flat.shape:
        print('\nCannot compare element-wise: output sizes differ')
        print(f'  ONNX size={onnx_flat.size}, RKNN size={rknn_flat.size}')
        return

    diff = np.abs(onnx_flat - rknn_flat)
    mae = float(np.mean(diff))
    max_abs = float(np.max(diff))

    denom = (np.linalg.norm(onnx_flat) * np.linalg.norm(rknn_flat))
    cosine = float(np.dot(onnx_flat, rknn_flat) / denom) if denom > 0 else 0.0

    print('\nDiff metrics:')
    print(f'  MAE: {mae:.8f}')
    print(f'  MaxAbs: {max_abs:.8f}')
    print(f'  Cosine similarity: {cosine:.8f}')

    onnx_std = float(np.std(onnx_flat))
    rknn_std = float(np.std(rknn_flat))
    if onnx_std > 1e-6 and rknn_std < 1e-7:
        print('  Diagnosis: ONNX has variation but RKNN is near-constant -> conversion/quantization collapse likely.')
    elif onnx_std < 1e-7 and rknn_std < 1e-7:
        print('  Diagnosis: both ONNX and RKNN near-constant -> source model/export likely collapsed.')
    else:
        print('  Diagnosis: both sides have variation; check MAE/MaxAbs for precision loss level.')


def check_onnx_initializers(onnx_path: str) -> None:
    if onnx is None or numpy_helper is None:
        print('ONNX initializer check skipped (onnx package unavailable).')
        return

    model = onnx.load(onnx_path)
    bad = []
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if np.issubdtype(arr.dtype, np.floating):
            if np.isnan(arr).any() or np.isinf(arr).any():
                bad.append(init.name)

    if bad:
        print(f'ONNX initializer check: FAILED, non-finite tensors={len(bad)}')
        print(f'  Examples: {bad[:5]}')
    else:
        print('ONNX initializer check: PASSED')


def save_inputs_npz(path: str, inputs: Dict[str, np.ndarray], input_names: List[str]) -> None:
    payload = {name: inputs[name] for name in input_names}
    np.savez(path, **payload)
    print(f'Input tensor package saved: {os.path.abspath(path)}')


def main():
    parser = argparse.ArgumentParser(description='Compare ONNX vs RKNN with the exact same input tensors')
    parser.add_argument('--onnx', required=True, help='Path to ONNX model')
    parser.add_argument('--rknn', required=True, help='Path to RKNN model')
    parser.add_argument('--config', default='./config/config.json', help='Preprocessor config path (used with --csv)')
    parser.add_argument('--csv', default=None, help='CSV path used to build realistic model inputs')
    parser.add_argument('--npz-in', default=None, help='Load input tensors from npz (highest priority)')
    parser.add_argument('--npz-out', default=None, help='Save prepared input tensors to npz')
    parser.add_argument('--window-size', type=int, default=8, help='Window size used only in random mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used only in random mode')
    parser.add_argument('--core-id', type=int, default=0, choices=[0, 1, 2], help='RKNN NPU core id')
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        raise RuntimeError(f'ONNX model not found: {args.onnx}')
    if not os.path.exists(args.rknn):
        raise RuntimeError(f'RKNN model not found: {args.rknn}')

    if args.npz_in:
        print('Input mode: npz')
        inputs = build_inputs_from_npz(args.npz_in, MODEL_INPUT_NAMES)
    elif args.csv:
        print('Input mode: csv+config')
        inputs = build_inputs_from_csv(args.csv, args.config)
    else:
        print('Input mode: random')
        inputs = build_random_inputs(MODEL_INPUT_NAMES, args.window_size, args.seed)

    print_input_stats(inputs, MODEL_INPUT_NAMES)

    if args.npz_out:
        save_inputs_npz(args.npz_out, inputs, MODEL_INPUT_NAMES)

    onnx_outputs, onnx_ms, onnx_input_names = run_onnx(args.onnx, inputs)
    print(f'\nONNX runtime: {onnx_ms:.3f} ms')
    print(f'ONNX input names: {onnx_input_names}')
    check_onnx_initializers(args.onnx)

    rknn_outputs, rknn_ms = run_rknn(args.rknn, inputs, MODEL_INPUT_NAMES, args.core_id)
    print(f'RKNN runtime: {rknn_ms:.3f} ms')
    print(f'RKNN input order: {MODEL_INPUT_NAMES}')

    onnx_out = first_output_array(onnx_outputs)
    rknn_out = first_output_array(rknn_outputs)

    print('\nFirst output vectors:')
    print(f'  ONNX: {np.array2string(onnx_out.reshape(-1), precision=8, separator=", ")}')
    print(f'  RKNN: {np.array2string(rknn_out.reshape(-1), precision=8, separator=", ")}')

    compare_outputs(onnx_out, rknn_out)


if __name__ == '__main__':
    main()
