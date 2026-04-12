import argparse
import json
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ONNX model to RKNN model (run on RKNN toolkit machine).'
    )
    parser.add_argument('--onnx', required=True, help='Path to input ONNX model')
    parser.add_argument('--output', required=True, help='Path to output RKNN model')
    parser.add_argument('--target', default='rk3588', help='Target platform, e.g. rk3588')
    parser.add_argument('--quantize', action='store_true', help='Enable quantization')
    parser.add_argument(
        '--dataset',
        default=None,
        help='Calibration dataset path (required by RKNN when --quantize is enabled)'
    )
    parser.add_argument(
        '--input-size-list',
        default=None,
        help='Optional JSON string for input_size_list, e.g. [[1,8,1],[1,8,32],...]'
    )
    parser.add_argument('--verbose', action='store_true', help='Enable RKNN verbose logs')
    parser.add_argument(
        '--copy-meta',
        action='store_true',
        help='Copy ONNX sidecar metadata to RKNN sidecar (*.meta.json) if it exists'
    )
    return parser.parse_args()


def maybe_parse_input_size_list(raw):
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f'Invalid --input-size-list JSON: {e}') from e
    if not isinstance(value, list):
        raise RuntimeError('--input-size-list must be a JSON list')
    return value


def copy_meta_if_exists(onnx_path, rknn_path):
    onnx_abs = os.path.abspath(onnx_path)
    rknn_abs = os.path.abspath(rknn_path)

    onnx_stem, _ = os.path.splitext(onnx_abs)
    rknn_stem, _ = os.path.splitext(rknn_abs)

    candidates = [
        f'{onnx_stem}.meta.json',
        f'{onnx_stem}.json',
    ]

    dst_meta = f'{rknn_stem}.meta.json'
    for src in candidates:
        if os.path.exists(src):
            shutil.copyfile(src, dst_meta)
            print(f'Copied metadata: {src} -> {dst_meta}')
            return

    print('No ONNX metadata sidecar found, skipped copy-meta step.')


def main():
    args = parse_args()

    try:
        from rknn.api import RKNN
    except ImportError as e:
        raise RuntimeError(
            'rknn.api not found. Please run this script on the conversion machine with rknn-toolkit2 installed.'
        ) from e

    onnx_path = os.path.abspath(args.onnx)
    output_path = os.path.abspath(args.output)
    input_size_list = maybe_parse_input_size_list(args.input_size_list)

    if not os.path.exists(onnx_path):
        raise RuntimeError(f'ONNX model not found: {onnx_path}')

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.quantize and not args.dataset:
        raise RuntimeError('--dataset is required when --quantize is enabled')

    print('--> Creating RKNN object')
    rknn = RKNN(verbose=args.verbose)

    try:
        print('--> Config model')
        rknn.config(target_platform=args.target)
        print('done')

        print('--> Loading ONNX model')
        if input_size_list is None:
            ret = rknn.load_onnx(model=onnx_path)
        else:
            ret = rknn.load_onnx(model=onnx_path, input_size_list=input_size_list)
        if ret != 0:
            raise RuntimeError(f'Load ONNX failed: {ret}')
        print('done')

        print('--> Building RKNN model')
        ret = rknn.build(do_quantization=args.quantize, dataset=args.dataset)
        if ret != 0:
            raise RuntimeError(f'Build RKNN failed: {ret}')
        print('done')

        print('--> Exporting RKNN model')
        ret = rknn.export_rknn(output_path)
        if ret != 0:
            raise RuntimeError(f'Export RKNN failed: {ret}')
        print(f'done: {output_path}')

        if args.copy_meta:
            copy_meta_if_exists(onnx_path, output_path)
    finally:
        rknn.release()


if __name__ == '__main__':
    main()