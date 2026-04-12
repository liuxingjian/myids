import numpy as np
import time
from nfstream import NFStreamer, NFPlugin
import sys
import os
import json
import pandas as pd
from collections import deque
from rknnlite.api import RKNNLite

# 添加preprocessing目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'preprocessing'))
from preprocess_dataframe import DataFramePreprocessor


DEFAULT_MODEL_INPUT_NAMES = [
    'input_OUT_PKTS',
    'input_IN_BYTES',
    'input_OUT_BYTES',
    'input_IN_PKTS',
    'input_FLOW_DURATION_MILLISECONDS',
    'input_L4_SRC_PORT',
    'input_L4_DST_PORT',
    'input_PROTOCOL',
    'input_L7_PROTO',
    'input_TCP_FLAGS'
]

DEFAULT_MODEL_EXPECTED_DIMS = {
    'L4_SRC_PORT': 32,
    'L4_DST_PORT': 32,
    'PROTOCOL': 5,
    'L7_PROTO': 32,
    'TCP_FLAGS': 18
}


def load_model_metadata(model_path, model_meta_path=None):
    """加载模型元信息（输入顺序/类别维度）。"""
    model_abs = os.path.abspath(model_path)
    stem, _ = os.path.splitext(model_abs)

    candidates = []
    if model_meta_path:
        candidates.append(os.path.abspath(model_meta_path))
    candidates.extend([
        f'{stem}.meta.json',
        f'{stem}.json',
    ])

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not os.path.exists(path):
            continue

        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as e:
            print(f'WARNING: Failed to read model metadata {path}: {e}')
            continue

        raw_names = payload.get('model_input_names', payload.get('input_names', []))
        if not isinstance(raw_names, list):
            raw_names = []
        input_names = [str(x) for x in raw_names if isinstance(x, str) and len(x) > 0]

        raw_dims = payload.get('model_expected_dims', {})
        dims = {}
        if isinstance(raw_dims, dict):
            for k, v in raw_dims.items():
                try:
                    v_int = int(v)
                except (TypeError, ValueError):
                    continue
                if v_int > 0:
                    dims[str(k)] = v_int

        if not input_names and not dims:
            print(f'WARNING: Model metadata {path} has no usable fields, skipped.')
            continue

        return {
            'path': path,
            'input_names': input_names,
            'model_expected_dims': dims,
        }

    return None


def write_aligned_config(base_config_path, output_path, model_expected_dims):
    """根据模型期望维度生成对齐后的预处理配置文件。"""
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    categorical_params = config.get('categorical_params', {})
    if not isinstance(categorical_params, dict):
        raise RuntimeError('Invalid config: categorical_params is missing or not a dict')

    print('Aligning config categorical dims to model expected dims...')
    for feat, expected_dim in model_expected_dims.items():
        if feat not in categorical_params:
            raise RuntimeError(f'Config missing categorical feature: {feat}')

        params = categorical_params[feat]
        levels = list(params.get('levels', []))
        old_dim = int(params.get('n_levels', len(levels)))

        if len(levels) > expected_dim:
            # 保护性截断，避免 levels 大于模型输入宽度。
            levels = levels[:expected_dim]

        params['levels'] = levels
        params['n_levels'] = expected_dim
        print(f'  - {feat}: {old_dim} -> {expected_dim}')

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f'Aligned config written to: {os.path.abspath(output_path)}')
    return output_path


class FlowSlicer(NFPlugin):
    def on_init(self, packet, flow):
        flow.udps.tcp_flags = 0

    def on_update(self, packet, flow):
        if(packet.syn):
            flow.udps.tcp_flags |= 0x02
        if(packet.ack):
            flow.udps.tcp_flags |= 0x10
        if(packet.fin):
            flow.udps.tcp_flags |= 0x01
        if(packet.rst):
            flow.udps.tcp_flags |= 0x04
        if(packet.psh):
            flow.udps.tcp_flags |= 0x08
        if(packet.urg):
            flow.udps.tcp_flags |= 0x20


class RealTimeFlowInference:
    def __init__(self, config_path, rknn_model_path, model_meta=None, allow_dim_mismatch=False):
        """初始化实时流量推理系统"""
        self.allow_dim_mismatch = allow_dim_mismatch
        self.config_path = config_path

        # 加载预处理器
        print('Loading preprocessor...')
        self.preprocessor = DataFramePreprocessor.load_from_json(config_path)
        print(f'  Loaded {len(self.preprocessor.numerical_features)} numerical features')
        print(f'  Loaded {len(self.preprocessor.categorical_features)} categorical features')
        print(f'  Window size: {self.preprocessor.window_size}')
        
        # 加载RKNN模型
        print('Loading RKNN model...')
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(rknn_model_path)
        if ret != 0:
            raise RuntimeError(f'Load RKNN model failed: {ret}')
        print('  Model loaded successfully')
        
        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            raise RuntimeError(f'Init runtime environment failed: {ret}')
        print('  Runtime initialized')
        
        # 模型期望的输入顺序和维度
        # 默认值用于兼容旧模型；若存在 sidecar 元信息则优先使用元信息。
        self.input_names = list(DEFAULT_MODEL_INPUT_NAMES)
        self.model_expected_dims = dict(DEFAULT_MODEL_EXPECTED_DIMS)
        if isinstance(model_meta, dict):
            meta_names = model_meta.get('input_names', [])
            if isinstance(meta_names, list) and len(meta_names) > 0:
                self.input_names = list(meta_names)

            meta_dims = model_meta.get('model_expected_dims', {})
            if isinstance(meta_dims, dict):
                for feat, dim in meta_dims.items():
                    try:
                        dim_int = int(dim)
                    except (TypeError, ValueError):
                        continue
                    if dim_int > 0:
                        self.model_expected_dims[str(feat)] = dim_int

        print(f'  Input order source: {"model metadata" if model_meta else "built-in default"}')
        print(f'  Input names: {self.input_names}')
        print(f'  Expected categorical dims: {self.model_expected_dims}')

        # 诊断参数：前几次和固定间隔打印输入统计，便于排查恒定输出。
        self._inference_count = 0
        self.debug_first_n = 3
        self.debug_every_n = 30

        self._validate_preprocessor_dims()
        
        # 维护滑动窗口（保存最近的window_size条流）
        self.window_size = self.preprocessor.window_size
        self.flow_window = deque(maxlen=self.window_size)
        # 维护窗口内每条流的IP信息（用于异常检测时打印）
        self.flow_ip_info = deque(maxlen=self.window_size)
        
        print(f'\nReal-time inference system initialized (window_size={self.window_size})')
        print('=' * 60)

    def _validate_preprocessor_dims(self):
        """检查预处理配置与模型输入维度是否一致。"""
        mismatches = []
        for feat, expected_dim in self.model_expected_dims.items():
            params = self.preprocessor.categorical_params.get(feat, {})
            configured_dim = params.get('n_levels', len(params.get('levels', [])))
            if configured_dim != expected_dim:
                mismatches.append((feat, configured_dim, expected_dim))

        if mismatches:
            print('WARNING: Preprocessor categorical dims do not fully match model dims:')
            for feat, configured_dim, expected_dim in mismatches:
                print(f'  - {feat}: config={configured_dim}, model={expected_dim}')
            print('  Extra model dims will be zero/unknown padded, which may degrade accuracy.')
            if not self.allow_dim_mismatch:
                raise RuntimeError(
                    'Config/model categorical dimensions mismatch. '
                    'Use matching config for this RKNN model, or run with --auto-align-config, '
                    'or pass --allow-dim-mismatch to force run.'
                )

    def _should_log_input_snapshot(self):
        if self._inference_count <= self.debug_first_n:
            return True
        return self.debug_every_n > 0 and (self._inference_count % self.debug_every_n == 0)

    def _log_input_snapshot(self, model_inputs_dict):
        print(f'Input snapshot (inference #{self._inference_count}):')
        for name in self.input_names:
            arr = model_inputs_dict[name]
            arr_min = float(np.min(arr))
            arr_max = float(np.max(arr))
            arr_mean = float(np.mean(arr))
            arr_std = float(np.std(arr))
            non_zero = int(np.count_nonzero(arr))
            print(
                f'  {name}: shape={arr.shape}, min={arr_min:.6f}, max={arr_max:.6f}, '
                f'mean={arr_mean:.6f}, std={arr_std:.6f}, nz={non_zero}/{arr.size}'
            )
    
    def flow_to_dataframe(self, flow):
        """将NFStreamer的flow对象转换为DataFrame"""
        # 处理L7_PROTO：如果application_name为空或None，使用默认值
        l7_proto = flow.application_name if flow.application_name else 'Unknown'
        
        # 创建DataFrame，只包含模型需要的特征
        df = pd.DataFrame([{
            'IN_PKTS': flow.dst2src_packets,
            'OUT_PKTS': flow.src2dst_packets,
            'IN_BYTES': flow.dst2src_bytes,
            'OUT_BYTES': flow.src2dst_bytes,
            'FLOW_DURATION_MILLISECONDS': flow.bidirectional_duration_ms,
            'L4_SRC_PORT': flow.src_port,
            'L4_DST_PORT': flow.dst_port,
            'PROTOCOL': flow.protocol,
            'L7_PROTO': l7_proto,
            'TCP_FLAGS': flow.udps.tcp_flags
        }])
        return df
    
    def process_flow(self, flow):
        """处理单条流：添加到窗口并进行推理"""
        # 保存当前流的IP信息（用于异常检测时打印）
        ip_info = {
            'src_ip': flow.src_ip,
            'dst_ip': flow.dst_ip,
            'src_port': flow.src_port,
            'dst_port': flow.dst_port,
            'protocol': flow.protocol,
            'l7_proto': flow.application_name if flow.application_name else 'Unknown'
        }
        
        # 将flow转换为DataFrame
        flow_df = self.flow_to_dataframe(flow)
        
        # 添加到滑动窗口
        self.flow_window.append(flow_df)
        self.flow_ip_info.append(ip_info)
        
        # 如果窗口未满，不进行推理
        if len(self.flow_window) < self.window_size:
            print(f'Window not full yet: {len(self.flow_window)}/{self.window_size} flows')
            return None
        
        # 合并窗口中的所有流
        window_df = pd.concat(list(self.flow_window), ignore_index=True)
        
        # 预处理
        try:
            preprocessed_df = self.preprocessor.preprocess_dataframe(
                window_df,
                drop_original=False,
                keep_other_columns=False
            )
        except Exception as e:
            print(f'Error preprocessing: {e}')
            return None
        
        # 转换为模型输入格式
        try:
            model_inputs_dict = self.preprocessor.dataframe_to_model_inputs(
                preprocessed_df,
                window_size=self.window_size,
                input_name_prefix="input_",
                batch_size=1,
                model_expected_dims=self.model_expected_dims
            )
        except Exception as e:
            print(f'Error converting to model inputs: {e}')
            return None

        missing_inputs = [name for name in self.input_names if name not in model_inputs_dict]
        if missing_inputs:
            print(f'Error: missing model inputs: {missing_inputs}')
            print(f'Available inputs: {sorted(model_inputs_dict.keys())}')
            return None
        
        # 按照模型要求的顺序组织输入
        input_data_list = [model_inputs_dict[name] for name in self.input_names]

        self._inference_count += 1
        if self._should_log_input_snapshot():
            self._log_input_snapshot(model_inputs_dict)
        
        # 进行推理
        try:
            t1 = time.time()
            outputs = self.rknn_lite.inference(inputs=input_data_list)
            t2 = time.time()
            inference_time = (t2 - t1) * 1000  # 转换为毫秒
            
            # 获取输出值（处理不同的输出格式）
            output_array = outputs[0] if isinstance(outputs, (list, tuple)) and len(outputs) > 0 else outputs
            if isinstance(output_array, np.ndarray):
                # 如果是分类输出（多类），取最大概率值；如果是二分类，取第二个值或最大值
                if len(output_array.shape) == 2:
                    # 形状为 (batch, classes)
                    if output_array.shape[1] > 1:
                        # 多分类：取最大概率值作为异常分数
                        max_prob = np.max(output_array, axis=1)[0]
                        predicted_class = np.argmax(output_array, axis=1)[0]
                    else:
                        # 单输出：直接使用
                        max_prob = output_array[0, 0] if output_array.shape[1] == 1 else output_array[0]
                        predicted_class = 0
                elif len(output_array.shape) == 1:
                    # 一维数组
                    max_prob = np.max(output_array)
                    predicted_class = np.argmax(output_array)
                else:
                    # 其他形状，取最大值
                    max_prob = np.max(output_array)
                    predicted_class = 0
            else:
                max_prob = float(output_array) if isinstance(output_array, (int, float)) else 0.0
                predicted_class = 0

            output_std = float(np.std(output_array)) if isinstance(output_array, np.ndarray) else 0.0
            if output_std < 1e-7:
                print('WARNING: Model output is near-uniform (std≈0). '
                      'Please verify model file/version and preprocessing config consistency.')
            
            # 返回推理结果，包含窗口内所有流的IP信息
            result = {
                'output': output_array,
                'inference_time_ms': inference_time,
                'max_probability': max_prob,
                'predicted_class': predicted_class,
                'output_std': output_std,
                'flow_info': {
                    'src_port': flow.src_port,
                    'dst_port': flow.dst_port,
                    'protocol': flow.protocol,
                    'l7_proto': flow.application_name if flow.application_name else 'Unknown'
                },
                # 窗口内所有流的IP信息（用于异常检测时打印）
                'window_ip_info': list(self.flow_ip_info)
            }
            return result
        except Exception as e:
            print(f'Error during inference: {e}')
            return None
    
    def release(self):
        """释放资源"""
        if self.rknn_lite:
            self.rknn_lite.release()


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time Flow Inference System")
    parser.add_argument('-m', '--model', required=True, help="Path to the RKNN model file (required)")
    parser.add_argument('-c', '--config', default='./config/config.json', help="Path to the preprocessor config file (default: ./config/config.json)")
    parser.add_argument('--model-meta', default=None, help="Optional model metadata json path (contains input order/dims). If omitted, auto-detect from model sidecar")
    parser.add_argument('--auto-align-config', action='store_true', help="Auto-generate an aligned config using model expected categorical dims")
    parser.add_argument('--aligned-config-out', default='./config/config.aligned_to_model.json', help="Output path for auto-aligned config (used with --auto-align-config)")
    parser.add_argument('--allow-dim-mismatch', action='store_true', help="Force run even if config categorical dims mismatch model dims")
    parser.add_argument('interface', nargs='?', default='eth0', help="Network interface or pcap file (default: eth0)")
    args = parser.parse_args()

    # 配置文件路径
    config_path = args.config
    # RKNN模型路径
    rknn_model = args.model

    print(f'Using config: {os.path.abspath(config_path)}')
    print(f'Using model : {os.path.abspath(rknn_model)}')

    if not os.path.exists(config_path):
        print(f'Config file not found: {config_path}')
        exit(1)
    if not os.path.exists(rknn_model):
        print(f'Model file not found: {rknn_model}')
        exit(1)

    model_meta = load_model_metadata(rknn_model, args.model_meta)
    if model_meta:
        print(f'Using model metadata: {model_meta["path"]}')

    effective_expected_dims = dict(DEFAULT_MODEL_EXPECTED_DIMS)
    if model_meta and isinstance(model_meta.get('model_expected_dims'), dict):
        effective_expected_dims.update(model_meta['model_expected_dims'])

    if args.auto_align_config:
        try:
            config_path = write_aligned_config(
                config_path,
                args.aligned_config_out,
                effective_expected_dims
            )
        except Exception as e:
            print(f'Failed to auto-align config: {e}')
            import traceback
            traceback.print_exc()
            exit(1)
    
    # 初始化实时推理系统
    try:
        inference_system = RealTimeFlowInference(
            config_path,
            rknn_model,
            model_meta=model_meta,
            allow_dim_mismatch=args.allow_dim_mismatch
        )
    except Exception as e:
        print(f'Failed to initialize inference system: {e}')
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 设置输入源
    input_filepaths = args.interface
    
    # 创建流收集器
    flow_streamer = NFStreamer(
        source=input_filepaths, 
        statistical_analysis=False, 
        idle_timeout=1,
        udps=FlowSlicer()
    )
    
    print('\nStarting real-time flow collection and inference...')
    print('Press Ctrl+C to stop\n')
    
    cnt = 0
    try:
        for flow in flow_streamer:
            cnt += 1
            
            # 处理流并进行推理
            result = inference_system.process_flow(flow)
            
            if result:
                output = result['output']
                max_prob = result.get('max_probability', 0.0)
                
                if isinstance(output, np.ndarray):
                    print(f'\n[Flow #{cnt}] Inference completed')
                    print(f'  Flow info: {result["flow_info"]}')
                    print(f'  Inference time: {result["inference_time_ms"]:.2f} ms')
                    print(f'  Output shape: {output.shape}')
                    print(f'  Output: min={output.min():.8f}, max={output.max():.8f}, mean={output.mean():.8f}')
                    print(f'  Output vector: {np.array2string(output.reshape(-1), precision=8, separator=", ")}')
                    print(f'  Output std: {result.get("output_std", 0.0):.8f}')
                    print(f'  Max probability: {max_prob:.8f}')
                    
                    # 如果输出是分类结果，显示预测类别
                    if len(output.shape) == 2 and output.shape[1] > 1:
                        predicted_class = result.get('predicted_class', 0)
                        confidence = output[0, predicted_class] if output.shape[0] > 0 else max_prob
                        print(f'  Predicted class: {predicted_class}, confidence: {confidence:.8f}')
                    
                    # 如果检测到异常（概率大于0.5），打印IP信息
                    if max_prob > 0.5:
                        separator = '=' * 60
                        print(f'\n  ⚠️  ALERT: Anomaly detected (probability: {max_prob:.4f} > 0.5)')
                        print(f'  {separator}')
                        # print(f'  Window IP Information (last {len(result["window_ip_info"])} flows):')
                        # for i, ip_info in enumerate(result['window_ip_info']):
                        #     print(f'    Flow {i+1}:')
                        #     print(f'      Source: {ip_info["src_ip"]}:{ip_info["src_port"]}')
                        #     print(f'      Destination: {ip_info["dst_ip"]}:{ip_info["dst_port"]}')
                        #     print(f'      Protocol: {ip_info["protocol"]} ({ip_info["l7_proto"]})')
                        # print(f'  {separator}')
                        # 特别标注当前流（最后一条）
                        if result['window_ip_info']:
                            current_flow = result['window_ip_info'][-1]
                            print(f'  🎯 Current Flow (most recent):')
                            print(f'      Source IP: {current_flow["src_ip"]}')
                            print(f'      Destination IP: {current_flow["dst_ip"]}')
                            print(f'      Source Port: {current_flow["src_port"]}')
                            print(f'      Destination Port: {current_flow["dst_port"]}')
                            print(f'      Protocol: {current_flow["protocol"]} ({current_flow["l7_proto"]})')
                        print(f'  {separator}\n')
            else:
                print(f'[Flow #{cnt}] Added to window (window size: {len(inference_system.flow_window)})')
                
    except KeyboardInterrupt:
        print('\n\nStopped by user')
    except Exception as e:
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()
    finally:
        print(f'\nTotal flows processed: {cnt}')
        inference_system.release()
        print('Resources released')