import numpy as np
import time
from nfstream import NFStreamer, NFPlugin
import sys
import os
import pandas as pd
from collections import deque
from rknnlite.api import RKNNLite

# 添加preprocessing目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
from preprocess_dataframe import DataFramePreprocessor


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
    def __init__(self, config_path, rknn_model_path):
        """初始化实时流量推理系统"""
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
        self.input_names = [
            'input_IN_BYTES',
            'input_OUT_BYTES',
            'input_FLOW_DURATION_MILLISECONDS',
            'input_IN_PKTS',
            'input_OUT_PKTS',
            'input_L4_SRC_PORT',
            'input_L4_DST_PORT',
            'input_PROTOCOL',
            'input_L7_PROTO',
            'input_TCP_FLAGS'
        ]
        
        self.model_expected_dims = {
            'L4_SRC_PORT': 32,
            'L4_DST_PORT': 32,
            'PROTOCOL': 3,
            'L7_PROTO': 19,
            'TCP_FLAGS': 9
        }
        
        # 维护滑动窗口（保存最近的window_size条流）
        self.window_size = self.preprocessor.window_size
        self.flow_window = deque(maxlen=self.window_size)
        # 维护窗口内每条流的IP信息（用于异常检测时打印）
        self.flow_ip_info = deque(maxlen=self.window_size)
        
        print(f'\nReal-time inference system initialized (window_size={self.window_size})')
        print('=' * 60)
    
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
        
        # 按照模型要求的顺序组织输入
        input_data_list = []
        for name in self.input_names:
            if name in model_inputs_dict:
                input_data_list.append(model_inputs_dict[name])
            else:
                # 创建占位符
                if 'IN_BYTES' in name or 'OUT_BYTES' in name or 'FLOW_DURATION' in name or 'PKTS' in name:
                    placeholder = np.zeros((1, 8, 1), dtype=np.float32)
                elif 'L4_SRC_PORT' in name or 'L4_DST_PORT' in name:
                    placeholder = np.zeros((1, 8, 32), dtype=np.float32)
                elif 'PROTOCOL' in name:
                    placeholder = np.zeros((1, 8, 3), dtype=np.float32)
                elif 'L7_PROTO' in name:
                    placeholder = np.zeros((1, 8, 19), dtype=np.float32)
                elif 'TCP_FLAGS' in name:
                    placeholder = np.zeros((1, 8, 9), dtype=np.float32)
                else:
                    placeholder = np.zeros((1, 8, 1), dtype=np.float32)
                input_data_list.append(placeholder)
        
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
            
            # 返回推理结果，包含窗口内所有流的IP信息
            result = {
                'output': output_array,
                'inference_time_ms': inference_time,
                'max_probability': max_prob,
                'predicted_class': predicted_class,
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


if __name__ == '__main__':
    # 配置文件路径
    config_path = "/home/lemon/zzu/ai-ids/config/config.json"
    # RKNN模型路径
    rknn_model = "/home/lemon/zzu/ai-ids/models/ids_transformer_model_new.rknn"
    
    # 初始化实时推理系统
    try:
        inference_system = RealTimeFlowInference(config_path, rknn_model)
    except Exception as e:
        print(f'Failed to initialize inference system: {e}')
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 设置输入源
    input_filepaths = ["eth0"]
    for path in sys.argv[1:]:
        input_filepaths.append(path)
    if len(input_filepaths) == 1:  # Single file / Interface
        input_filepaths = input_filepaths[0]
    
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
                    print(f'  Output: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}')
                    print(f'  Max probability: {max_prob:.4f}')
                    
                    # 如果输出是分类结果，显示预测类别
                    if len(output.shape) == 2 and output.shape[1] > 1:
                        predicted_class = result.get('predicted_class', 0)
                        confidence = output[0, predicted_class] if output.shape[0] > 0 else max_prob
                        print(f'  Predicted class: {predicted_class}, confidence: {confidence:.4f}')
                    
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