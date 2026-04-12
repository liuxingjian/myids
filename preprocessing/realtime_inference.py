"""
实时流量预处理和ONNX模型推理工具

该工具独立于仓库其他代码，可以用于：
1. 对实时提取的流量数据进行预处理
2. 加载ONNX模型并进行推理
3. 支持窗口化处理（用于时序模型）

依赖：
    pip install numpy pandas onnxruntime

使用示例：
    from realtime_inference import TrafficPreprocessor, ONNXInferenceEngine
    
    # 1. 初始化预处理器（需要提供预处理参数）
    preprocessor = TrafficPreprocessor(
        numerical_features=['IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS', 'IN_BYTES', 'OUT_BYTES'],
        categorical_features=['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'TCP_FLAGS'],
        numerical_params={
            'IN_PKTS': {'min': 0.0, 'range': 1000000.0},
            'OUT_PKTS': {'min': 0.0, 'range': 1000000.0},
            # ... 其他数值特征的参数
        },
        categorical_params={
            'L4_SRC_PORT': {'levels': [80, 443, 22, ...], 'n_levels': 32},
            # ... 其他类别特征的参数
        },
        window_size=8,
        top_k=32
    )
    
    # 2. 加载ONNX模型
    engine = ONNXInferenceEngine('path/to/model.onnx')
    
    # 3. 处理单条流量
    flow_data = {
        'IN_PKTS': 100,
        'OUT_PKTS': 200,
        # ... 其他特征
    }
    result = engine.predict_single(flow_data, preprocessor)
    
    # 4. 批量处理
    flows = [flow1, flow2, ...]
    results = engine.predict_batch(flows, preprocessor)
"""
"""
使用示例: 导出数据集的预处理器配置
python realtime_inference.py --extract-config preprocessor_config_UNSW_NB15_test.json \
    --dataset /home/yaojiahao/ydd_ws/new/yddflow/example/converted_flows_nfv1.csv \
    --numerical-features IN_PKTS OUT_PKTS FLOW_DURATION_MILLISECONDS IN_BYTES OUT_BYTES \
    --categorical-features L4_SRC_PORT L4_DST_PORT PROTOCOL L7_PROTO TCP_FLAGS

"""
"""
使用示例: 测试预处理器和推理引擎（使用真实数据集）
python realtime_inference.py --test \
    --model /home/yaojiahao/ydd_ws/new/yddflow/artifacts/student_test_2026_2114_m1.onnx \
    --config preprocessor_config_UNSW_NB15_test.json \
    --test-data converted_flows_nfv1.csv \
    --test-n-rows 10
"""
import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is required. Install it with: pip install onnxruntime"
    )


class TrafficPreprocessor:
    """
    流量数据预处理器
    
    实现与训练时相同的预处理逻辑：
    - 数值特征：对数归一化
    - 类别特征：Top-K编码 + OneHot编码
    """
    
    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        numerical_params: Dict[str, Dict[str, float]],
        categorical_params: Dict[str, Dict],
        window_size: int = 8,
        top_k: int = 32,
        clip_numerical: bool = False
    ):
        """
        初始化预处理器
        
        Args:
            numerical_features: 数值特征名称列表
            categorical_features: 类别特征名称列表
            numerical_params: 数值特征预处理参数
                {feature_name: {'min': float, 'range': float}}
            categorical_params: 类别特征预处理参数
                {feature_name: {'levels': List, 'n_levels': int}}
            window_size: 窗口大小（用于时序模型）
            top_k: 类别特征Top-K数量
            clip_numerical: 是否裁剪数值特征到[0,1]
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.numerical_params = numerical_params
        self.categorical_params = categorical_params
        self.window_size = window_size
        self.top_k = top_k
        self.clip_numerical = clip_numerical
        
        # 验证参数完整性
        for feat in numerical_features:
            if feat not in numerical_params:
                raise ValueError(f"Missing numerical params for feature: {feat}")
            if 'min' not in numerical_params[feat] or 'range' not in numerical_params[feat]:
                raise ValueError(f"Invalid numerical params for feature: {feat}")
        
        for feat in categorical_features:
            if feat not in categorical_params:
                raise ValueError(f"Missing categorical params for feature: {feat}")
            if 'levels' not in categorical_params[feat]:
                raise ValueError(f"Missing 'levels' in categorical params for feature: {feat}")
    
    def transform_numerical(self, feature_name: str, values: np.ndarray) -> np.ndarray:
        """
        转换数值特征
        
        Args:
            feature_name: 特征名称
            values: 特征值数组
            
        Returns:
            转换后的特征值数组
        """
        params = self.numerical_params[feature_name]
        col_min = params['min']
        col_range = params['range']
        
        if col_range == 0:
            return np.zeros_like(values, dtype=np.float32)
        
        # 复制以避免修改原数组
        values = values.copy().astype(np.float32)
        
        # 中心化
        values -= col_min
        
        # 对数变换
        col_values = np.log(values + 1)
        
        # 归一化
        col_values *= 1.0 / np.log(col_range + 1)
        
        # 可选裁剪
        if self.clip_numerical:
            col_values = np.clip(col_values, 0.0, 1.0)
        
        return col_values.astype(np.float32)
    
    def transform_categorical(
        self, 
        feature_name: str, 
        values: Union[np.ndarray, List]
    ) -> np.ndarray:
        """
        转换类别特征为OneHot编码
        
        Args:
            feature_name: 特征名称
            values: 特征值数组或列表
            
        Returns:
            OneHot编码数组，形状为 (n_samples, n_levels)
        """
        params = self.categorical_params[feature_name]
        encoded_levels = params['levels']
        n_levels = params.get('n_levels', len(encoded_levels))
        
        values = np.asarray(values)
        result_values = np.ones(len(values), dtype=np.uint32)  # 默认值1表示未知
        
        # 编码：0=未知，1+为已知类别
        for level_i, level in enumerate(encoded_levels):
            level_mask = values == level
            result_values[level_mask] = level_i + 1
        
        # OneHot编码
        # 创建one-hot矩阵
        onehot = np.zeros((len(result_values), n_levels), dtype=np.float32)
        for i, val in enumerate(result_values):
            if 0 < val <= n_levels:
                onehot[i, val - 1] = 1.0
        
        return onehot
    
    def preprocess_single_flow(self, flow_data: Dict) -> Dict[str, np.ndarray]:
        """
        预处理单条流量数据
        
        Args:
            flow_data: 流量数据字典，键为特征名称
            
        Returns:
            预处理后的特征字典，键为特征名称，值为numpy数组
        """
        preprocessed = {}
        
        # 处理数值特征
        for feat in self.numerical_features:
            if feat not in flow_data:
                raise ValueError(f"Missing numerical feature: {feat}")
            value = flow_data[feat]
            # 转换为数组并处理
            values = np.array([value], dtype=np.float32)
            preprocessed[feat] = self.transform_numerical(feat, values)
        
        # 处理类别特征
        for feat in self.categorical_features:
            if feat not in flow_data:
                raise ValueError(f"Missing categorical feature: {feat}")
            value = flow_data[feat]
            # 转换为数组并处理
            values = np.array([value])
            preprocessed[feat] = self.transform_categorical(feat, values)
        
        return preprocessed
    
    def build_window(
        self, 
        flows: List[Dict],
        ordered: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        构建窗口数据（用于时序模型）
        
        Args:
            flows: 流量数据列表（至少window_size条）
            ordered: 是否按顺序排列
            
        Returns:
            窗口化的特征字典，每个特征的形状为 (1, window_size, feature_dim)
        """
        if len(flows) < self.window_size:
            raise ValueError(
                f"Need at least {self.window_size} flows, got {len(flows)}"
            )
        
        # 取最后window_size条
        window_flows = flows[-self.window_size:] if ordered else flows[:self.window_size]
        
        # 预处理每条流量
        preprocessed_list = []
        for flow in window_flows:
            preprocessed_list.append(self.preprocess_single_flow(flow))
        
        # 构建窗口
        window_data = {}
        
        # 数值特征：形状 (window_size, 1) -> (1, window_size, 1)
        for feat in self.numerical_features:
            values = np.array([preprocessed_list[i][feat][0] for i in range(self.window_size)])
            window_data[feat] = values.reshape(1, self.window_size, 1).astype(np.float32)
        
        # 类别特征：形状 (window_size, n_levels) -> (1, window_size, n_levels)
        for feat in self.categorical_features:
            # 获取期望的维度（从配置中）
            expected_n_levels = self.categorical_params[feat].get('n_levels', len(self.categorical_params[feat]['levels']))
            
            # 收集所有预处理后的特征值
            feature_values = []
            for i in range(self.window_size):
                feat_data = preprocessed_list[i][feat]
                # 确保是一维数组
                if len(feat_data.shape) == 0:
                    feat_data = feat_data.reshape(1)
                elif len(feat_data.shape) > 1:
                    feat_data = feat_data.flatten()
                
                # 调整到期望的维度
                if len(feat_data) < expected_n_levels:
                    # 填充零
                    padded = np.zeros(expected_n_levels, dtype=np.float32)
                    padded[:len(feat_data)] = feat_data
                    feature_values.append(padded)
                elif len(feat_data) > expected_n_levels:
                    # 截取
                    feature_values.append(feat_data[:expected_n_levels])
                else:
                    feature_values.append(feat_data)
            
            # 转换为数组并重塑
            values = np.array(feature_values)  # shape: (window_size, expected_n_levels)
            window_data[feat] = values.reshape(1, self.window_size, expected_n_levels).astype(np.float32)
        
        return window_data
    
    @classmethod
    def load_from_json(cls, json_path: str) -> 'TrafficPreprocessor':
        """
        从JSON文件加载预处理器配置
        
        Args:
            json_path: JSON配置文件路径
            
        Returns:
            TrafficPreprocessor实例
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return cls(
            numerical_features=config['numerical_features'],
            categorical_features=config['categorical_features'],
            numerical_params=config['numerical_params'],
            categorical_params=config['categorical_params'],
            window_size=config.get('window_size', 8),
            top_k=config.get('top_k', 32),
            clip_numerical=config.get('clip_numerical', False)
        )
    
    def save_to_json(self, json_path: str):
        """
        保存预处理器配置到JSON文件
        
        Args:
            json_path: 保存路径
        """
        config = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'numerical_params': self.numerical_params,
            'categorical_params': self.categorical_params,
            'window_size': self.window_size,
            'top_k': self.top_k,
            'clip_numerical': self.clip_numerical
        }
        
        os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else '.', exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


class ONNXInferenceEngine:
    """
    ONNX模型推理引擎
    """
    
    def __init__(
        self, 
        model_path: str,
        providers: Optional[List[str]] = None
    ):
        """
        初始化推理引擎
        
        Args:
            model_path: ONNX模型文件路径
            providers: 执行提供者列表，默认使用CPU
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 获取输入输出信息
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.input_shapes = [inp.shape for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"Loaded ONNX model: {model_path}")
        print(f"Inputs: {len(self.input_names)}")
        for i, (name, shape) in enumerate(zip(self.input_names, self.input_shapes)):
            print(f"  [{i}] {name}: {shape}")
        print(f"Outputs: {len(self.output_names)}")
        for i, name in enumerate(self.output_names):
            print(f"  [{i}] {name}")
    
    def predict(
        self, 
        inputs: Dict[str, np.ndarray]
    ) -> List[np.ndarray]:
        """
        执行推理
        
        Args:
            inputs: 输入字典，键为输入名称，值为numpy数组
            
        Returns:
            输出列表
        """
        # 验证输入
        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"Missing input: {name}")
        
        # 准备输入（按顺序）
        feed_dict = {name: inputs[name] for name in self.input_names}
        
        # 执行推理
        outputs = self.session.run(self.output_names, feed_dict)
        
        return outputs
    
    def predict_single(
        self,
        flow_data: Dict,
        preprocessor: TrafficPreprocessor,
        use_window: bool = True
    ) -> np.ndarray:
        """
        对单条流量进行预测（自动构建窗口）
        
        Args:
            flow_data: 单条流量数据
            preprocessor: 预处理器
            use_window: 是否使用窗口（需要历史数据）
            
        Returns:
            预测结果（通常是概率值）
        """
        if use_window:
            # 需要历史数据构建窗口
            # 这里假设flow_data包含历史数据，或者需要外部提供
            raise NotImplementedError(
                "Single flow prediction with window requires historical data. "
                "Use predict_batch or build_window manually."
            )
        else:
            # 不使用窗口，直接处理单条数据
            preprocessed = preprocessor.preprocess_single_flow(flow_data)
            # 扩展维度以匹配模型输入
            inputs = {}
            for name in self.input_names:
                feat_name = name.replace('input_', '')
                if feat_name in preprocessed:
                    # 数值特征: (1,) -> (1, 1, 1)
                    if feat_name in preprocessor.numerical_features:
                        inputs[name] = preprocessed[feat_name].reshape(1, 1, 1)
                    # 类别特征: (n_levels,) -> (1, 1, n_levels)
                    else:
                        inputs[name] = preprocessed[feat_name].reshape(1, 1, -1)
            
            outputs = self.predict(inputs)
            return outputs[0]  # 返回第一个输出
    
    def predict_batch(
        self,
        flows: List[Dict],
        preprocessor: TrafficPreprocessor
    ) -> np.ndarray:
        """
        批量预测（自动构建窗口）
        
        Args:
            flows: 流量数据列表
            preprocessor: 预处理器
            
        Returns:
            预测结果数组
        """
        # 构建窗口
        window_data = preprocessor.build_window(flows)
        
        # 准备输入（匹配模型输入名称）
        inputs = {}
        for i, name in enumerate(self.input_names):
            feat_name = name.replace('input_', '')
            if feat_name in window_data:
                data = window_data[feat_name]
                expected_shape = self.input_shapes[i]
                
                # 验证和调整维度
                # 如果模型期望的维度是固定的（不是动态的），需要确保匹配
                if len(expected_shape) == 3:  # 形状为 (batch, seq, feature)
                    batch_dim, seq_dim, feat_dim = expected_shape
                    
                    # 如果特征维度是固定的（不是 'unk__xxx'），需要确保匹配
                    if isinstance(feat_dim, int):
                        current_shape = data.shape
                        if current_shape[2] != feat_dim:
                            # 需要调整维度
                            if current_shape[2] < feat_dim:
                                # 填充零
                                padded_data = np.zeros((current_shape[0], current_shape[1], feat_dim), dtype=data.dtype)
                                padded_data[:, :, :current_shape[2]] = data
                                data = padded_data
                            else:
                                # 截取
                                data = data[:, :, :feat_dim]
                    
                    # 确保序列维度匹配
                    if isinstance(seq_dim, int) and data.shape[1] != seq_dim:
                        if data.shape[1] < seq_dim:
                            # 填充零（在序列维度）
                            padded_data = np.zeros((data.shape[0], seq_dim, data.shape[2]), dtype=data.dtype)
                            padded_data[:, :data.shape[1], :] = data
                            data = padded_data
                        else:
                            # 截取
                            data = data[:, :seq_dim, :]
                
                inputs[name] = data
            else:
                raise ValueError(
                    f"Model input '{name}' not found in preprocessed data. "
                    f"Expected feature name: {feat_name}"
                )
        
        # 执行推理
        outputs = self.predict(inputs)
        return outputs[0]  # 返回第一个输出


def extract_config_from_dataset(
    dataset_path: str,
    output_path: str,
    numerical_features: List[str],
    categorical_features: List[str],
    top_k: int = 32,
    window_size: int = 8,
    clip_numerical: bool = False,
    n_rows: Optional[int] = None,
    class_column: Optional[str] = None,
    numerical_filter: float = 1_000_000_000
):
    """
    从实际数据集文件中提取预处理参数并生成配置文件
    
    Args:
        dataset_path: 数据集CSV文件路径
        output_path: 输出配置文件路径
        numerical_features: 数值特征名称列表
        categorical_features: 类别特征名称列表
        top_k: 类别特征Top-K数量
        window_size: 窗口大小
        clip_numerical: 是否裁剪数值特征
        n_rows: 读取的行数限制（None表示全部）
        class_column: 类别列名称（如果存在，会被排除）
        numerical_filter: 数值特征过滤阈值（超出此范围的值会被设为0）
    
    Returns:
        生成的配置字典
    """
    print(f"Reading dataset from: {dataset_path}")
    
    # 读取数据集
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path, nrows=n_rows)
    elif dataset_path.endswith('.feather'):
        df = pd.read_feather(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # 验证特征列是否存在
    all_features = numerical_features + categorical_features
    missing_features = [f for f in all_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
    
    # 提取数值特征参数
    print("\nExtracting numerical feature parameters...")
    numerical_params = {}
    for feat in numerical_features:
        print(f"  Processing {feat}...")
        values = df[feat].values.astype(np.float32)
        
        # 处理非有限值
        values[~np.isfinite(values)] = 0
        
        # 过滤极端值
        values[values < -numerical_filter] = 0
        values[values > numerical_filter] = 0
        
        # 计算最小值和范围
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        v_range = v_max - v_min
        
        numerical_params[feat] = {
            "min": v_min,
            "range": v_range
        }
        
        print(f"    min={v_min:.2f}, max={v_max:.2f}, range={v_range:.2f}")
    
    # 提取类别特征参数
    print("\nExtracting categorical feature parameters...")
    categorical_params = {}
    for feat in categorical_features:
        print(f"  Processing {feat}...")
        values = df[feat].values
        
        # 统计类别频率
        levels, level_counts = np.unique(values, return_counts=True)
        sorted_levels = sorted(zip(levels, level_counts), key=lambda x: x[1], reverse=True)
        
        # 选择Top-K
        top_levels = [s[0] for s in sorted_levels[:top_k]]
        
        # 转换为Python原生类型（处理numpy类型）
        top_levels_list = []
        for level in top_levels:
            if isinstance(level, (np.integer, np.floating)):
                top_levels_list.append(level.item())
            else:
                top_levels_list.append(level)
        
        categorical_params[feat] = {
            "levels": top_levels_list,
            "n_levels": len(top_levels_list)
        }
        
        print(f"    Found {len(levels)} unique values, selected top {len(top_levels_list)}")
        print(f"    Top levels: {top_levels_list[:10]}..." if len(top_levels_list) > 10 else f"    Levels: {top_levels_list}")
    
    # 构建配置
    config = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "numerical_params": numerical_params,
        "categorical_params": categorical_params,
        "window_size": window_size,
        "top_k": top_k,
        "clip_numerical": clip_numerical
    }
    
    # 保存配置
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Config extracted and saved to: {output_path}")
    return config


def create_example_config(
    output_path: str = "preprocessor_config.json",
    dataset_name: str = "nf_unsw_nb15"
):
    """
    创建示例预处理器配置文件（硬编码的示例值）
    
    注意：此函数生成的是示例配置，参数值是硬编码的。
    如需从实际数据集提取参数，请使用 extract_config_from_dataset() 函数。
    
    Args:
        output_path: 输出路径
        dataset_name: 数据集名称（用于生成示例配置）
    """
    # 示例配置（基于UNSW_NB15数据集）
    config = {
        "numerical_features": [
            "IN_PKTS",
            "OUT_PKTS",
            "FLOW_DURATION_MILLISECONDS",
            "IN_BYTES",
            "OUT_BYTES"
        ],
        "categorical_features": [
            "L4_SRC_PORT",
            "L4_DST_PORT",
            "PROTOCOL",
            "L7_PROTO",
            "TCP_FLAGS"
        ],
        "numerical_params": {
            "IN_PKTS": {"min": 0.0, "range": 1000000.0},
            "OUT_PKTS": {"min": 0.0, "range": 1000000.0},
            "FLOW_DURATION_MILLISECONDS": {"min": 0.0, "range": 3600000.0},
            "IN_BYTES": {"min": 0.0, "range": 1000000000.0},
            "OUT_BYTES": {"min": 0.0, "range": 1000000000.0}
        },
        "categorical_params": {
            "L4_SRC_PORT": {
                "levels": [80, 443, 22, 53, 25, 21, 23, 993, 995, 3306, 5432, 8080, 8443, 3389, 1433, 1521, 27017, 6379, 11211, 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010, 5011, 5012],
                "n_levels": 32
            },
            "L4_DST_PORT": {
                "levels": [80, 443, 22, 53, 25, 21, 23, 993, 995, 3306, 5432, 8080, 8443, 3389, 1433, 1521, 27017, 6379, 11211, 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010, 5011, 5012],
                "n_levels": 32
            },
            "PROTOCOL": {
                "levels": [6, 17, 1, 47, 50, 51, 58, 59, 60, 132, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157],
                "n_levels": 32
            },
            "L7_PROTO": {
                "levels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                "n_levels": 32
            },
            "TCP_FLAGS": {
                "levels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                "n_levels": 15
            }
        },
        "window_size": 8,
        "top_k": 32,
        "clip_numerical": False
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Example config saved to: {output_path}")
    print("\n⚠️  Note: You need to update the numerical_params and categorical_params")
    print("   based on your actual training data statistics!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="实时流量推理工具")
    parser.add_argument("--model", type=str, help="ONNX模型路径")
    parser.add_argument("--config", type=str, help="预处理器配置文件路径")
    parser.add_argument("--create-config", type=str, help="创建示例配置文件（硬编码值）")
    parser.add_argument("--extract-config", type=str, help="从数据集提取配置文件路径")
    parser.add_argument("--dataset", type=str, help="数据集CSV文件路径（用于--extract-config）")
    parser.add_argument("--numerical-features", type=str, nargs="+", help="数值特征名称列表（用于--extract-config）")
    parser.add_argument("--categorical-features", type=str, nargs="+", help="类别特征名称列表（用于--extract-config）")
    parser.add_argument("--top-k", type=int, default=32, help="类别特征Top-K数量（默认32）")
    parser.add_argument("--window-size", type=int, default=8, help="窗口大小（默认8）")
    parser.add_argument("--n-rows", type=int, help="读取数据集的行数限制（用于--extract-config）")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--test-data", type=str, help="测试数据集CSV文件路径（用于--test）")
    parser.add_argument("--test-n-rows", type=int, default=10, help="测试时读取的数据行数（默认10）")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_example_config(args.create_config)
        exit(0)
    
    if args.extract_config:
        if not args.dataset:
            print("Error: --extract-config requires --dataset")
            parser.print_help()
            exit(1)
        if not args.numerical_features or not args.categorical_features:
            print("Error: --extract-config requires --numerical-features and --categorical-features")
            print("\nExample:")
            print("  python realtime_inference.py --extract-config config.json \\")
            print("    --dataset dataset.csv \\")
            print("    --numerical-features IN_PKTS OUT_PKTS IN_BYTES OUT_BYTES FLOW_DURATION_MILLISECONDS \\")
            print("    --categorical-features L4_SRC_PORT L4_DST_PORT PROTOCOL L7_PROTO TCP_FLAGS")
            exit(1)
        
        extract_config_from_dataset(
            dataset_path=args.dataset,
            output_path=args.extract_config,
            numerical_features=args.numerical_features,
            categorical_features=args.categorical_features,
            top_k=args.top_k,
            window_size=args.window_size,
            n_rows=args.n_rows
        )
        exit(0)
    
    if args.test:
        # 测试代码
        print("=" * 60)
        print("Testing TrafficPreprocessor and ONNXInferenceEngine")
        print("=" * 60)
        
        # 检查必需的参数
        if not args.config:
            print("Error: --config is required for --test")
            print("Please provide a preprocessor config file")
            parser.print_help()
            exit(1)
        
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            exit(1)
        
        # 加载预处理器
        print(f"\n[1] Loading preprocessor from: {args.config}")
        preprocessor = TrafficPreprocessor.load_from_json(args.config)
        print(f"    Numerical features: {preprocessor.numerical_features}")
        print(f"    Categorical features: {preprocessor.categorical_features}")
        print(f"    Window size: {preprocessor.window_size}")
        
        # 读取测试数据
        if args.test_data:
            if not os.path.exists(args.test_data):
                print(f"Error: Test data file not found: {args.test_data}")
                exit(1)
            
            print(f"\n[2] Loading test data from: {args.test_data}")
            test_df = pd.read_csv(args.test_data, nrows=args.test_n_rows)
            print(f"    Loaded {len(test_df)} rows")
            print(f"    Columns: {list(test_df.columns)}")
            
            # 检查必需的列是否存在
            all_required_features = preprocessor.numerical_features + preprocessor.categorical_features
            missing_cols = [col for col in all_required_features if col not in test_df.columns]
            if missing_cols:
                print(f"\n⚠️  Warning: Missing columns in dataset: {missing_cols}")
                print(f"    Available columns: {list(test_df.columns)}")
                print(f"    Required columns: {all_required_features}")
            
            # 将DataFrame转换为字典列表
            test_flows = []
            for idx, row in test_df.iterrows():
                flow_dict = {}
                # 添加数值特征
                for feat in preprocessor.numerical_features:
                    if feat in test_df.columns:
                        # 处理NaN值
                        value = row[feat]
                        if pd.isna(value):
                            value = 0.0
                        flow_dict[feat] = float(value)
                    else:
                        print(f"⚠️  Warning: Missing numerical feature '{feat}' in row {idx}, using 0")
                        flow_dict[feat] = 0.0
                
                # 添加类别特征
                for feat in preprocessor.categorical_features:
                    if feat in test_df.columns:
                        value = row[feat]
                        # 处理NaN值
                        if pd.isna(value):
                            # 使用第一个已知级别作为默认值
                            if preprocessor.categorical_params[feat]['levels']:
                                value = preprocessor.categorical_params[feat]['levels'][0]
                            else:
                                value = 0
                        flow_dict[feat] = value
                    else:
                        print(f"⚠️  Warning: Missing categorical feature '{feat}' in row {idx}, using default")
                        # 使用第一个已知级别作为默认值
                        if preprocessor.categorical_params[feat]['levels']:
                            flow_dict[feat] = preprocessor.categorical_params[feat]['levels'][0]
                        else:
                            flow_dict[feat] = 0
                
                test_flows.append(flow_dict)
            
            print(f"    Converted {len(test_flows)} flows to dictionaries")
            
        else:
            # 如果没有提供测试数据，使用随机生成的数据（向后兼容）
            print("\n[2] No test data provided, generating random test data...")
            test_flows = []
            for i in range(max(args.test_n_rows, preprocessor.window_size)):
                test_flows.append({
                    'IN_PKTS': 100 + i * 10,
                    'OUT_PKTS': 200 + i * 10,
                    'FLOW_DURATION_MILLISECONDS': 1000 + i * 100,
                    'IN_BYTES': 10000 + i * 1000,
                    'OUT_BYTES': 20000 + i * 2000,
                    'L4_SRC_PORT': 80 + (i % 5),
                    'L4_DST_PORT': 443 + (i % 5),
                    'PROTOCOL': 6 + (i % 3),
                    'L7_PROTO': i % 10,
                    'TCP_FLAGS': i % 15
                })
        
        # 确保有足够的数据构建窗口
        if len(test_flows) < preprocessor.window_size:
            print(f"\n⚠️  Warning: Need at least {preprocessor.window_size} flows for window, got {len(test_flows)}")
            print(f"    Using first {preprocessor.window_size} flows")
            test_flows = test_flows[:preprocessor.window_size]
        
        # 构建窗口
        print(f"\n[3] Building window from {len(test_flows)} flows...")
        window_data = preprocessor.build_window(test_flows)
        print("\nWindow data shapes:")
        for name, data in window_data.items():
            print(f"  {name}: {data.shape}")
        
        # 加载模型（如果提供）
        if args.model and os.path.exists(args.model):
            print(f"\n[4] Loading ONNX model from: {args.model}")
            engine = ONNXInferenceEngine(args.model)
            print(test_flows)
            # 预测
            print("\n[5] Running inference...")
            result = engine.predict_batch(test_flows, preprocessor)
            print(f"\nPrediction result shape: {result.shape}")
            print(f"Prediction result (first few): {result[0][:5] if len(result[0]) > 5 else result[0]}")
            
            # 如果有多个样本，显示所有结果
            if len(result) > 1:
                print(f"\nAll prediction results:")
                for i, pred in enumerate(result):
                    print(f"  Sample {i}: {pred}")
        else:
            print(f"\n⚠️  Model not found: {args.model}")
            print("   Skipping inference test.")
            print("   Use --model to specify model path for inference test.")
        
        print("\n✅ Test completed!")
    else:
        if not args.model or not os.path.exists(args.model):
            print("Error: Model path is required and must exist")
            parser.print_help()
            exit(1)
        
        if not args.config or not os.path.exists(args.config):
            print("Error: Config path is required and must exist")
            print("Create a config file with --create-config option")
            parser.print_help()
            exit(1)
        
        # 加载预处理器和模型
        preprocessor = TrafficPreprocessor.load_from_json(args.config)
        engine = ONNXInferenceEngine(args.model)
        
        print("\n✅ Ready for inference!")
        print("Use the classes programmatically:")
        print("  from realtime_inference import TrafficPreprocessor, ONNXInferenceEngine")
        print("  preprocessor = TrafficPreprocessor.load_from_json('config.json')")
        print("  engine = ONNXInferenceEngine('model.onnx')")
        print("  result = engine.predict_batch(flows, preprocessor)")

