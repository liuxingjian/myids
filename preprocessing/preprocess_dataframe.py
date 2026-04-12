"""
DataFrame 批量预处理工具

该工具用于加载预处理配置文件，并对 DataFrame 进行批量预处理。
支持数值特征的对数归一化和类别特征的 OneHot 编码。

依赖：
    pip install numpy pandas

使用示例：
    from preprocess_dataframe import DataFramePreprocessor
    
    # 加载预处理器
    preprocessor = DataFramePreprocessor.load_from_json('config.json')
    
    # 预处理 DataFrame
    df = pd.read_csv('data.csv')
    preprocessed_df = preprocessor.preprocess_dataframe(df)
    
    # 或者使用命令行
    python preprocess_dataframe.py \
        --config config.json \
        --input data.csv \
        --output preprocessed_data.csv
"""
import json
import os
from typing import List, Dict, Optional
import numpy as np
import pandas as pd


class DataFramePreprocessor:
    """
    DataFrame 批量预处理器
    
    实现与训练时相同的预处理逻辑：
    - 数值特征：对数归一化（保持为单列）
    - 类别特征：Top-K编码 + OneHot编码（展开为多列）
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
            window_size: 窗口大小（用于时序模型，此脚本中不使用）
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
        
        # 处理非有限值
        values[~np.isfinite(values)] = 0
        
        # 中心化
        values -= col_min
        
        # 对数变换 (保证内部总是正数，避免 RuntimeWarning)
        # 用 np.maximum 将值限制在 0 及以上，也就是 values + 1 最小为 1
        safe_values = np.maximum(values + 1.0, 1.0)
        col_values = np.log(safe_values)
        
        # 归一化
        col_values *= 1.0 / np.log(col_range + 1)
        
        # 可选裁剪
        if self.clip_numerical:
            col_values = np.clip(col_values, 0.0, 1.0)
        
        return col_values.astype(np.float32)
    
    def transform_categorical(
        self, 
        feature_name: str, 
        values: np.ndarray
    ) -> np.ndarray:
        """
        转换类别特征为OneHot编码
        
        Args:
            feature_name: 特征名称
            values: 特征值数组
            
        Returns:
            OneHot编码数组，形状为 (n_samples, n_levels)
        """
        params = self.categorical_params[feature_name]
        encoded_levels = params['levels']
        n_levels = params.get('n_levels', len(encoded_levels))
        
        values = np.asarray(values)
        # 默认值0表示未知，避免未知被错误编码为第一个已知类别。
        result_values = np.zeros(len(values), dtype=np.uint32)
        
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
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        drop_original: bool = False,
        keep_other_columns: bool = True,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        预处理整个 DataFrame
        
        Args:
            df: 输入的 DataFrame
            drop_original: 是否删除原始特征列（默认False，保留原始列）
            keep_other_columns: 是否保留不在配置中的其他列（默认True）
            verbose: 是否打印处理进度
        """
        # 复制 DataFrame 以避免修改原始数据
        result_df = df.copy()
        
        # 检查必需的列是否存在
        all_features = self.numerical_features + self.categorical_features
        missing_features = [f for f in all_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in DataFrame: {missing_features}")
        
        # 处理数值特征
        if verbose: print("Processing numerical features...")
        new_num_cols = {}
        for feat in self.numerical_features:
            if feat not in df.columns:
                if verbose: print(f"  Warning: {feat} not found in DataFrame, skipping")
                continue
            
            values = df[feat].values
            transformed = self.transform_numerical(feat, values)
            
            # 创建新列名（如果drop_original=False，则添加后缀）
            new_col_name = f"{feat}_preprocessed" if not drop_original else feat
            new_num_cols[new_col_name] = transformed
            
            if verbose: print(f"  {feat}: {len(transformed)} values processed")
            
        if new_num_cols:
            num_df = pd.DataFrame(new_num_cols, index=result_df.index)
            result_df = pd.concat([result_df, num_df], axis=1)
        
        # 处理类别特征（OneHot编码，展开为多列）
        if verbose: print("\nProcessing categorical features...")
        new_cols_dict = {}
        for feat in self.categorical_features:
            if feat not in df.columns:
                if verbose: print(f"  Warning: {feat} not found in DataFrame, skipping")
                continue
            
            values = df[feat].values
            onehot = self.transform_categorical(feat, values)
            
            # 获取类别数量
            n_levels = self.categorical_params[feat].get('n_levels', len(self.categorical_params[feat]['levels']))
            
            # 创建新列名
            if drop_original:
                # 如果删除原始列，直接使用特征名作为前缀
                col_prefix = feat
            else:
                # 否则添加后缀
                col_prefix = f"{feat}_onehot"
            
            # 为每个OneHot维度创建一列
            for i in range(n_levels):
                col_name = f"{col_prefix}_{i}"
                new_cols_dict[col_name] = onehot[:, i]
            
            if verbose: print(f"  {feat}: {n_levels} one-hot columns created")
            
        if new_cols_dict:
            new_df = pd.DataFrame(new_cols_dict, index=result_df.index)
            result_df = pd.concat([result_df, new_df], axis=1)
        
        # 如果drop_original=True，删除原始特征列
        # 注意：只删除原始DataFrame中存在的列，避免删除预处理后的新列
        if drop_original:
            # 只删除原始DataFrame中存在的列
            original_cols_to_drop = [f for f in all_features if f in df.columns]
            result_df = result_df.drop(columns=original_cols_to_drop, errors='ignore')
        
        # 如果keep_other_columns=False，只保留预处理后的列
        if not keep_other_columns:
            # 收集所有预处理后的列名
            preprocessed_cols = []
            # 数值特征列
            for feat in self.numerical_features:
                col_name = feat if drop_original else f"{feat}_preprocessed"
                if col_name in result_df.columns:
                    preprocessed_cols.append(col_name)
            # 类别特征列
            for feat in self.categorical_features:
                n_levels = self.categorical_params[feat].get('n_levels', len(self.categorical_params[feat]['levels']))
                col_prefix = feat if drop_original else f"{feat}_onehot"
                for i in range(n_levels):
                    col_name = f"{col_prefix}_{i}"
                    if col_name in result_df.columns:
                        preprocessed_cols.append(col_name)
            
            result_df = result_df[preprocessed_cols]
        
        return result_df
    
    def dataframe_to_model_inputs(
        self,
        preprocessed_df: pd.DataFrame,
        window_size: Optional[int] = None,
        input_name_prefix: str = "input_",
        batch_size: int = 1,
        model_expected_dims: Optional[Dict[str, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        将预处理后的DataFrame转换为模型可接受的输入格式
        
        Args:
            preprocessed_df: 预处理后的DataFrame（已调用preprocess_dataframe处理）
            window_size: 窗口大小，如果为None则使用self.window_size
            input_name_prefix: 模型输入名称前缀（默认为"input_"）
            batch_size: 批次大小（默认为1）
            model_expected_dims: 模型期望的特征维度字典，格式为 {feature_name: expected_dim}
                                如果为None，则使用配置中的n_levels
                                如果实际维度小于期望维度，会在特征维度上填充零
                                如果实际维度大于期望维度，会截取前N个维度
            
        Returns:
            字典，键为模型输入名称，值为numpy数组，形状为 (batch, sequence, feature)
            
        注意：
            - 数值特征：形状为 (batch, sequence, 1)
            - 类别特征：形状为 (batch, sequence, expected_dim)
            - 如果DataFrame行数不足window_size，会在序列维度填充零
            - 如果DataFrame行数超过window_size，会取最后window_size行
        """
        if window_size is None:
            window_size = self.window_size
        
        # 确定实际使用的行数
        n_rows = len(preprocessed_df)
        if n_rows < window_size:
            print(f"Warning: DataFrame has {n_rows} rows, but window_size is {window_size}. "
                  f"Will pad with zeros in sequence dimension.")
            actual_rows = n_rows
        else:
            actual_rows = window_size
        
        # 准备输入字典
        model_inputs = {}
        
        # 处理数值特征
        for feat in self.numerical_features:
            # 查找预处理后的列名（可能是 feat 或 feat_preprocessed）
            col_name = None
            if feat in preprocessed_df.columns:
                col_name = feat
            elif f"{feat}_preprocessed" in preprocessed_df.columns:
                col_name = f"{feat}_preprocessed"
            
            if col_name is None:
                raise ValueError(
                    f"Preprocessed column for numerical feature '{feat}' not found. "
                    f"Expected '{feat}' or '{feat}_preprocessed'. "
                    f"Available columns: {list(preprocessed_df.columns)}"
                )
            
            # 获取数据（取最后window_size行）
            if n_rows >= window_size:
                values = preprocessed_df[col_name].values[-window_size:]
            else:
                values = preprocessed_df[col_name].values
            
            # 转换为 (batch, sequence, 1) 形状
            # 如果数据不足，在序列维度填充零
            if len(values) < window_size:
                padded_values = np.zeros(window_size, dtype=np.float32)
                padded_values[:len(values)] = values.astype(np.float32)
                values = padded_values
            else:
                values = values.astype(np.float32)
            
            # 重塑为 (batch, sequence, 1)
            values = values.reshape(1, window_size, 1)
            
            # 如果需要更大的batch_size，复制数据
            if batch_size > 1:
                values = np.repeat(values, batch_size, axis=0)
            
            # 模型输入名称
            input_name = f"{input_name_prefix}{feat}"
            model_inputs[input_name] = values
        
        # 处理类别特征（OneHot编码）
        for feat in self.categorical_features:
            # 获取实际预处理后的维度
            actual_n_levels = self.categorical_params[feat].get('n_levels', len(self.categorical_params[feat]['levels']))
            
            # 获取模型期望的维度
            if model_expected_dims and feat in model_expected_dims:
                expected_dim = model_expected_dims[feat]
            else:
                expected_dim = actual_n_levels
            
            # 查找OneHot列（可能是 feat_0, feat_1, ... 或 feat_onehot_0, feat_onehot_1, ...）
            onehot_cols = []
            col_prefix = None
            
            # 尝试不同的列名前缀
            if all(f"{feat}_{i}" in preprocessed_df.columns for i in range(actual_n_levels)):
                col_prefix = feat
            elif all(f"{feat}_onehot_{i}" in preprocessed_df.columns for i in range(actual_n_levels)):
                col_prefix = f"{feat}_onehot"
            else:
                raise ValueError(
                    f"OneHot columns for categorical feature '{feat}' not found. "
                    f"Expected '{feat}_0' to '{feat}_{actual_n_levels-1}' or "
                    f"'{feat}_onehot_0' to '{feat}_onehot_{actual_n_levels-1}'. "
                    f"Available columns: {list(preprocessed_df.columns)}"
                )
            
            # 收集所有OneHot列的数据
            onehot_data = []
            for i in range(actual_n_levels):
                col_name = f"{col_prefix}_{i}"
                if col_name not in preprocessed_df.columns:
                    raise ValueError(f"OneHot column '{col_name}' not found")
                onehot_data.append(preprocessed_df[col_name].values)
            
            # 组合成 (n_rows, actual_n_levels) 形状
            onehot_matrix = np.column_stack(onehot_data).astype(np.float32)
            
            # 取最后window_size行
            if n_rows >= window_size:
                onehot_matrix = onehot_matrix[-window_size:]
                valid_length = window_size
            else:
                valid_length = len(onehot_matrix)
                # 如果数据不足，在序列维度填充零
                if len(onehot_matrix) < window_size:
                    padded = np.zeros((window_size, actual_n_levels), dtype=np.float32)
                    padded[:len(onehot_matrix)] = onehot_matrix
                    onehot_matrix = padded

            # 标记未知类别（真实样本内 one-hot 全零）的位置。
            unknown_mask = np.zeros(window_size, dtype=np.float32)
            if valid_length > 0:
                unknown_mask[:valid_length] = (onehot_matrix[:valid_length].sum(axis=1) == 0).astype(np.float32)
            
            # 调整特征维度以匹配模型期望
            if actual_n_levels < expected_dim:
                # 如果实际维度小于期望维度，在特征维度上填充零
                padded_matrix = np.zeros((window_size, expected_dim), dtype=np.float32)
                padded_matrix[:, :actual_n_levels] = onehot_matrix
                # 当模型维度大于配置维度时，把 unknown 信号放到第一个扩展维度。
                padded_matrix[:, actual_n_levels] = unknown_mask
                onehot_matrix = padded_matrix
                #print(f"  {feat}: 维度从 {actual_n_levels} 填充到 {expected_dim}")
            elif actual_n_levels > expected_dim:
                # 如果实际维度大于期望维度，截取前N个维度
                onehot_matrix = onehot_matrix[:, :expected_dim]
                #print(f"  {feat}: 维度从 {actual_n_levels} 截取到 {expected_dim}")
            
            # 重塑为 (batch, sequence, expected_dim)
            onehot_matrix = onehot_matrix.reshape(1, window_size, expected_dim)
            
            # 如果需要更大的batch_size，复制数据
            if batch_size > 1:
                onehot_matrix = np.repeat(onehot_matrix, batch_size, axis=0)
            
            # 模型输入名称
            input_name = f"{input_name_prefix}{feat}"
            model_inputs[input_name] = onehot_matrix
        
        return model_inputs
    
    def dataframe_to_model_inputs_batch(
        self,
        preprocessed_df: pd.DataFrame,
        window_size: Optional[int] = None,
        input_name_prefix: str = "input_",
        stride: Optional[int] = None,
        model_expected_dims: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        将预处理后的DataFrame转换为多个窗口批次（用于批量推理）
        
        Args:
            preprocessed_df: 预处理后的DataFrame
            window_size: 窗口大小，如果为None则使用self.window_size
            input_name_prefix: 模型输入名称前缀
            stride: 窗口滑动步长，如果为None则等于window_size（无重叠）
            model_expected_dims: 模型期望的特征维度字典，格式为 {feature_name: expected_dim}
            
        Returns:
            列表，每个元素是一个模型输入字典
        """
        if window_size is None:
            window_size = self.window_size
        if stride is None:
            stride = window_size
        
        n_rows = len(preprocessed_df)
        if n_rows < window_size:
            # 如果数据不足一个窗口，返回一个填充零的窗口
            return [self.dataframe_to_model_inputs(
                preprocessed_df, window_size, input_name_prefix, 
                batch_size=1, model_expected_dims=model_expected_dims
            )]
        
        # 生成多个窗口
        batch_inputs = []
        start_idx = 0
        
        while start_idx + window_size <= n_rows:
            # 提取当前窗口的DataFrame
            window_df = preprocessed_df.iloc[start_idx:start_idx + window_size].copy()
            
            # 转换为模型输入
            model_inputs = self.dataframe_to_model_inputs(
                window_df, window_size, input_name_prefix, 
                batch_size=1, model_expected_dims=model_expected_dims
            )
            batch_inputs.append(model_inputs)
            
            start_idx += stride
        
        return batch_inputs
    
    @classmethod
    def load_from_json(cls, json_path: str) -> 'DataFramePreprocessor':
        """
        从JSON文件加载预处理器配置
        
        Args:
            json_path: JSON配置文件路径
            
        Returns:
            DataFramePreprocessor实例
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


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="批量预处理 DataFrame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 预处理 CSV 文件
  python preprocess_dataframe.py \\
      --config config.json \\
      --input data.csv \\
      --output preprocessed_data.csv \\
      --drop-original
  
  # 保留原始列
  python preprocess_dataframe.py \\
      --config config.json \\
      --input data.csv \\
      --output preprocessed_data.csv
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="预处理配置文件路径（JSON格式）"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入数据文件路径（CSV或Feather格式）"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出数据文件路径（CSV或Feather格式）"
    )
    parser.add_argument(
        "--drop-original",
        action="store_true",
        help="是否删除原始特征列（默认False，保留原始列）"
    )
    parser.add_argument(
        "--keep-other-columns",
        action="store_true",
        default=True,
        help="是否保留不在配置中的其他列（默认True）"
    )
    parser.add_argument(
        "--no-keep-other-columns",
        dest="keep_other_columns",
        action="store_false",
        help="不保留不在配置中的其他列"
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        help="读取输入文件的行数限制（None表示全部）"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)
    
    # 加载预处理器
    print(f"Loading preprocessor config from: {args.config}")
    try:
        preprocessor = DataFramePreprocessor.load_from_json(args.config)
        print(f"  Loaded {len(preprocessor.numerical_features)} numerical features")
        print(f"  Loaded {len(preprocessor.categorical_features)} categorical features")
    except Exception as e:
        print(f"Error loading config: {e}")
        exit(1)
    
    # 读取输入数据
    print(f"\nReading input data from: {args.input}")
    try:
        if args.input.endswith('.csv'):
            df = pd.read_csv(args.input, nrows=args.n_rows)
        elif args.input.endswith('.feather'):
            df = pd.read_feather(args.input)
        else:
            print(f"Error: Unsupported input file format: {args.input}")
            print("Supported formats: CSV, Feather")
            exit(1)
        
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error reading input file: {e}")
        exit(1)
    
    # 预处理数据
    print("\nPreprocessing DataFrame...")
    try:
        preprocessed_df = preprocessor.preprocess_dataframe(
            df,
            drop_original=args.drop_original,
            keep_other_columns=args.keep_other_columns
        )
        print(f"\n  Preprocessed DataFrame shape: {preprocessed_df.shape}")
        print(f"  Columns: {list(preprocessed_df.columns)}")
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 保存结果
    print(f"\nSaving preprocessed data to: {args.output}")
    try:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        
        if args.output.endswith('.csv'):
            preprocessed_df.to_csv(args.output, index=False)
        elif args.output.endswith('.feather'):
            preprocessed_df.to_feather(args.output)
        else:
            print(f"Error: Unsupported output file format: {args.output}")
            print("Supported formats: CSV, Feather")
            exit(1)
        
        print(f"✅ Preprocessed data saved successfully!")
    except Exception as e:
        print(f"Error saving output file: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    #main()
    # 加载预处理器
    preprocessor = DataFramePreprocessor.load_from_json('../config/config.json')
    
    # 手动创建测试 DataFrame
    df = pd.DataFrame({
        # 数值特征
        'IN_PKTS': [10, 20, 30, 15, 25, 20, 30, 15, 25],
        'OUT_PKTS': [5, 15, 25, 10, 20, 15, 25, 10, 20],
        'IN_BYTES': [100, 200, 300, 150, 250, 200, 300, 150, 250],
        'OUT_BYTES': [200, 300, 400, 250, 350, 300, 400, 250, 350],
        'FLOW_DURATION_MILLISECONDS': [100, 200, 300, 150, 250, 200, 300, 150, 250],
        # 类别特征
        'L4_SRC_PORT': [7000, 52990, 58578, 7000, 5353, 52990, 58578, 7000, 5353],
        'L4_DST_PORT': [1900, 7007, 42844, 1900, 5353, 7007, 42844, 1900, 5353],
        'PROTOCOL': [6, 17, 6, 17, 6, 17, 6, 17, 6],
        'L7_PROTO': ['TLS', 'SSDP', 'MQTT', 'TLS', 'MDNS', 'SSDP', 'MQTT', 'TLS', 'MDNS'],
        'TCP_FLAGS': [24, 0, 24, 0, 24, 0, 24, 0, 24]
    })
    
    print("原始 DataFrame:")
    print(df)
    print(f"\n原始列: {list(df.columns)}")
    
    # 预处理 DataFrame
    preprocessed_df = preprocessor.preprocess_dataframe(df, keep_other_columns=False)

    print(f"\n预处理后的列: {list(preprocessed_df.columns)}")
    print(f"\n预处理后的 DataFrame:")
    print(preprocessed_df.head())
    
    # 转换为模型输入格式
    print("\n" + "="*60)
    print("转换为模型输入格式")
    print("="*60)
    
    # 定义模型期望的特征维度（根据 models/READMD）
    model_expected_dims = {
        'L4_SRC_PORT': 32,
        'L4_DST_PORT': 32,
        'PROTOCOL': 32,
        'L7_PROTO': 32,
        'TCP_FLAGS': 15
    }
    
    try:
        model_inputs = preprocessor.dataframe_to_model_inputs(
            preprocessed_df,
            window_size=preprocessor.window_size,
            input_name_prefix="input_",
            model_expected_dims=model_expected_dims
        )
        
        print(f"\n模型输入字典（共 {len(model_inputs)} 个输入）:")
        for input_name, input_data in model_inputs.items():
            print(f"  {input_name}: shape={input_data.shape}, dtype={input_data.dtype}")
            print(f"    min={input_data.min():.4f}, max={input_data.max():.4f}, mean={input_data.mean():.4f}")
        
        # 演示批量转换（如果数据足够）
        if len(preprocessed_df) >= preprocessor.window_size * 2:
            print(f"\n批量转换（滑动窗口）:")
            batch_inputs = preprocessor.dataframe_to_model_inputs_batch(
                preprocessed_df,
                window_size=preprocessor.window_size,
                stride=preprocessor.window_size,
                model_expected_dims=model_expected_dims
            )
            print(f"  生成了 {len(batch_inputs)} 个窗口批次")
            for i, batch_input in enumerate(batch_inputs):
                print(f"  批次 {i+1}: {len(batch_input)} 个输入")
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()

