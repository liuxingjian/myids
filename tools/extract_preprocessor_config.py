"""
从数据集提取预处理器配置的工具

该工具用于从实际数据集中提取预处理参数（数值特征的min/range，类别特征的Top-K levels），
并生成JSON配置文件，供实时推理工具使用。

依赖：
    pip install numpy pandas

使用示例：
    # 从数据集提取配置
    python extract_preprocessor_config.py \
        --dataset /home/yaojiahao/ydd_ws/new/yddflow/example/converted_flows_nfv1_0130_balanced.csv \
        --output config.json \
        --numerical-features IN_PKTS OUT_PKTS IN_BYTES OUT_BYTES FLOW_DURATION_MILLISECONDS \
        --categorical-features L4_SRC_PORT L4_DST_PORT PROTOCOL L7_PROTO TCP_FLAGS \
        --top-k 32 \
        --window-size 8
    
    # 创建示例配置（硬编码值）
    python extract_preprocessor_config.py \
        --create-example example_config.json
"""
import json
import os
from typing import List, Optional
import numpy as np
import pandas as pd


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
    
    parser = argparse.ArgumentParser(
        description="从数据集提取预处理器配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 从数据集提取配置
  python extract_preprocessor_config.py \\
      --dataset dataset.csv \\
      --output config.json \\
      --numerical-features IN_PKTS OUT_PKTS IN_BYTES OUT_BYTES FLOW_DURATION_MILLISECONDS \\
      --categorical-features L4_SRC_PORT L4_DST_PORT PROTOCOL L7_PROTO TCP_FLAGS \\
      --top-k 32 \\
      --window-size 8
  
  # 创建示例配置（硬编码值）
  python extract_preprocessor_config.py \\
      --create-example example_config.json
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="数据集文件路径（支持CSV和Feather格式）"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出配置文件路径（JSON格式）"
    )
    parser.add_argument(
        "--numerical-features",
        type=str,
        nargs="+",
        help="数值特征名称列表"
    )
    parser.add_argument(
        "--categorical-features",
        type=str,
        nargs="+",
        help="类别特征名称列表"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="类别特征Top-K数量（默认32）"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="窗口大小（默认8）"
    )
    parser.add_argument(
        "--clip-numerical",
        action="store_true",
        help="是否裁剪数值特征到[0,1]"
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        help="读取数据集的行数限制（None表示全部）"
    )
    parser.add_argument(
        "--class-column",
        type=str,
        help="类别列名称（如果存在，会被排除）"
    )
    parser.add_argument(
        "--numerical-filter",
        type=float,
        default=1_000_000_000,
        help="数值特征过滤阈值（超出此范围的值会被设为0，默认1e9）"
    )
    parser.add_argument(
        "--create-example",
        type=str,
        help="创建示例配置文件（硬编码值），参数为输出路径"
    )
    
    args = parser.parse_args()
    
    # 如果指定了创建示例配置
    if args.create_example:
        create_example_config(args.create_example)
        exit(0)
    
    # 验证必需参数
    if not args.dataset:
        print("Error: --dataset is required")
        parser.print_help()
        exit(1)
    
    if not args.output:
        print("Error: --output is required")
        parser.print_help()
        exit(1)
    
    if not args.numerical_features:
        print("Error: --numerical-features is required")
        parser.print_help()
        exit(1)
    
    if not args.categorical_features:
        print("Error: --categorical-features is required")
        parser.print_help()
        exit(1)
    
    # 检查数据集文件是否存在
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        exit(1)
    
    # 提取配置
    try:
        extract_config_from_dataset(
            dataset_path=args.dataset,
            output_path=args.output,
            numerical_features=args.numerical_features,
            categorical_features=args.categorical_features,
            top_k=args.top_k,
            window_size=args.window_size,
            clip_numerical=args.clip_numerical,
            n_rows=args.n_rows,
            class_column=args.class_column,
            numerical_filter=args.numerical_filter
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

