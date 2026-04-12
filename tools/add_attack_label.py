"""
为数据集添加 Attack 标签列

该脚本读取 CSV 文件，根据 IPV4_SRC_ADDR 列的值添加 Attack 标签列：
- 如果 IPV4_SRC_ADDR == '192.168.3.146'，则 Attack = 'attack'
- 否则 Attack = 'Benign'

依赖：
    pip install pandas

使用示例：
    # 直接修改原文件
    python add_attack_label.py \
        --input converted_flows_nfv1_0129.csv \
        --inplace
    
    # 保存为新文件
    python add_attack_label.py \
        --input converted_flows_nfv1_0129.csv \
        --output converted_flows_nfv1_0129_labeled.csv
"""
import argparse
import os
import pandas as pd


def add_attack_label(
    input_path: str,
    output_path: str = None,
    attack_ip: str = '192.168.3.146',
    attack_label: str = 'attack',
    benign_label: str = 'Benign',
    column_name: str = 'IPV4_SRC_ADDR',
    label_column_name: str = 'Attack'
):
    """
    为数据集添加 Attack 标签列
    
    Args:
        input_path: 输入 CSV 文件路径
        output_path: 输出 CSV 文件路径（如果为 None，则覆盖原文件）
        attack_ip: 攻击 IP 地址（默认 '192.168.3.146'）
        attack_label: 攻击标签值（默认 'attack'）
        benign_label: 正常标签值（默认 'Benign'）
        column_name: 用于判断的列名（默认 'IPV4_SRC_ADDR'）
        label_column_name: 新添加的标签列名（默认 'Attack'）
    
    Returns:
        处理后的 DataFrame
    """
    print(f"Reading CSV file: {input_path}")
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # 读取 CSV 文件
    print("Loading data...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # 检查必需的列是否存在
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # 检查标签列是否已存在
    if label_column_name in df.columns:
        print(f"  Warning: Column '{label_column_name}' already exists. It will be overwritten.")
    
    # 添加 Attack 标签列
    print(f"\nAdding '{label_column_name}' column based on '{column_name}' column...")
    print(f"  Attack IP: {attack_ip}")
    print(f"  Attack label: {attack_label}")
    print(f"  Benign label: {benign_label}")
    
    # 根据 IPV4_SRC_ADDR 列的值设置 Attack 标签
    df[label_column_name] = df[column_name].apply(
        lambda x: attack_label if str(x) == attack_ip else benign_label
    )
    
    # 统计标签分布
    label_counts = df[label_column_name].value_counts()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        percentage = count / len(df) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    # 确定输出路径
    if output_path is None:
        output_path = input_path
        print(f"\nSaving to original file: {output_path}")
    else:
        print(f"\nSaving to new file: {output_path}")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Successfully added '{label_column_name}' column and saved to: {output_path}")
    print(f"   Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description="为数据集添加 Attack 标签列",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 直接修改原文件
  python add_attack_label.py \\
      --input converted_flows_nfv1_0129.csv \\
      --inplace
  
  # 保存为新文件
  python add_attack_label.py \\
      --input converted_flows_nfv1_0129.csv \\
      --output converted_flows_nfv1_0129_labeled.csv
  
  # 自定义攻击 IP 和标签
  python add_attack_label.py \\
      --input data.csv \\
      --output labeled_data.csv \\
      --attack-ip 192.168.1.100 \\
      --attack-label malicious \\
      --benign-label normal
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 CSV 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出 CSV 文件路径（如果不指定且不使用 --inplace，则覆盖原文件）"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="直接修改原文件（等同于 --output 设为输入文件路径）"
    )
    parser.add_argument(
        "--attack-ip",
        type=str,
        default='192.168.3.146',
        help="攻击 IP 地址（默认 '192.168.3.146'）"
    )
    parser.add_argument(
        "--attack-label",
        type=str,
        default='attack',
        help="攻击标签值（默认 'attack'）"
    )
    parser.add_argument(
        "--benign-label",
        type=str,
        default='Benign',
        help="正常标签值（默认 'Benign'）"
    )
    parser.add_argument(
        "--column-name",
        type=str,
        default='IPV4_SRC_ADDR',
        help="用于判断的列名（默认 'IPV4_SRC_ADDR'）"
    )
    parser.add_argument(
        "--label-column-name",
        type=str,
        default='Attack',
        help="新添加的标签列名（默认 'Attack'）"
    )
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.inplace:
        output_path = args.input
    else:
        output_path = args.output
    
    try:
        add_attack_label(
            input_path=args.input,
            output_path=output_path,
            attack_ip=args.attack_ip,
            attack_label=args.attack_label,
            benign_label=args.benign_label,
            column_name=args.column_name,
            label_column_name=args.label_column_name
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

