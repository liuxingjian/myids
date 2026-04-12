"""
平衡数据集中的 Attack 和 Benign 标签样本数量

该脚本读取包含 Attack 标签列的 CSV 文件，将两类样本的数量平衡到一致：
- 统计 Attack 和 Benign 标签的样本数量
- 取较小的数量作为目标数量
- 对两类样本都进行随机下采样到目标数量
- 合并并保存到新的 CSV 文件

依赖：
    pip install pandas numpy

使用示例：
    python balance_dataset.py \
        --input converted_flows_nfv1_0130_labeled.csv \
        --output converted_flows_nfv1_0130_balanced.csv \
        --label-column Attack \
        --attack-label attack \
        --benign-label Benign \
        --seed 42
"""
import argparse
import os
import pandas as pd
import numpy as np


def balance_dataset(
    input_path: str,
    output_path: str,
    label_column: str = 'Attack',
    attack_label: str = 'attack',
    benign_label: str = 'Benign',
    seed: int = 42,
    strategy: str = 'downsample'
):
    """
    平衡数据集中的 Attack 和 Benign 标签样本数量
    
    Args:
        input_path: 输入 CSV 文件路径
        output_path: 输出 CSV 文件路径
        label_column: 标签列名（默认 'Attack'）
        attack_label: 攻击标签值（默认 'attack'）
        benign_label: 正常标签值（默认 'Benign'）
        seed: 随机种子（默认 42）
        strategy: 平衡策略，'downsample'（下采样到较小数量）或 'upsample'（上采样到较大数量）
    
    Returns:
        平衡后的 DataFrame
    """
    print(f"Reading CSV file: {input_path}")
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # 读取 CSV 文件
    print("Loading data...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # 检查标签列是否存在
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # 统计标签分布
    label_counts = df[label_column].value_counts()
    print(f"\nOriginal label distribution:")
    for label, count in label_counts.items():
        percentage = count / len(df) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    # 获取两类样本
    attack_mask = df[label_column] == attack_label
    benign_mask = df[label_column] == benign_label
    
    attack_df = df[attack_mask].copy()
    benign_df = df[benign_mask].copy()
    
    n_attack = len(attack_df)
    n_benign = len(benign_df)
    
    print(f"\nSample counts:")
    print(f"  {attack_label}: {n_attack}")
    print(f"  {benign_label}: {n_benign}")
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 确定目标数量
    if strategy == 'downsample':
        target_count = min(n_attack, n_benign)
        print(f"\nStrategy: Downsample to smaller count")
        print(f"Target count: {target_count}")
        
        # 下采样两类样本
        if n_attack > target_count:
            attack_df = attack_df.sample(n=target_count, random_state=seed).reset_index(drop=True)
            print(f"  Downsampled {attack_label} from {n_attack} to {target_count}")
        else:
            print(f"  Kept {attack_label} at {n_attack}")
        
        if n_benign > target_count:
            benign_df = benign_df.sample(n=target_count, random_state=seed).reset_index(drop=True)
            print(f"  Downsampled {benign_label} from {n_benign} to {target_count}")
        else:
            print(f"  Kept {benign_label} at {n_benign}")
    
    elif strategy == 'upsample':
        target_count = max(n_attack, n_benign)
        print(f"\nStrategy: Upsample to larger count")
        print(f"Target count: {target_count}")
        
        # 上采样少数类样本
        if n_attack < target_count:
            n_repeat = target_count // n_attack
            n_remainder = target_count % n_attack
            attack_df_upsampled = pd.concat([attack_df] * n_repeat, ignore_index=True)
            if n_remainder > 0:
                attack_df_remainder = attack_df.sample(n=n_remainder, random_state=seed).reset_index(drop=True)
                attack_df = pd.concat([attack_df_upsampled, attack_df_remainder], ignore_index=True)
            else:
                attack_df = attack_df_upsampled
            print(f"  Upsampled {attack_label} from {n_attack} to {len(attack_df)}")
        else:
            print(f"  Kept {attack_label} at {n_attack}")
        
        if n_benign < target_count:
            n_repeat = target_count // n_benign
            n_remainder = target_count % n_benign
            benign_df_upsampled = pd.concat([benign_df] * n_repeat, ignore_index=True)
            if n_remainder > 0:
                benign_df_remainder = benign_df.sample(n=n_remainder, random_state=seed).reset_index(drop=True)
                benign_df = pd.concat([benign_df_upsampled, benign_df_remainder], ignore_index=True)
            else:
                benign_df = benign_df_upsampled
            print(f"  Upsampled {benign_label} from {n_benign} to {len(benign_df)}")
        else:
            print(f"  Kept {benign_label} at {n_benign}")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be 'downsample' or 'upsample'")
    
    # 合并两类样本
    balanced_df = pd.concat([attack_df, benign_df], ignore_index=True)
    
    # 打乱顺序
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 统计平衡后的标签分布
    balanced_label_counts = balanced_df[label_column].value_counts()
    print(f"\nBalanced label distribution:")
    for label, count in balanced_label_counts.items():
        percentage = count / len(balanced_df) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    # 保存结果
    print(f"\nSaving balanced dataset to: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    balanced_df.to_csv(output_path, index=False)
    
    print(f"✅ Successfully balanced dataset and saved to: {output_path}")
    print(f"   Final shape: {balanced_df.shape[0]} rows, {balanced_df.shape[1]} columns")
    print(f"   Reduction: {len(df)} -> {len(balanced_df)} rows ({len(balanced_df)/len(df)*100:.2f}% retained)")
    
    return balanced_df


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description="平衡数据集中的 Attack 和 Benign 标签样本数量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 下采样到较小数量（默认）
  python balance_dataset.py \\
      --input converted_flows_nfv1_0130_labeled.csv \\
      --output converted_flows_nfv1_0130_balanced.csv
  
  # 上采样到较大数量
  python balance_dataset.py \\
      --input converted_flows_nfv1_0130_labeled.csv \\
      --output converted_flows_nfv1_0130_balanced.csv \\
      --strategy upsample
  
  # 自定义标签列和标签值
  python balance_dataset.py \\
      --input data.csv \\
      --output balanced_data.csv \\
      --label-column Label \\
      --attack-label malicious \\
      --benign-label normal \\
      --seed 123
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
        required=True,
        help="输出 CSV 文件路径"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default='Attack',
        help="标签列名（默认 'Attack'）"
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
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default='downsample',
        choices=['downsample', 'upsample'],
        help="平衡策略：'downsample'（下采样到较小数量）或 'upsample'（上采样到较大数量，默认 'downsample'）"
    )
    
    args = parser.parse_args()
    
    try:
        balance_dataset(
            input_path=args.input,
            output_path=args.output,
            label_column=args.label_column,
            attack_label=args.attack_label,
            benign_label=args.benign_label,
            seed=args.seed,
            strategy=args.strategy
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

