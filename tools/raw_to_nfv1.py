"""
将从 nfstream 导出的原始 CSV 转换为 NF-UNSW-NB15 风格格式，并添加 Attack 标签列。
"""
import argparse
import os
import pandas as pd


def convert_raw_to_nfv1(input_path: str, output_path: str, attack_value: str):
    # 1. 读取 CSV 文件
    df = pd.read_csv(input_path)

    # 2. 查看列名
    print("原始列名:")
    print(df.columns.tolist())

    # 3. 重命名列（将原始 NetFlow 格式转换为 NF-UNSW-NB15 格式）
    df = df.rename(columns={
        'src_ip': 'IPV4_SRC_ADDR',
        'dst_ip': 'IPV4_DST_ADDR',
        'src_port': 'L4_SRC_PORT',
        'dst_port': 'L4_DST_PORT',
        'protocol': 'PROTOCOL',
        'application_name': 'L7_PROTO',
        'udps.tcp_flags': 'TCP_FLAGS',
        'dst2src_packets': 'IN_PKTS',
        'dst2src_bytes': 'IN_BYTES',
        'src2dst_packets': 'OUT_PKTS',
        'src2dst_bytes': 'OUT_BYTES',
        'bidirectional_duration_ms': 'FLOW_DURATION_MILLISECONDS'
    })

    # 4. 删除不需要的列（仅删除实际存在的列，避免 KeyError）
    columns_to_drop = [
        'id', 'expiration_id', 'src_mac', 'src_oui',
        'dst_mac', 'dst_oui', 'ip_version', 'vlan_id',
        'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
        'tunnel_id', 'application_category_name', 'application_is_guessed',
        'application_confidence', 'requested_server_name', 'client_fingerprint',
        'server_fingerprint', 'user_agent', 'content_type',
        'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms',
        'bidirectional_packets', 'bidirectional_bytes',
        'dst2src_last_seen_ms', 'dst2src_first_seen_ms',
        'dst2src_duration_ms', 'src2dst_duration_ms'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)

    # 5. 添加 Attack 标签列
    df['Attack'] = attack_value

    # 6. 查看结果
    print("\n修改后的列名:")
    print(df.columns.tolist())
    print("\n数据形状:", df.shape)
    print("\nAttack 列取值:", attack_value)
    print("\n前几行数据:")
    print(df.head())

    # 7. 保存到新文件
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="将 nfstream 原始 CSV 转换为 NFV1 风格并添加 Attack 列"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入 CSV 路径，例如 ./flows.0.csv"
    )
    parser.add_argument(
        "--attack",
        required=True,
        help="Attack 列的值，例如 attack 或 Benign"
    )
    parser.add_argument(
        "--output",
        default="./converted_flows_nfv1.csv",
        help="输出 CSV 路径，默认 ./converted_flows_nfv1.csv"
    )
    args = parser.parse_args()

    convert_raw_to_nfv1(
        input_path=args.input,
        output_path=args.output,
        attack_value=args.attack
    )


if __name__ == '__main__':
    main()