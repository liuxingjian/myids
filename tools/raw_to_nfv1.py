"""
将原始的从nfstream中导出的flows.csv文件转换为NF-UNSW-NB15格式
"""
import pandas as pd

# 1. 读取CSV文件
df = pd.read_csv('./flows.0.csv')

# 2. 查看列名
print("原始列名:")
print(df.columns.tolist())

# 3. 重命名列（将原始NetFlow格式转换为NF-UNSW-NB15格式）
df = df.rename(columns={
    'src_ip': 'IPV4_SRC_ADDR',
    'dst_ip': 'IPV4_DST_ADDR',
    'src_port': 'L4_SRC_PORT',
    'dst_port': 'L4_DST_PORT',
    'protocol': 'PROTOCOL',
    'application_name': 'L7_PROTO',
    'udps.tcp_flags': 'TCP_FLAGS',
    'dst2src_packets': 'IN_PKTS',  # 假设这是入包数
    'dst2src_bytes': 'IN_BYTES',    # 假设这是入字节数
    'src2dst_packets': 'OUT_PKTS',  # 假设这是入包数
    'src2dst_bytes': 'OUT_BYTES',    # 假设这是入字节数
    'bidirectional_duration_ms': 'FLOW_DURATION_MILLISECONDS'
})

# 4. 删除不需要的列
columns_to_drop = [
    'id', 'expiration_id', 'src_mac', 'src_oui',
    'dst_mac', 'dst_oui', 'ip_version', 'vlan_id',
    'src2dst_first_seen_ms','src2dst_last_seen_ms',
    'tunnel_id', 'application_category_name', 'application_is_guessed',
    'application_confidence', 'requested_server_name', 'client_fingerprint',
    'server_fingerprint', 'user_agent', 'content_type','bidirectional_first_seen_ms','bidirectional_last_seen_ms',
    'bidirectional_packets','bidirectional_bytes','dst2src_last_seen_ms','dst2src_first_seen_ms','dst2src_duration_ms','src2dst_duration_ms'
]

df = df.drop(columns=columns_to_drop)

# 5. 或者只保留需要的列
# df = df[['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 
#          'TCP_FLAGS', 'IN_PKTS', 'IN_BYTES', 'FLOW_DURATION_MILLISECONDS']]

# 6. 查看结果
print("\n修改后的列名:")
print(df.columns.tolist())
print("\n数据形状:", df.shape)
print("\n前几行数据:")
print(df.head())

# 7. 保存到新文件
df.to_csv('./converted_flows_nfv1.csv', index=False)