import os
import sys
import time
import threading
import pandas as pd
from nfstream import NFStreamer, NFPlugin

try:
    from tqdm import tqdm
except ImportError:
    print("[!] 未检测到 tqdm，正在为您静默安装以便显示进度条...")
    os.system(f"{sys.executable} -m pip install tqdm -q")
    from tqdm import tqdm

# 自定义 NFPlugin 用于解析 TCP 标志位
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

def process_pcap(pcap_file, label, progress_bar):
    file_size_mb = os.path.getsize(pcap_file) / (1024 * 1024)
    progress_bar.set_postfix_str(f"提取特征中... ({file_size_mb:.1f} MB)")
    
    # 1. 使用 NFStreamer 提取流量特征
    flow_streamer = NFStreamer(
        source=pcap_file, 
        statistical_analysis=False, 
        idle_timeout=1,
        udps=FlowSlicer()
    )
    
    # 提取为 Pandas DataFrame (耗时操作)
    df = flow_streamer.to_pandas(columns_to_anonymize=[])
    
    progress_bar.set_postfix_str(f"格式清洗中... ({len(df)} 条特征)")
    
    # 2. 列名重命名 (NF-UNSW-NB15格式)
    rename_mapping = {
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
    }
    df = df.rename(columns=rename_mapping)
    
    # 3. 删除不需要的列
    columns_to_drop = [
        'id', 'expiration_id', 'src_mac', 'src_oui',
        'dst_mac', 'dst_oui', 'ip_version', 'vlan_id',
        'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
        'tunnel_id', 'application_category_name', 'application_is_guessed',
        'application_confidence', 'requested_server_name', 'client_fingerprint',
        'server_fingerprint', 'user_agent', 'content_type', 'bidirectional_first_seen_ms',
        'bidirectional_last_seen_ms', 'bidirectional_packets', 'bidirectional_bytes',
        'dst2src_last_seen_ms', 'dst2src_first_seen_ms', 'dst2src_duration_ms', 'src2dst_duration_ms'
    ]
    # 只删除确实存在于df中的列
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    
    # 4. 打上 Attack 标签
    df['Attack'] = label
    
    return df

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义需要处理的文件及其对应的标签
    # e1, e2 为攻击流量 (attack)
    # n1, n2 为正常流量 (Benign)
    pcap_files = [
        # ('e1.pcap', 'attack'),
        ('e2.pcap', 'attack'),
        ('n1.pcap', 'Benign'),
        ('n2.pcap', 'Benign')
    ]
    
    all_dfs = []
    
    # 进度条包装
    with tqdm(total=len(pcap_files), desc="[*] 整体进度", unit="文件") as pbar:
        for filename, label in pcap_files:
            filepath = os.path.join(base_dir, filename)
            pbar.set_description(f"[*] 处理中: {filename}")
            if not os.path.exists(filepath):
                pbar.write(f"[!] 警告: 文件不存在 {filepath}，跳过")
                pbar.update(1)
                continue
                
            try:
                # 记录开始时间
                start_time = time.time()
                
                df = process_pcap(filepath, label, pbar)
                all_dfs.append(df)
                
                cost = time.time() - start_time
                pbar.write(f"[+] {filename} 完毕 | 标签: {label} | 流数量: {len(df)} | 耗时: {cost:.1f}秒")
            except Exception as e:
                pbar.write(f"[!] 处理 {filename} 时出错: {str(e)}")
            pbar.update(1)
    
    # 5. 合并数据集并保存
    if all_dfs:
        print("\n[*] 正在合并并保存总数据集... (请稍候)")
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        output_file = os.path.join(base_dir, 'final_dataset_labeled.csv')
        final_df.to_csv(output_file, index=False)
        print(f"[+] 成功构建总数据集，共 {len(final_df)} 条数据。")
        print(f"[+] 已保存到: {output_file}")
        
        print("\n标签分布:")
        print(final_df['Attack'].value_counts())
    else:
        print("[-] 没有处理任何数据。")

if __name__ == '__main__':
    main()
