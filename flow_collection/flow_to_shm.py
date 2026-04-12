import argparse
import mmap
import os
import struct
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
from nfstream import NFPlugin, NFStreamer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
from preprocess_dataframe import DataFramePreprocessor

AI_IDS_SHM_MAGIC = 0x314D4853  # "SHM1" little-endian
AI_IDS_SHM_VERSION = 1
AI_IDS_SHM_MAX_INPUTS = 16
HEADER_FMT = '<IIQII' + ('I' * AI_IDS_SHM_MAX_INPUTS)
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class FlowSlicer(NFPlugin):
    def on_init(self, packet, flow):
        flow.udps.tcp_flags = 0

    def on_update(self, packet, flow):
        if packet.syn:
            flow.udps.tcp_flags |= 0x02
        if packet.ack:
            flow.udps.tcp_flags |= 0x10
        if packet.fin:
            flow.udps.tcp_flags |= 0x01
        if packet.rst:
            flow.udps.tcp_flags |= 0x04
        if packet.psh:
            flow.udps.tcp_flags |= 0x08
        if packet.urg:
            flow.udps.tcp_flags |= 0x20


class SharedMemoryFeatureWriter:
    def __init__(self, shm_path: str, shm_size: int):
        self.shm_path = shm_path
        self.shm_size = shm_size
        self.seq = 0

        if self.shm_size <= HEADER_SIZE:
            raise ValueError(f'shm_size must be > {HEADER_SIZE}')

        fd = os.open(self.shm_path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            os.ftruncate(fd, self.shm_size)
            self.mm = mmap.mmap(fd, self.shm_size, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
        finally:
            os.close(fd)

        self._write_header(0, 0, [])
        print(f'[PY] shared memory ready: {self.shm_path}, size={self.shm_size}')

    def _write_header(self, seq: int, total_floats: int, elem_counts):
        header = struct.pack(
            HEADER_FMT,
            AI_IDS_SHM_MAGIC,
            AI_IDS_SHM_VERSION,
            seq,
            len(elem_counts),
            total_floats,
            *(list(elem_counts) + [0] * (AI_IDS_SHM_MAX_INPUTS - len(elem_counts))),
        )
        self.mm.seek(0)
        self.mm.write(header)

    def publish(self, tensors):
        if len(tensors) > AI_IDS_SHM_MAX_INPUTS:
            raise ValueError(f'too many model inputs: {len(tensors)} > {AI_IDS_SHM_MAX_INPUTS}')

        elem_counts = [arr.size for arr in tensors]
        total_floats = sum(elem_counts)
        payload_bytes = total_floats * 4
        if HEADER_SIZE + payload_bytes > self.shm_size:
            raise ValueError(
                f'shared memory too small, need {HEADER_SIZE + payload_bytes}, got {self.shm_size}'
            )

        payload = np.concatenate([arr.reshape(-1) for arr in tensors], axis=0).astype(np.float32, copy=False)
        self.mm.seek(HEADER_SIZE)
        self.mm.write(payload.tobytes(order='C'))

        self.seq += 1
        self._write_header(self.seq, total_floats, elem_counts)
        self.mm.flush()

    def close(self):
        self.mm.close()


class RealTimeFlowToShm:
    def __init__(self, config_path: str, shm_path: str, shm_size: int):
        print('[PY] loading preprocessor...')
        self.preprocessor = DataFramePreprocessor.load_from_json(config_path)
        self.window_size = self.preprocessor.window_size
        self.flow_window = deque(maxlen=self.window_size)
        self.writer = SharedMemoryFeatureWriter(shm_path, shm_size)

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
            'input_TCP_FLAGS',
        ]
        self.model_expected_dims = {
            'L4_SRC_PORT': 32,
            'L4_DST_PORT': 32,
            'PROTOCOL': 3,
            'L7_PROTO': 19,
            'TCP_FLAGS': 9,
        }

    @staticmethod
    def flow_to_dataframe(flow):
        l7_proto = flow.application_name if flow.application_name else 'Unknown'
        return pd.DataFrame([
            {
                'IN_PKTS': flow.dst2src_packets,
                'OUT_PKTS': flow.src2dst_packets,
                'IN_BYTES': flow.dst2src_bytes,
                'OUT_BYTES': flow.src2dst_bytes,
                'FLOW_DURATION_MILLISECONDS': flow.bidirectional_duration_ms,
                'L4_SRC_PORT': flow.src_port,
                'L4_DST_PORT': flow.dst_port,
                'PROTOCOL': flow.protocol,
                'L7_PROTO': l7_proto,
                'TCP_FLAGS': flow.udps.tcp_flags,
            }
        ])

    def process_flow(self, flow):
        self.flow_window.append(self.flow_to_dataframe(flow))
        if len(self.flow_window) < self.window_size:
            return False

        window_df = pd.concat(list(self.flow_window), ignore_index=True)
        preprocessed_df = self.preprocessor.preprocess_dataframe(
            window_df,
            drop_original=False,
            keep_other_columns=False,
        )
        model_inputs = self.preprocessor.dataframe_to_model_inputs(
            preprocessed_df,
            window_size=self.window_size,
            input_name_prefix='input_',
            batch_size=1,
            model_expected_dims=self.model_expected_dims,
        )

        tensor_list = []
        for name in self.input_names:
            if name not in model_inputs:
                raise RuntimeError(f'missing model input: {name}')
            tensor_list.append(np.asarray(model_inputs[name], dtype=np.float32).reshape(-1))

        self.writer.publish(tensor_list)
        return True

    def run(self, interface_name: str):
        streamer = NFStreamer(
            source=interface_name,
            statistical_analysis=False,
            idle_timeout=1,
            udps=FlowSlicer(),
        )

        print(f'[PY] start collecting flows from {interface_name}')
        print('[PY] press Ctrl+C to stop')
        cnt = 0
        sent = 0
        try:
            for flow in streamer:
                cnt += 1
                if self.process_flow(flow):
                    sent += 1
                    print(f'[PY] flow={cnt}, published window_seq={self.writer.seq}, sent={sent}')
        except KeyboardInterrupt:
            print('\n[PY] interrupted by user')
        finally:
            self.writer.close()
            print(f'[PY] done, total_flows={cnt}, total_windows={sent}')


def parse_args():
    parser = argparse.ArgumentParser(description='Extract flow features and publish tensors to shared memory.')
    parser.add_argument('--interface', required=True, help='Network interface, e.g. eth0')
    parser.add_argument('--config', default='config/config.json', help='Preprocessor config path')
    parser.add_argument('--shm-path', default='/dev/shm/ai_ids_feature_shm', help='Shared memory file path')
    parser.add_argument('--shm-size', type=int, default=2 * 1024 * 1024, help='Shared memory size in bytes')
    return parser.parse_args()


def main():
    args = parse_args()
    app = RealTimeFlowToShm(args.config, args.shm_path, args.shm_size)
    app.run(args.interface)


if __name__ == '__main__':
    main()
