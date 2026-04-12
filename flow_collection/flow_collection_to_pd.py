import numpy as np
import time
from nfstream import NFStreamer,NFPlugin
import sys
import pandas as pd


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
    def on_expire(self, flow):
        to_send = np.array([flow.src_ip,flow.dst_ip,flow.src_port,flow.dst_port,flow.protocol,
        flow.src2dst_packets,flow.src2dst_bytes,flow.dst2src_bytes,
                                  flow.dst2src_packets,flow.bidirectional_duration_ms,flow.application_name,
                                  flow.udps.tcp_flags]).reshape((1,-1))
        #print(to_send)
        df = pd.DataFrame(to_send, columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'OUT_PKTS', 
        'OUT_BYTES', 'IN_BYTES', 'IN_PKTS', 'FLOW_DURATION_MILLISECONDS','L7_PROTO' ,'TCP_FLAGS'])
        print(df)
        

if __name__ == '__main__':
    input_filepaths = ["eth0"]
    for path in sys.argv[1:]:
        input_filepaths.append(path)
    if len(input_filepaths) == 1:  # Single file / Interface
        input_filepaths = input_filepaths[0]

    flow_streamer = NFStreamer(
        source=input_filepaths, statistical_analysis=False, idle_timeout=1,udps=FlowSlicer()
    )
    cnt = 0
    for flow in flow_streamer:
        cnt += 1
        print(cnt)