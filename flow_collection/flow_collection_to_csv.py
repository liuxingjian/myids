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
        

if __name__ == '__main__':
    input_filepaths = ["eth0"]
    for path in sys.argv[1:]:
        input_filepaths.append(path)
    if len(input_filepaths) == 1:  # Single file / Interface
        input_filepaths = input_filepaths[0]

    flow_streamer = NFStreamer(
        source=input_filepaths, statistical_analysis=False, idle_timeout=1,udps=FlowSlicer()
    )
    total_flows_count = flow_streamer.to_csv(path='./flows.csv', columns_to_anonymize=[], flows_per_file=1000000, rotate_files=0)
