import os
import time
import torch
import pdb

class Profiler(object):
    def __init__(self, silent=False):
        self.silent = silent
        torch.cuda.synchronize()
        self.start = time.time()
        self.cache_time = self.start

    def reset(self, silent=None):
        if silent is None:
            silent = self.silent
        self.__init__(silent=silent)

    def report_process(self, process_name):
        if self.silent:
            return None
        torch.cuda.synchronize()
        now = time.time()
        print('{0}\t: {1:.4f}'.format(process_name, now - self.cache_time))
        self.cache_time = now

    def report_all(self, whole_process_name):
        if self.silent:
            return None
        torch.cuda.synchronize()
        now = time.time()
        print('{0}\t: {1:.4f}'.format(whole_process_name, now - self.start))
        pdb.set_trace()

