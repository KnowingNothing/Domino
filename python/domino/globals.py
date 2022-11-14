from typing import Dict, Any, List, Set, Optional, Union, Tuple
import time 

class Timer:
    def __init__(self):
        self.starts: Dict[str, float] = {}
        self.times: Dict[str, float] = {}
    def start(self, name):
        if name not in self.starts:
            self.starts[name] = time.time() 
    def stop(self, name):
        if name not in self.starts:
            p = 0
        else: 
            p = time.time() - self.starts[name]
            self.starts.pop(name)
        self.times[name] = self.times.get(name, 0) + p 
    def show(self, title='timing'):
        print(title, ':')
        for t in self.times.items():
            print ('\t', t[0], ":", t[1])
    def clear(self):
        self.starts = {}
        self.times = {}

global_timer = Timer()