"""Bounded lightweight telemetry for a future smoke; no external server access."""
import json,shutil,threading,time
from pathlib import Path
import torch

class Monitor:
    def __init__(self,output,device,min_disk,interval=2):
        self.output=Path(output);self.device=device;self.min_disk=min_disk;self.interval=interval
        self.error=None;self.event=threading.Event();self.thread=threading.Thread(target=self.loop,daemon=True)
    def loop(self):
        try:
            with self.output.open('x',encoding='utf-8') as f:
                while not self.event.is_set():
                    free=shutil.disk_usage(self.output.parent).free
                    if free<self.min_disk:raise OSError('Smoke disk reserve violated')
                    row={'unix_time':time.time(),'disk_free_bytes':free,'cuda_allocated':torch.cuda.memory_allocated(self.device),
                         'cuda_reserved':torch.cuda.memory_reserved(self.device),'cuda_peak_allocated':torch.cuda.max_memory_allocated(self.device)}
                    f.write(json.dumps(row)+'\n');f.flush();self.event.wait(self.interval)
        except Exception as exc:self.error=repr(exc)
    def check(self):
        if self.error is not None:raise RuntimeError('Resource monitor failure: '+self.error)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.event.set();self.thread.join(timeout=5);self.check()
