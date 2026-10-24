"""Detached one-run controller; stops on any gate or training failure."""
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import yaml
from state import atomic_json

ROOT=Path(__file__).resolve().parent


def main():
    import fcntl
    c=yaml.safe_load((ROOT/'config.yaml').read_text())
    artifacts=ROOT/'artifacts'; artifacts.mkdir(exist_ok=True)
    status=artifacts/'controller_status.json'
    with open('/tmp/dense_sam_ssam50k_seeds12_controller.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            atomic_json(status,{'status':'prechecking','pid':os.getpid(),'unix_time':time.time()})
            with (artifacts/'precheck.log').open('x') as log:
                subprocess.run([sys.executable,'-u','precheck.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            out=ROOT/c['output']; out.mkdir(parents=True,exist_ok=True)
            with (out/'stdout.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u','train.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(artifacts/'processes.json',{'controller_pid':os.getpid(),'training_pid':child.pid,
                                                        'start_unix_time':time.time()})
                atomic_json(status,{'status':'formal_training','pid':os.getpid(),'training_pid':child.pid})
                code=child.wait()
                if code: raise RuntimeError(f'Training failed with exit{code}; no automatic restart')
            atomic_json(status,{'status':'completed','pid':os.getpid(),'unix_time':time.time(),
                                'additional_gpu_tasks':False})
        except BaseException as exc:
            atomic_json(artifacts/'controller_failure.json',{'type':type(exc).__name__,'message':str(exc),
                        'traceback':traceback.format_exc(),'automatic_restart':False})
            atomic_json(status,{'status':'failed','error':str(exc)})
            raise


if __name__=='__main__': main()
