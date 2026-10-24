"""One detached controller: mandatory gates, one training run, then stop."""
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from state import atomic_json

ROOT=Path(__file__).resolve().parent


def main():
    import fcntl
    artifacts=ROOT/'artifacts'; artifacts.mkdir(exist_ok=True)
    status=artifacts/'controller_status.json'
    with open('/tmp/dense_sam_anchor_seed0_controller.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            atomic_json(status,{'status':'prechecking','controller_pid':os.getpid()})
            with (artifacts/'precheck.log').open('w') as log:
                subprocess.run([sys.executable,'-u','precheck.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            out=ROOT/'artifacts/dense_sam_anchor_20261024/seed0'
            out.mkdir(parents=True,exist_ok=True)
            atomic_json(status,{'status':'formal_training','controller_pid':os.getpid()})
            with (out/'stdout.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u','train.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(artifacts/'processes.json',{'controller_pid':os.getpid(),'training_pid':child.pid})
                code=child.wait()
                if code:
                    raise RuntimeError(f'Formal training exited {code}; no automatic restart')
            atomic_json(status,{'status':'completed','controller_pid':os.getpid(),'follow_on_gpu_tasks':False})
        except BaseException as exc:
            atomic_json(artifacts/'controller_failure.json',{'type':type(exc).__name__,'message':str(exc),
                         'traceback':traceback.format_exc(),'restart':False})
            atomic_json(status,{'status':'failed','error':str(exc),'controller_pid':os.getpid()})
            raise


if __name__=='__main__':
    main()
