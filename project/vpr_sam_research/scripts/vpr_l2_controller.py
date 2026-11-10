"""Durable one-run queue with strict failure evidence. No automatic restart."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from vpr_sam.l2_protocol import OUTPUT, CONFIG, validate_protocol
from vpr_sam.repro import atomic_json, source_hashes, file_hash

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/vpr_l2'
CONTROL=OUT/'control'
child=None


def stop_child():
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


def main():
    global child
    CONTROL.mkdir(parents=True,exist_ok=True)
    lock=(CONTROL/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=CONTROL/'status.json'
    if status.exists():raise RuntimeError('Existing campaign state; refusing duplicate/restart')
    validate_protocol(ROOT)
    gates=json.loads((CONTROL/'gates.json').read_text())
    def frozen():
        if source_hashes(ROOT)!=gates['source_hashes'] or file_hash(ROOT/CONFIG)!=gates['config_sha256']:
            raise RuntimeError('Source/config provenance changed')
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=gates['commit']:
            raise RuntimeError('Frozen Git HEAD changed')
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
            raise RuntimeError('Frozen worktree became dirty')
    frozen()
    shutil.copyfile(ROOT/'docs/VPR_L2_PROTOCOL_FREEZE.md',ROOT/'VPR_L2_PROTOCOL_FREEZE.md')
    for name in ('TRAINING_REPORT','GEOMETRY_ANALYSIS','PRUNING_EVALUATION','SEED0_COMPARISON','FINAL_AUDIT'):
        (ROOT/f'VPR_L2_{name}.md').write_text('# VPR-L2 '+name+'\n\nPENDING: experiment or evaluation not yet complete.\n',encoding='utf8')
    env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['PYTHONUNBUFFERED']='1'
    for stage,script in [('training','vpr_l2_train.py'),('evaluation','vpr_l2_evaluate.py'),('reporting','vpr_l2_report.py')]:
        frozen()
        with (OUT/f'{stage}.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(status,dict(status='RUNNING',stage=stage,pid=child.pid,controller_pid=os.getpid(),time=time.time(),commit=gates['commit']))
            while child.poll() is None:
                free=shutil.disk_usage(ROOT).free
                if free<4*1024**3:raise RuntimeError('Disk reserve fell below 4 GiB')
                gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
                if gpu.returncode:raise RuntimeError('GPU monitoring/hardware query failed: '+gpu.stderr)
                telemetry=dict(time=time.time(),stage=stage,pid=child.pid,free_bytes=free,gpu=gpu.stdout.strip())
                run=ROOT/OUTPUT
                if (run/'failure.json').exists():raise RuntimeError('Training failure evidence published')
                if (run/'epochs.json').exists():
                    ep=json.loads((run/'epochs.json').read_text())
                    if ep:telemetry.update(completed_epoch=ep[-1]['epoch'],step=ep[-1]['step'],validation=ep[-1]['validation'],geometry=ep[-1]['geometry'])
                atomic_json(CONTROL/'latest_telemetry.json',telemetry)
                # Server-side wait only: durable process survives SSH disconnection.
                time.sleep(30)
            if child.returncode:raise RuntimeError(f'{stage} failed with exit {child.returncode}; see {stage}.log')
            child=None
    frozen()
    atomic_json(status,dict(status='COMPLETE',time=time.time(),commit=gates['commit'],
        authorized_trajectory_count=1,subsequent_gpu_tasks=0,report='VPR_L2_FINAL_AUDIT.md'))
    print('COMPLETE: approved single experiment and evaluations; no further GPU task',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        stop_child()
        CONTROL.mkdir(parents=True,exist_ok=True)
        evidence=dict(status='FAILED',time=time.time(),error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
            affected_pid=child.pid if child else None,automatic_restart=False)
        atomic_json(CONTROL/'failure.json',evidence)
        # Duplicate launch failures must not overwrite a live controller's status.
        if child is not None:atomic_json(CONTROL/'status.json',evidence)
        raise
