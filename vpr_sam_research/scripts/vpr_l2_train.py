"""Only the single frozen seed-0 fixed_l2 trajectory, with passive diagnostics."""
import json
from pathlib import Path
from vpr_sam.l2_observer import ObservedStepper
from vpr_sam.l2_protocol import validate_protocol, CONFIG
from vpr_sam.repro import file_hash, source_hashes
from vpr_sam.training import engine

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    c, _ = validate_protocol(ROOT)
    gates = json.loads((ROOT/'artifacts/vpr_l2/control/gates.json').read_text())
    if gates['status'] != 'PASS' or gates['source_hashes'] != source_hashes(ROOT) or gates['config_sha256'] != file_hash(ROOT/CONFIG):
        raise RuntimeError('Frozen precheck/source identity mismatch')
    engine.Stepper = ObservedStepper
    # No resume/warm-start/recovery CLI or alternate config/method/seed.
    engine.train(c, ROOT, formal_allowed=True)
