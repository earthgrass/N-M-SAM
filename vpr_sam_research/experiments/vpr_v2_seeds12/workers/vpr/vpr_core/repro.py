import hashlib,json,torch

def finite(tensor,label):
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"Non-finite {label}")

def seed_for(seed,*parts):
    value=json.dumps([seed,*parts],ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8],"little")%(2**63-1)
