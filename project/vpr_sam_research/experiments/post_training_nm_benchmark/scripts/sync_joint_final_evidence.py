"""Copy final Joint experiment evidence without modifying remote files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path

from phase4_remote_ops import connect


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", default="root")
    parser.add_argument("--known-hosts", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    client = connect(args)
    try:
        command = (
            "cd '" + args.remote_root + "' && "
            "find phase5_joint_only_20261101 phase6_sam_gpt_20261101 -type f "
            "! -name 'hessian_cache_epoch_*.pt' ! -name 'last.pt' "
            "! -name '*.tmp' -print0 | sort -z | xargs -0 sha256sum"
        )
        _stdin, stdout, stderr = client.exec_command(command)
        lines = stdout.read().decode().splitlines()
        error = stderr.read().decode()
        if stdout.channel.recv_exit_status() or error:
            raise RuntimeError(f"Remote manifest failed: {error}")
        entries = []
        for line in lines:
            checksum, relative = line.split("  ", 1)
            entries.append((relative, checksum))
        if args.small_only:
            entries = [(relative, checksum) for relative, checksum in entries if not relative.endswith(".pt")]
        sftp = client.open_sftp()
        try:
            for relative, checksum in entries:
                target = args.local_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and digest(target) == checksum:
                    print("VERIFIED", relative, flush=True)
                    continue
                remote = args.remote_root.rstrip("/") + "/" + relative
                partial = target.with_name(target.name + ".partial")
                remote_size = sftp.stat(remote).st_size
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > remote_size:
                    raise RuntimeError(f"Partial file larger than remote: {relative}")
                if remote_size > 10 * 1024 * 1024:
                    command = f"tail -c +{offset + 1} -- {shlex.quote(remote)}"
                    _stdin, stdout, stderr = client.exec_command(command)
                    with partial.open("ab") as sink:
                        while True:
                            block = stdout.channel.recv(1024 * 1024)
                            if not block:
                                break
                            sink.write(block)
                    error = stderr.read().decode()
                    if stdout.channel.recv_exit_status() or error:
                        raise RuntimeError(f"SSH stream failed: {relative}: {error}")
                else:
                    with sftp.open(remote, "rb") as source, partial.open("ab") as sink:
                        source.seek(offset)
                        while offset < remote_size:
                            block = source.read(min(64 * 1024, remote_size - offset))
                            if not block:
                                raise RuntimeError(f"Unexpected EOF: {relative}")
                            sink.write(block)
                            offset += len(block)
                if digest(partial) != checksum:
                    raise RuntimeError(f"SHA256 mismatch: {relative}")
                if target.exists():
                    raise FileExistsError(f"Existing target differs: {target}")
                os.replace(partial, target)
                print("COPIED", relative, remote_size, flush=True)
        finally:
            sftp.close()
        manifest = {relative: checksum for relative, checksum in entries}
        out = args.local_root / ("SMALL_SHA256_MANIFEST.json" if args.small_only else "SHA256_MANIFEST.json")
        if out.exists() and json.loads(out.read_text(encoding="utf-8")) != manifest:
            raise FileExistsError("Existing manifest differs")
        out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("COMPLETE", len(entries), "files", flush=True)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
