"""Minimal Paramiko transport for the user-authorized Phase-4 host.

Credentials are read from environment variables and are never written to disk.
The first connection records the public host key; later calls require it.
"""
from __future__ import annotations
import argparse, base64, hashlib, os
from pathlib import Path
import paramiko

def connect(args):
    password=os.environ.get('APBR_SSH_PASSWORD')
    if not password: raise RuntimeError('APBR_SSH_PASSWORD is required')
    keyfile=Path(args.known_hosts)
    client=paramiko.SSHClient()
    if keyfile.exists():
        client.load_host_keys(str(keyfile));client.set_missing_host_key_policy(paramiko.RejectPolicy())
    else:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host,port=args.port,username=args.user,password=password,
                   look_for_keys=False,allow_agent=False,timeout=20,banner_timeout=20,auth_timeout=20)
    key=client.get_transport().get_remote_server_key()
    fingerprint=hashlib.sha256(key.asbytes()).digest()
    print('HOST_KEY_SHA256='+base64.b64encode(fingerprint).decode().rstrip('='))
    if not keyfile.exists():
        keyfile.parent.mkdir(parents=True,exist_ok=True);client.save_host_keys(str(keyfile))
    return client

def main():
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--user',default='root');p.add_argument('--known-hosts',required=True)
    sub=p.add_subparsers(dest='action',required=True)
    e=sub.add_parser('exec');e.add_argument('--command',required=True)
    u=sub.add_parser('upload');u.add_argument('--local',type=Path,required=True);u.add_argument('--remote',required=True)
    d=sub.add_parser('download');d.add_argument('--remote',required=True);d.add_argument('--local',type=Path,required=True)
    a=p.parse_args();client=connect(a)
    try:
        if a.action=='exec':
            stdin,stdout,stderr=client.exec_command(a.command,get_pty=False,timeout=None)
            for line in iter(stdout.readline,''): print(line,end='')
            error=stderr.read().decode(errors='replace')
            if error: print(error,end='')
            code=stdout.channel.recv_exit_status();print(f'REMOTE_EXIT_CODE={code}');return code
        sftp=client.open_sftp()
        try:
            if a.action=='upload':
                sftp.put(str(a.local.resolve()),a.remote,confirm=True);print('UPLOAD_COMPLETE='+a.remote)
            else:
                a.local.parent.mkdir(parents=True,exist_ok=True)
                sftp.get(a.remote,str(a.local.resolve()));print('DOWNLOAD_COMPLETE='+str(a.local.resolve()))
        finally:sftp.close()
        return 0
    finally:client.close()
if __name__=='__main__':raise SystemExit(main())
