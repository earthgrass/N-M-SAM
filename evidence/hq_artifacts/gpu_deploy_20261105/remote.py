import argparse, pathlib, paramiko, sys

def connect():
    client=paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.nmb2.seetacloud.com',port=25375,username='root',key_filename=str(pathlib.Path.home()/'.ssh/codex_apbr_seed245_nopass_20261103'),timeout=25)
    if client.get_transport().get_remote_server_key().get_fingerprint().hex()!='34afaa28d40dc70c5179bbe25ee2d6c5':
        client.close();raise RuntimeError('SSH host key changed')
    return client

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cmd');p.add_argument('--put',nargs=2);p.add_argument('--get',nargs=2);a=p.parse_args()
    c=connect()
    try:
        if a.cmd:
            _,o,e=c.exec_command(a.cmd,timeout=120)
            sys.stdout.buffer.write(o.read());sys.stderr.buffer.write(e.read());sys.exit(o.channel.recv_exit_status())
        else:
            s=c.open_sftp()
            if a.put:s.put(*a.put)
            if a.get:s.get(*a.get)
    finally:c.close()
