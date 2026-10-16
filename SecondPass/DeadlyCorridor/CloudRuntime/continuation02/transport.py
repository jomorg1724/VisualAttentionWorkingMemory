"""Run-specific SSH operations; private credentials stay on their original hosts."""
import argparse,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'sshdeps'))
import paramiko

def connect():
    d=json.loads((BASE/'deployment.json').read_text())['ssh']
    c=paramiko.SSHClient();c.load_host_keys(str(BASE/'known_hosts'))
    c.connect(d['host'],port=d['port'],username=d['user'],key_filename=str(Path.home()/'.ssh/id_ed25519'),timeout=20)
    return c

def main():
    p=argparse.ArgumentParser();p.add_argument('--script',type=Path);p.add_argument('--python',default='python3')
    p.add_argument('--upload',type=Path);p.add_argument('--remote');a=p.parse_args()
    c=connect()
    try:
        if a.upload:
            f=c.open_sftp();last=[0]
            def progress(n,total):
                if n-last[0]>=50_000_000 or n==total:print(f'Uploaded {n/1e6:.1f}/{total/1e6:.1f} MB',flush=True);last[0]=n
            f.put(str(a.upload),a.remote,callback=progress);f.close()
        if a.script:
            i,o,e=c.exec_command('cd /workspace/vawm && '+a.python+' -');i.write(a.script.read_text(encoding='utf8'));i.channel.shutdown_write()
            print(o.read().decode(),flush=True);err=e.read().decode()
            if err:print(err,flush=True)
            raise SystemExit(o.channel.recv_exit_status())
    finally:c.close()

if __name__=='__main__':main()
