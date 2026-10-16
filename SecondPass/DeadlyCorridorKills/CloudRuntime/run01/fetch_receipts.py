import json
from pathlib import Path
from transport import connect,BASE

client=connect();sftp=client.open_sftp()
remote='/workspace/vawm/SecondPass/DeadlyCorridorKills/CloudRuntime/run01/'
for name in ['preparation_verified.json','independent_production_verified.json','worker.json','guard.json','training/config.json','training/status.json']:
    dest=BASE/name;dest.parent.mkdir(parents=True,exist_ok=True)
    sftp.get(remote+name,str(dest))
sftp.close();client.close()
print((BASE/'independent_production_verified.json').read_text())
