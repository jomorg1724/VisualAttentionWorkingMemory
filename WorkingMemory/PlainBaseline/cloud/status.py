"""Print the latest pulled state of the accumulator pod run. Usage: python -m WorkingMemory.PlainBaseline.cloud.status [run_dir]"""
import json,glob,time,os,sys
R=sys.argv[1] if len(sys.argv)>1 else json.load(open('WorkingMemory/PlainBaseline/runs/latest_cloud.json'))['run']
c=json.load(open(os.path.join(R,'cloud_provisioning.json')));print('pod',c['pod_id'],'status',c['status'],'deadline in %.2f h'%((c['deadline_unix']-time.time())/3600))
for f in sorted(glob.glob(os.path.join(R,'pulled','results','*','*','*','validation.json'))):
    h=json.load(open(f));r=h[-1];p=f.replace(os.sep,'/').split('/')
    print(f"{p[-4]}/{p[-3]}/{p[-2]:7s} update {r['update']:5d} BA {r['mean_ba']:.3f} "+' '.join(f"{k.split('_')[-1]}={v['balanced_accuracy']:.2f}" for k,v in r['cells'].items()))
for f in sorted(glob.glob(os.path.join(R,'pulled','results','*','program_receipt.json'))):
    j=json.load(open(f));print(os.path.basename(os.path.dirname(f)),'program status',j.get('status'),{k:v['status'] for k,v in j['stages'].items()})
