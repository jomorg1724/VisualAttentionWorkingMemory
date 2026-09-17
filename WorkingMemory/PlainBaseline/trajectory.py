"""Print binned training trajectories (loss, accuracy, gradient norm) for every metrics.csv under a directory."""
import sys,glob,os
import pandas as pd
root=sys.argv[1];bins=[(1,100),(100,400),(400,800),(800,1200),(1200,1600),(1600,2400),(2400,3200),(3200,10**9)]
for f in sorted(glob.glob(os.path.join(root,'**','metrics.csv'),recursive=True)):
    m=pd.read_csv(f);u=m['update'];print(os.path.relpath(os.path.dirname(f),root))
    for a,b in bins:
        s=m[(u>a)&(u<=b)]
        if len(s):print(f'  {a:5d}-{min(b,int(u.max())):5d} loss {s.loss.mean():.4f} acc {s.accuracy.mean():.3f} gnorm mean {s.grad_norm.mean():.3f} max {s.grad_norm.max():.2f} s/upd {s.seconds.mean():.2f} wait {s.wait_seconds.mean():.2f}')
