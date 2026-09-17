"""Saved-feature bootstrap; no models/inference/optimizers."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import json,time
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent;start=time.time();data=np.load(p/'test_features.npz');pred=np.load(p/'heldout_predictions.npz')
t=data['counts'].reshape(-1,2,3);target=t[:,1]-t[:,0];groups=data['template_group'][::2];rng=np.random.default_rng(16951001+991)
members=[np.flatnonzero(groups==g) for g in np.unique(groups)];boot=np.array([np.concatenate([members[i] for i in rng.integers(len(members),size=len(members))]) for _ in range(1000)])
den=(target[boot]**2).sum((1,2));results={};samples={}
for name in pred.files:
 if not name.startswith('counts'):continue
 z=pred[name].reshape(-1,2,3);diff=z[:,1]-z[:,0];r2=1-((diff-target)[boot]**2).sum((1,2))/den;samples[name]=r2
 results[name]=dict(paired_difference_r2_ci95=np.quantile(r2,[.025,.975]).tolist())
for prefix,final in [('counts_ridge_prefix_r','counts_ridge_final_r'),('counts_ridge_prefix_ra','counts_ridge_final_ra'),('counts_mlp_prefix_ra','counts_mlp_final_ra')]:
 results[final]['final_minus_prefix_r2_ci95']=np.quantile(samples[final]-samples[prefix],[.025,.975]).tolist()
(p/'count_uncertainty.json').write_text(json.dumps(dict(results=results,bootstrap='1000 canonical-pair-template clusters, partners/all4rotations together',wall_seconds=time.time()-start),indent=2))
