"""Read-only CPU replay of every retained example against current native renderers."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
import json,hashlib,datetime
from pathlib import Path
import numpy as np
from .export_assets import DemoStream,DEMO,source_hashes
from .validate_assets import validate_episode,validate_inventory,validate_coverage

def main():
    manifest=json.loads((DEMO/'artifacts/manifest.json').read_text())
    validate_inventory(manifest);validate_coverage(manifest['episodes'])
    before=source_hashes();rows=[]
    for cell in manifest['cells']:
        selected=[e for e in manifest['episodes'] if e['id'] in cell['episode_ids']]
        stream=DemoStream(cell['task_id'],cell['condition_id'])
        wanted={e['native_ordinal']:e for e in selected}
        for ordinal in range(max(wanted)+1):
            args=(1,stream.task) if stream.task in ('motion_direction','orientation','contrast','spatial_frequency','chromatic_increment','contour','natural_spectrum') else (1,stream.task,stream.kwargs)
            x,y,meta=stream.native.batch(*args)
            if ordinal not in wanted:continue
            e=wanted[ordinal];raw=np.load(DEMO/e['raw_path'],allow_pickle=False);current=x[0].numpy()
            try: assets=validate_episode(e)
            except ValueError as exc: assets={'error':str(exc)}
            from PIL import Image
            png=np.stack([np.asarray(Image.open(DEMO/'dist'/f).convert('RGB')).transpose(2,0,1) for f in e['frames']])
            assert np.array_equal(png,np.rint(np.clip(raw,0,1)*255).astype(np.uint8)), e['id']+' PNG mismatch'
            changed=[k for k in set(meta[0])|set(e['metadata']) if meta[0].get(k)!=e['metadata'].get(k)]
            a=np.rint(np.clip(raw,0,1)*255).astype(np.uint8);b=np.rint(np.clip(current,0,1)*255).astype(np.uint8)
            rows.append(dict(id=e['id'],frames=len(raw),float_exact=bool(np.array_equal(raw,current)),max_float_difference=float(np.max(np.abs(raw-current))),display_exact=bool(np.array_equal(a,b)),display_differing_values=int(np.count_nonzero(a!=b)),label_exact=int(y[0])==e['label'],metadata_changed_keys=changed,asset_validation=assets))
        print(cell['task_id'],cell['condition_id'],'checked',len(selected),flush=True)
    assert before==source_hashes(),'Native source modified during audit'
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mode='fresh direct renderer replay; CPU2; no model operations',source_identity=before,original_dataset_manifest_sha256=manifest['provenance']['dataset_manifest_sha256'],episodes=rows,
             summary=dict(episodes=len(rows),frames=sum(r['frames'] for r in rows),float_exact=sum(r['float_exact'] for r in rows),display_exact=sum(r['display_exact'] for r in rows),labels_exact=sum(r['label_exact'] for r in rows)))
    (DEMO/'verification/current_native_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(out['summary'])
if __name__=='__main__':main()
