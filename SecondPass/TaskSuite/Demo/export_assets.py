"""Deterministic, train-photo-only native atlas exporter; never imports models.

Run with Demo/.venv/bin/python -m SecondPass.TaskSuite.Demo.export_assets.
Numerical limits are established before importing any renderer dependency.
"""
from __future__ import annotations
import os
import sys
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
for _key in THREAD_KEYS:
    os.environ[_key] = '2'
import copy
import hashlib
import json
from pathlib import Path

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parents[2]
SOURCE_FILES = (
    'SecondPass/TaskSuite/catalog.json', 'SecondPass/TaskSuite/suite.py',
    'PreAttentiveVision/neuroscience_stimuli.py', 'PreAttentiveVision/natural_stimuli.py',
    'WorkingMemory/PlainBaseline/variants.py', 'WorkingMemory/SpatialTaskBattery/stimuli.py',
    'WorkingMemory/stimuli.py',
)
DATASET = ROOT / 'PreAttentiveVision/data/bsds500/manifest.json'
CATALOG = json.loads((ROOT / SOURCE_FILES[0]).read_text())
TASKS = {t['id']: t for t in CATALOG['tasks']}
NAMESPACE = 'visual-task-suite-atlas-demo-only-v1'


def source_hashes():
    return {'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_FILES},
            'dataset_manifest_sha256': hashlib.sha256(DATASET.read_bytes()).hexdigest()}


# The imported local closure has been traced in source before import. In
# particular WorkingMemory.stimuli imports only sensory renderer + libraries.
IMPORT_BASELINE = source_hashes()
import numpy as np
import torch
from PIL import Image
from PreAttentiveVision.neuroscience_stimuli import TaskStream
from WorkingMemory.PlainBaseline.variants import VariantStream
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(2)
except RuntimeError:
    if torch.get_num_interop_threads() > 2:
        raise


def import_trace():
    paths = set()
    for mod in list(sys.modules.values()):
        path = getattr(mod, '__file__', None)
        if path:
            try:
                rel = Path(path).resolve().relative_to(ROOT).as_posix()
            except ValueError:
                continue
            if not rel.startswith('SecondPass/TaskSuite/Demo/'):
                paths.add(rel)
    return sorted(paths)


class DemoStream:
    """One cell-local native stream, with a separate hash-derived demo seed."""
    def __init__(self, task, cell):
        if task not in TASKS:
            raise ValueError('Unknown task')
        found = [c for c in TASKS[task]['conditions'] if c['id'] == cell]
        if len(found) != 1:
            raise ValueError('Unknown condition')
        self.task, self.cell = task, cell
        self.kwargs = copy.deepcopy(found[0]['kwargs'])
        self.seed = int.from_bytes(hashlib.sha256(f'{NAMESPACE}/{task}/{cell}'.encode()).digest()[:8], 'big') % (2**53)
        cls = TaskStream if TASKS[task]['group'] == 'sensory' else VariantStream if task == 'orientation_ring' else SpatialBatteryStream
        self.native = cls(self.seed, 'train')
        self.ordinal = 0
        self.provenance = source_hashes()

    def next(self):
        args = (1, self.task) if TASKS[self.task]['group'] == 'sensory' else (1, self.task, self.kwargs)
        x, y, metadata = self.native.batch(*args)
        self.ordinal += 1
        return x[0].numpy().copy(), int(y[0]), copy.deepcopy(metadata[0])

    def state_dict(self):
        return copy.deepcopy(dict(namespace=NAMESPACE, task=self.task, cell=self.cell, seed=self.seed,
                                  kwargs=self.kwargs, ordinal=self.ordinal,
                                  provenance=self.provenance, native=self.native.state_dict()))

    def load_state_dict(self, state):
        for key, value in [('namespace', NAMESPACE), ('task', self.task), ('cell', self.cell),
                           ('seed', self.seed), ('kwargs', self.kwargs), ('provenance', source_hashes())]:
            if state.get(key) != value:
                raise ValueError(f'Demo stream {key} mismatch')
        # Validate into a fresh independent object before replacing progress.
        fresh = DemoStream(self.task, self.cell)
        fresh.native.load_state_dict(copy.deepcopy(state['native']))
        self.native, self.ordinal = fresh.native, state['ordinal']


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    partial.replace(path)


def export_episode(root, stream, raw, label, metadata, reason):
    """Export a native observation sequence, with no annotations or extra frames."""
    from PIL import ImageSequence
    import zipfile
    from .validate_assets import expected_phases, coverage_map
    root = Path(root).resolve()
    root.relative_to(DEMO)  # no writes outside the authorized subtree
    dist = root / 'dist'
    task, cell = stream.task, stream.cell
    native_ordinal = int(metadata['trial_id'].rsplit('/',1)[1])
    episode_id = f'{task}--{cell}--{native_ordinal:05d}'
    def location(relative):
        path = dist / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    def sha(data): return hashlib.sha256(data).hexdigest()
    raw = np.ascontiguousarray(raw,dtype='<f4')
    if raw.shape[1:] != (3,100,100) or not np.isfinite(raw).all() or raw.min()<0 or raw.max()>1:
        raise ValueError('Invalid native raster')
    raw_path = f'artifacts/raw/{episode_id}.npy'
    (root / raw_path).parent.mkdir(parents=True,exist_ok=True)
    np.save(root / raw_path, raw, allow_pickle=False)
    display = np.rint(np.clip(raw,0,1)*255).astype(np.uint8).transpose(0,2,3,1)
    images = [Image.fromarray(frame) for frame in display]
    frames, hashes = [], []
    for i,image in enumerate(images):
        relative = f'assets/frames/{episode_id}/{i:03d}.png'
        path = location(relative); image.save(path, format='PNG', optimize=False)
        frames.append(relative); hashes.append(sha(path.read_bytes()))
    # A palette built from all native display frames is reused without dithering.
    montage = Image.fromarray(display.reshape(-1,100,3))
    palette = montage.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quantized = [im.quantize(palette=palette,dither=Image.Dither.NONE) for im in images]
    gif = f'assets/gifs/{episode_id}.gif'
    quantized[0].save(location(gif), format='GIF', save_all=True, append_images=quantized[1:],
                      duration=[300]*len(images), loop=0, optimize=False, disposal=1)
    mapping, decoded = [], []
    with Image.open(location(gif)) as image:
        for i,frame in enumerate(ImageSequence.Iterator(image)):
            duration = frame.info['duration']
            if duration % 300: raise ValueError('GIF changed native timing')
            start = len(decoded)
            decoded.extend([np.asarray(frame.convert('RGB')).copy()] * (duration//300))
            mapping.append(dict(encoded_index=i,source_indices=list(range(start,len(decoded))),duration_ms=duration))
    if len(decoded)!=len(raw) or not all(np.array_equal(a,np.asarray(b.convert('RGB'))) for a,b in zip(decoded,quantized)):
        raise ValueError('GIF failed decoded native timeline equality')
    poster_index = metadata.get('sample_frames', metadata.get('moving_frames', metadata.get('study_frames', [0])))[0] if metadata.get('sample_frames') or metadata.get('moving_frames') or metadata.get('study_frames') else metadata.get('reference_frame',0)
    poster = f'assets/posters/{episode_id}.png'
    images[poster_index].save(location(poster),format='PNG',optimize=False)
    frame_zip = f'assets/frame-zips/{episode_id}.zip'
    with zipfile.ZipFile(location(frame_zip),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        contents = [(Path(p).name,location(p).read_bytes()) for p in frames]
        contents.append(('source-frame-map.json',json.dumps(dict(native_trial_id=metadata['trial_id'],
                         frame_ms=300, frames=[dict(index=i,file=Path(p).name) for i,p in enumerate(frames)]),sort_keys=True).encode()))
        for name,data in contents:
            info = zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o644 << 16
            archive.writestr(info,data,compresslevel=6)
    difference = np.abs(raw.astype(np.float64)-display.transpose(0,3,1,2).astype(np.float64)/255)
    gif_error = np.abs(np.asarray(decoded).astype(np.int16)-display.astype(np.int16))
    photo_ids = list(metadata.get('study_source_ids',[]))
    if metadata.get('probe_source_id'): photo_ids.append(metadata['probe_source_id'])
    if metadata.get('base_id'): photo_ids.append(metadata['base_id'])
    episode = dict(id=episode_id,task_id=task,condition_id=cell,kwargs=stream.kwargs,seed=stream.seed,
        native_trial_id=metadata['trial_id'], native_ordinal=native_ordinal, label=label,
        label_meaning=TASKS[task]['labels'][label],metadata=copy.deepcopy(metadata),frame_count=len(raw),
        frames=frames, frame_sha256=hashes, float_sha256=sha(raw.tobytes()),raw_path=raw_path,
        raw_file_sha256=sha((root/raw_path).read_bytes()),phases=expected_phases(task,metadata,len(raw)),
        gif=gif,poster=poster,poster_frame_index=poster_index,frame_zip=frame_zip,
        metadata_path=f'assets/metadata/{episode_id}.json',source_split='train',photo_ids=sorted(set(photo_ids)),
        timing=dict(frame_ms=300,total_ms=len(raw)*300,gif_frame_map=mapping,
                    native_frame_ms=10 if task=='krauzlis_cued_motion' else None,
                    interpretation='Illustration speed, not calibrated biological presentation; loop restarts trial'),
        gif_palette=palette.getpalette(),
        conversion_error=dict(max_abs=float(difference.max()),mean_abs=float(difference.mean()),units='native [0,1]'),
        gif_quantization_error=dict(max_abs=int(gif_error.max()),mean_abs=float(gif_error.mean()),units='uint8 channel levels',
                                    normalized_mean_abs=float(gif_error.mean()/255)),
        curation_reason=reason, covered_variants=[],
        derived_fields=['phases','timing','covered_variants','conversion_error','gif_quantization_error','poster_frame_index'],
        asset_sha256={p:sha(location(p).read_bytes()) for p in [gif,poster,frame_zip]})
    episode['covered_variants'] = [r['id'] for r in coverage_map([episode])['requirements'] if episode_id in r['episode_ids']]
    atomic_json(location(episode['metadata_path']),episode)
    return episode


def contact_sheets(manifest, root):
    """Annotated analysis sheets; original observer assets remain untouched."""
    from PIL import ImageDraw
    root = Path(root); out = root/'verification'; out.mkdir(parents=True,exist_ok=True)
    lookup = {e['id']:e for e in manifest['episodes']}
    sheets = []
    def grid(name, rows, columns=5):
        width, height = columns*220, ((len(rows)+columns-1)//columns)*145+40
        canvas = Image.new('RGB',(width,height),'#f6f2e9'); draw=ImageDraw.Draw(canvas)
        draw.text((10,10),'ANALYSIS CONTACT SHEET - native 100x100 pixels, labels outside',fill='#222222')
        for j,(ep,index,caption) in enumerate(rows):
            x,y=(j%columns)*220,(j//columns)*145+40
            with Image.open(root/'dist'/ep['frames'][index]) as image: canvas.paste(image,(x+5,y))
            draw.text((x+5,y+103),ep['task_id'],fill='#222222')
            draw.text((x+5,y+116),caption,fill='#222222')
        path=out/f'export-contact-{name}.png'; canvas.save(path); sheets.append(path.relative_to(root).as_posix())
    grid('all-conditions',[(lookup[c['showcase_id']],lookup[c['showcase_id']]['poster_frame_index'],c['condition_id']) for c in manifest['cells']])
    for task in CATALOG['tasks']:
        episodes=[e for e in manifest['episodes'] if e['task_id']==task['id']]
        if episodes: grid(task['id'],[(e,e['poster_frame_index'],f"{e['condition_id']} #{e['native_ordinal']} y={e['label']}") for e in episodes])
    for cell in manifest['cells']:
        ep=lookup[cell['showcase_id']]
        grid(f"{cell['task_id']}--{cell['condition_id']}--all-frames",
             [(ep,i,f"{i}: {p['phase']}") for i,p in enumerate(ep['phases'])],columns=8)
    return sheets


def export_bank(root=DEMO, cells=None, max_attempts=400, max_new_draws=None):
    """Finite greedy coverage curation, atomic after each native candidate.

    Kept trials cover a previously missing category or increase the requested
    photo/color diversity. Sampling never overrides native levels or labels.
    Rejected candidates advance ONLY these isolated demo streams. Completed
    resume validates every asset and performs no further native draws.
    """
    import fcntl
    import platform
    import PIL
    import scipy
    from .validate_assets import coverage_map, validate_episode
    root=Path(root).resolve(); root.relative_to(DEMO)
    (root/'verification').mkdir(parents=True,exist_ok=True)
    (DEMO/'verification').mkdir(parents=True,exist_ok=True)
    lock=(DEMO/'verification/export-render.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close(); raise RuntimeError('One native renderer process at a time')
    try:
        selected_cells = cells if cells is not None else [(t['id'],c['id']) for t in CATALOG['tasks'] for c in t['conditions']]
        selected_cells = [list(c) for c in selected_cells]
        if len({tuple(c) for c in selected_cells})!=len(selected_cells): raise ValueError('Duplicate requested cells')
        for task,cell in selected_cells: DemoStream(task,cell)
        config=dict(namespace=NAMESPACE,cells=selected_cells,max_attempts=max_attempts,frame_ms=300,
                    source=source_hashes(),libraries=dict(python=platform.python_version(),numpy=np.__version__,
                    torch=torch.__version__,pillow=PIL.__version__,scipy=scipy.__version__),
                    exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    validator_sha256=hashlib.sha256(Path(__file__).with_name('validate_assets.py').read_bytes()).hexdigest())
        progress_path=root/'verification/export-progress.json'
        if progress_path.exists():
            progress=json.loads(progress_path.read_text())
            if progress['config']!=config: raise ValueError('Export resume source/config mismatch; preserve old export and use a new Demo-local output root')
            for ep in progress['episodes']: validate_episode(ep,root)
        else:
            progress=dict(config=config,episodes=[],cell_index=0,stream_state=None,draws_by_cell={})
        new_draws=0
        while progress['cell_index'] < len(selected_cells):
            task,cell=selected_cells[progress['cell_index']]
            stream=DemoStream(task,cell)
            if progress['stream_state'] is not None: stream.load_state_dict(progress['stream_state'])
            while True:
                before=coverage_map(progress['episodes'])
                needed=[r for r in before['requirements'] if r['task_id']==task and
                        r['condition_id'] in (None,cell) and not r['satisfied'] and
                        not (task=='image_recognition' and stream.kwargs['load']==0 and r['attribute']=='membership_position')]
                if not needed: break
                if stream.ordinal>=max_attempts:
                    atomic_json(root/'verification/export-coverage-failure.json',dict(task=task,cell=cell,attempts=stream.ordinal,missing=[r['id'] for r in needed]))
                    raise ValueError('Bounded coverage failure: '+', '.join(r['id'] for r in needed))
                if max_new_draws is not None and new_draws>=max_new_draws:
                    return dict(complete=False,episodes=len(progress['episodes']),draws=sum(progress['draws_by_cell'].values()))
                raw,label,metadata=stream.next(); new_draws+=1
                candidate=dict(id='candidate',task_id=task,condition_id=cell,label=label,metadata=metadata)
                after={r['id']:r for r in coverage_map(progress['episodes']+[candidate])['requirements']}
                gains=[r['id'] for r in needed if after[r['id']]['satisfied'] or
                       (r['distinct'] and after[r['id']]['distinct_count']>r['distinct_count'])]
                if gains:
                    episode=export_episode(root,stream,raw,label,metadata,'Native draw selected to add coverage: '+', '.join(gains))
                    validate_episode(episode,root)
                    progress['episodes'].append(episode)
                progress['stream_state']=stream.state_dict()
                progress['draws_by_cell'][f'{task}/{cell}']=stream.ordinal
                atomic_json(progress_path,progress)
            progress['cell_index']+=1; progress['stream_state']=None
            atomic_json(progress_path,progress)
            print(f'{task}/{cell}: {stream.ordinal} native candidates; {sum(e["task_id"]==task and e["condition_id"]==cell for e in progress["episodes"])} kept',flush=True)
        coverage=coverage_map(progress['episodes'])
        manifest_cells=[]
        for task,cell in selected_cells:
            episodes=[e for e in progress['episodes'] if (e['task_id'],e['condition_id'])==(task,cell)]
            def readability(e):
                m=e['metadata']
                return max(float(m.get(k,0)) for k in ('rotation_magnitude','displacement_pixels','contrast_increment','frequency_octave_increment','chromatic_increment','beta_delta'))
            showcase=max(episodes,key=readability)
            manifest_cells.append(dict(task_id=task,condition_id=cell,kwargs=episodes[0]['kwargs'],
                episode_ids=[e['id'] for e in episodes],showcase_id=showcase['id']))
        if source_hashes()!=config['source'] or source_hashes()!=IMPORT_BASELINE:
            raise ValueError('Native sources changed during export')
        manifest=dict(schema_version=1,catalog=copy.deepcopy(CATALOG),
            provenance=dict(config,**config['source'],import_trace=import_trace(),device='cpu',threads=2,
                torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
                thread_environment={k:os.environ[k] for k in THREAD_KEYS},
                seed_derivation='big-endian first 8 SHA256 bytes of namespace/task/cell modulo 2^53; batch size 1',
                source_immutable=True,training_updates=0,model_operations=0,
                curation='Finite greedy example coverage, not representative frequencies; Krauzlis native 57/29/14 sampling unchanged',
                independent_delay_cells=True,draws_by_cell=progress['draws_by_cell']),
            display=dict(mapping='uint8 = rint(clip(float32, 0, 1) * 255)',gamma='none',normalization='none',
                         native_size=[100,100],channels='RGB',raw_layout='TCHW',raw_dtype='little-endian float32',
                         enlargement='integer nearest-neighbor',authoritative='lossless PNG frames',
                         gif_palette='one 256-color median-cut palette over whole episode',gif_dither=False),
            episodes=progress['episodes'],cells=manifest_cells)
        atomic_json(root/'artifacts/manifest.json',manifest)
        atomic_json(root/'artifacts/coverage.json',coverage)
        sheets=contact_sheets(manifest,root)
        files={p for e in manifest['episodes'] for p in e['frames']+[e['gif'],e['poster'],e['frame_zip'],e['metadata_path']]}
        receipt=dict(complete=True,tasks=len({c['task_id'] for c in manifest_cells}),cells=len(manifest_cells),
                     episodes=len(manifest['episodes']),gifs=len({e['gif'] for e in manifest['episodes']}),
                     png_frames=sum(e['frame_count'] for e in manifest['episodes']),draws=sum(progress['draws_by_cell'].values()),
                     asset_bytes=sum((root/'dist'/p).stat().st_size for p in files),
                     raw_bytes=sum((root/e['raw_path']).stat().st_size for e in manifest['episodes']),
                     contact_sheets=sheets,coverage_required=coverage['required_count'],coverage_missing=coverage['missing'],
                     source_immutable=True,model_operations=0,threads=2)
        atomic_json(root/'verification/export-receipt.json',receipt)
        return receipt
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all',action='store_true',required=True)
    parser.add_argument('--device',choices=['cpu'],default='cpu')
    parser.add_argument('--threads',type=int,choices=[2],default=2)
    parser.add_argument('--output',type=Path,default=DEMO)
    parser.add_argument('--max-attempts',type=int,default=400)
    args=parser.parse_args()
    result=export_bank(args.output,max_attempts=args.max_attempts)
    if result['coverage_missing']: raise ValueError('Incomplete coverage')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
