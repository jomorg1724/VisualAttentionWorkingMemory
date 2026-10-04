"""Independent static checks for native atlas inventory, coverage and assets."""
from __future__ import annotations
import os
import sys
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
for _thread_key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_key] = '2'
import json
from pathlib import Path

DEMO = Path(__file__).resolve().parent
CATALOG = json.loads((DEMO.parent / 'catalog.json').read_text())
TASKS = {t['id']: t for t in CATALOG['tasks']}


def validate_inventory(manifest):
    if manifest.get('catalog') != CATALOG:
        raise ValueError('Catalog differs from authoritative source')
    expected = {(t['id'], c['id']): c['kwargs'] for t in CATALOG['tasks'] for c in t['conditions']}
    actual = [(c['task_id'], c['condition_id']) for c in manifest['cells']]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError('Missing, duplicate or extra primary cells')
    for cell in manifest['cells']:
        if cell['kwargs'] != expected[cell['task_id'], cell['condition_id']]:
            raise ValueError('Native condition kwargs changed')
    return {'tasks': len(TASKS), 'cells': len(actual)}


def required_variants():
    """Reviewed finite requirements; cells come only from the native catalog.

    No implicit factorial expansion: only contrast explicitly crosses levels.
    Three orientation-rule outcomes are required at each delay (stronger than
    bank-wide coverage). Both labels/four answers are required in every cell.
    """
    rows = []
    def add(task, attribute, values, cell=None, distinct=False):
        for value in values:
            token = json.dumps(value, separators=(',', ':'), sort_keys=True)
            rows.append(dict(id=f'{task}/{cell or "bank"}/{attribute}/{token}',
                             task_id=task, condition_id=cell, attribute=attribute,
                             value=value, distinct=distinct))
    for task in CATALOG['tasks']:
        for cell in task['conditions']:
            add(task['id'], 'primary', [True], cell['id'])
            labels = [0] if task['id'] == 'image_recognition' and cell['kwargs']['load'] == 0 else list(range(task['classes']))
            add(task['id'], 'label', labels, cell['id'])
            if task['id'] == 'krauzlis_cued_motion':
                add(task['id'], 'event_type', ['target','foil','catch'], cell['id'])
            if task['id'] == 'orientation_cued':
                add(task['id'], 'orientation_case', ['aligned','unchanged','opposite'], cell['id'])
    add('motion_direction', 'displacement_pixels', [1.,2.,3.])
    add('orientation', 'orientation_magnitude', [4.,10.,22.])
    add('contrast', 'contrast_pair', [[i,p] for i in (.025,.06,.13) for p in (.08,.18,.30)])
    add('spatial_frequency', 'frequency_octave_increment', [.08,.18,.35])
    add('chromatic_increment', 'chromatic_increment', [.018,.045,.10])
    add('chromatic_increment', 'base_color', [3], distinct=True)
    add('contour', 'alignment_jitter_degrees', [2.,8.,16.])
    add('natural_spectrum', 'beta_delta', [.15,.30,.60])
    add('natural_spectrum', 'base_id', [3], distinct=True)
    for task in ('orientation_ring','orientation_cued'):
        add(task, 'rotation_magnitude', [15,30,45])
        add(task, 'target_location', list(range(4)))
    add('orientation_cued', 'cue_sign', [-1,1])
    add('motion_duration_cued', 'target_location', list(range(4)))
    add('motion_duration_cued', 'step_pixels', [.8,1.2,1.6])
    add('motion_duration_cued', 'last_differs', [True])
    add('motion_duration_cued', 'foil_differs', [True])
    add('krauzlis_cued_motion', 'target_location', [0,1])
    add('krauzlis_cued_motion', 'event_magnitude', [26,28])
    add('krauzlis_cued_motion', 'signed_event', [-1,1])
    add('spatial_binding', 'target_location', list(range(4)))
    add('image_recognition', 'membership_position', ['early','middle','late'])
    return rows


def episode_features(episode):
    """Derive coverage from native records, never from claimed coverage tags."""
    m, task, label = episode['metadata'], episode['task_id'], episode['label']
    out = dict(m, primary=True, label=label)
    if task == 'orientation':
        out['orientation_magnitude'] = abs(m['signed_orientation_degrees'])
    elif task == 'contrast':
        out['contrast_pair'] = [m['contrast_increment'], m['pedestal']]
    elif task == 'chromatic_increment':
        out['base_color'] = m['frame_colors'][1-label]
    elif task == 'orientation_cued':
        relative = m['rotations_degrees'][m['target_location']] * m['cue_sign']
        out['orientation_case'] = 'aligned' if relative > 0 else 'unchanged' if relative == 0 else 'opposite'
    elif task == 'motion_duration_cued':
        target = m['target_location']
        out['last_differs'] = m['directions_by_patch'][target][-1] != label
        out['foil_differs'] = any(k != label for i,k in enumerate(m['winner_by_patch']) if i != target)
    elif task == 'krauzlis_cued_motion' and m['event_type'] != 'catch':
        out['event_magnitude'] = abs(m['signed_change_degrees'])
        out['signed_event'] = 1 if m['signed_change_degrees'] > 0 else -1
    elif task == 'image_recognition' and label:
        i, n = m['seen_study_index'], m['study_load']
        out['membership_position'] = 'early' if i == 0 else 'late' if i == n-1 else 'middle'
    return out


def coverage_map(episodes):
    indexed = [(e, episode_features(e)) for e in episodes]
    rows = []
    for rule in required_variants():
        matches = [(e,f) for e,f in indexed if e['task_id'] == rule['task_id']
                   and (rule['condition_id'] is None or e['condition_id'] == rule['condition_id'])
                   and rule['attribute'] in f]
        if rule['distinct']:
            distinct = {json.dumps(f[rule['attribute']], sort_keys=True) for _,f in matches}
            satisfied = len(distinct) >= rule['value']
        else:
            matches = [(e,f) for e,f in matches if f[rule['attribute']] == rule['value']]
            distinct = set()
            satisfied = bool(matches)
        rows.append(dict(rule, satisfied=satisfied, episode_ids=[e['id'] for e,_ in matches],
                         gif_ids=[e.get('gif', e['id']) for e,_ in matches],
                         distinct_count=len(distinct) if rule['distinct'] else None))
    missing = [r['id'] for r in rows if not r['satisfied']]
    return dict(schema_version=1, requirements=rows, missing=missing,
                required_count=len(rows), covered_count=len(rows)-len(missing),
                membership_position_definition='early=first, late=last, middle=interior study position',
                diversity_minimum=3, curation='Illustrative coverage, not empirical frequencies; native sampling unchanged')


def validate_coverage(episodes):
    result = coverage_map(episodes)
    if result['missing']:
        raise ValueError('Missing native coverage: ' + ', '.join(result['missing']))
    return result


def expected_frame_count(task, kwargs):
    if TASKS[task]['group'] == 'sensory': return 2
    if task in ('orientation_ring','orientation_cued'): return kwargs['delay'] + 4
    if task == 'motion_duration_cued': return kwargs['delay'] + 11
    if task == 'spatial_binding': return kwargs['delay'] + 5
    if task == 'krauzlis_cued_motion': return kwargs['baseline_transitions'] + 17
    return kwargs['load'] + 4 + kwargs['probe_hold']


def expected_phases(task, metadata, count):
    """Semantic indices from recorded native timing; no brightness heuristics."""
    m = metadata
    phases = ['instruction'] * count
    if TASKS[task]['group'] == 'sensory':
        phases = ['interval 0', 'interval 1']
    else:
        for field, phase in [('sample_frames','sample'), ('study_frames','study'),
                             ('blank_frames','blank'), ('fixation_only_frames','fixation'),
                             ('moving_frames','motion'), ('probe_frames','probe')]:
            for i in m.get(field, []): phases[i] = phase
        for field, phase in [('reference_frame','reference'), ('probe_frame','probe'), ('report_frame','report')]:
            if field in m: phases[m[field]] = phase
        for i in m.get('cue_frames', []):
            if phases[i] == 'instruction': phases[i] = 'query' if task == 'spatial_binding' else 'cue'
        if task == 'krauzlis_cued_motion':
            for i in range(m['reference_frame']+1, m['first_postchange_frame']): phases[i] = 'baseline motion'
            for i in range(m['first_postchange_frame'], m['report_frame']): phases[i] = 'post-event motion'
    return [dict(index=i, phase=p, cue_visible=i in m.get('cue_frames', [])) for i,p in enumerate(phases)]


def validate_episode(episode, root=DEMO):
    """Verify raw/PNG/GIF bytes, semantic frames, membership and downloads."""
    import hashlib
    import zipfile
    import numpy as np
    from PIL import Image, ImageSequence
    root = Path(root).resolve()
    def require(ok, message):
        if not ok: raise ValueError(episode.get('id','?') + ': ' + message)
    def file(relative, base):
        p = (base / relative).resolve()
        require(p.is_relative_to(base), 'asset path escape')
        require(p.is_file(), 'missing asset ' + relative)
        return p
    def sha(data): return hashlib.sha256(data).hexdigest()
    dist = root / 'dist'
    task, m, label = episode['task_id'], episode['metadata'], episode['label']
    require(task in TASKS, 'unknown task')
    condition = next((c for c in TASKS[task]['conditions'] if c['id']==episode['condition_id']), None)
    require(condition is not None and condition['kwargs']==episode['kwargs'], 'invalid native condition')
    require(label in range(TASKS[task]['classes']) and label == m['label'], 'label mismatch')
    require(episode['label_meaning'] == TASKS[task]['labels'][label], 'label meaning mismatch')
    require(episode['native_trial_id']==m['trial_id'], 'native trial identity mismatch')
    require(episode['source_split']=='train' and m['split']=='train', 'non-train source')
    n = expected_frame_count(task, episode['kwargs'])
    raw_file = file(episode['raw_path'], root)
    raw = np.load(raw_file, allow_pickle=False)
    require(raw.dtype == np.float32 and raw.shape == (n,3,100,100), 'raw dimensions/dtype')
    require(np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1, 'raw range')
    require(sha(raw.astype('<f4',copy=False).tobytes())==episode['float_sha256'], 'float hash mismatch')
    require(sha(raw_file.read_bytes())==episode['raw_file_sha256'], 'raw file hash mismatch')
    require(episode['frame_count']==n and len(episode['frames'])==n and len(episode['frame_sha256'])==n, 'frame count')
    require(episode['phases']==expected_phases(task,m,n), 'phase/cue timing mismatch')
    display = np.rint(np.clip(raw,0,1)*255).astype(np.uint8).transpose(0,2,3,1)
    for i, path in enumerate(episode['frames']):
        p = file(path, dist)
        require(sha(p.read_bytes())==episode['frame_sha256'][i], 'PNG file hash')
        with Image.open(p) as image:
            require(image.mode=='RGB' and image.size==(100,100), 'PNG mode/dimensions')
            require(np.array_equal(np.asarray(image), display[i]), 'PNG pixels not exact display mapping')
    for path, digest in episode['asset_sha256'].items():
        require(sha(file(path,dist).read_bytes())==digest, 'asset file hash')
    require(json.loads(file(episode['metadata_path'],dist).read_text())==episode, 'metadata download mismatch')
    with zipfile.ZipFile(file(episode['frame_zip'],dist)) as archive:
        require(len(archive.namelist())==n+1, 'frame ZIP inventory')
        for path in episode['frames']:
            require(archive.read(Path(path).name)==file(path,dist).read_bytes(), 'frame ZIP bytes')
    with Image.open(file(episode['poster'],dist)) as poster:
        require(np.array_equal(np.asarray(poster),display[episode['poster_frame_index']]), 'poster pixels changed')
    # Independently reconstruct fixed-palette quantization, then expand each
    # decoded GIF duration back to source observations (duplicates may merge).
    palette = Image.new('P',(1,1)); palette.putpalette(episode['gif_palette'])
    quantized = [np.asarray(Image.fromarray(x).quantize(palette=palette,dither=Image.Dither.NONE).convert('RGB')) for x in display]
    decoded, mapping = [], []
    with Image.open(file(episode['gif'],dist)) as gif:
        require(gif.size==(100,100) and gif.info.get('loop')==0, 'GIF dimensions/loop')
        for encoded, frame in enumerate(ImageSequence.Iterator(gif)):
            duration = frame.info['duration']
            require(duration > 0 and duration % 300 == 0, 'GIF duration is not native-index multiple')
            start = len(decoded)
            decoded.extend([np.asarray(frame.convert('RGB')).copy()] * (duration//300))
            mapping.append(dict(encoded_index=encoded, source_indices=list(range(start,len(decoded))), duration_ms=duration))
    require(len(decoded)==n, 'decoded GIF timeline frame count')
    require(all(np.array_equal(a,b) for a,b in zip(decoded,quantized)), 'decoded GIF timeline differs from stable-palette source')
    require(mapping==episode['timing']['gif_frame_map'], 'GIF source map mismatch')
    require(episode['timing']['frame_ms']==300 and episode['timing']['total_ms']==n*300, 'illustration timing mismatch')
    conversion=np.abs(raw.astype(np.float64)-display.transpose(0,3,1,2).astype(np.float64)/255)
    require(episode['conversion_error']['max_abs']==float(conversion.max()) and
            episode['conversion_error']['mean_abs']==float(conversion.mean()), 'conversion error measurement mismatch')
    quant_error=np.abs(np.asarray(decoded).astype(np.int16)-display.astype(np.int16))
    require(episode['gif_quantization_error']['max_abs']==int(quant_error.max()) and
            episode['gif_quantization_error']['mean_abs']==float(quant_error.mean()) and
            episode['gif_quantization_error']['normalized_mean_abs']==float(quant_error.mean()/255), 'quantization error measurement mismatch')
    if task.startswith('orientation'):
        if task == 'orientation': semantic = int(m['signed_orientation_degrees']>0)
        else: semantic = int(m['rotations_degrees'][m['target_location']] * (m['cue_sign'] if task=='orientation_cued' else 1)>0)
        require(label==semantic, 'orientation label rule')
    elif task == 'contrast': require(m['frame_contrasts'][label] > m['frame_contrasts'][1-label], 'contrast interval')
    elif task == 'spatial_frequency': require(m['frame_frequencies'][label] > m['frame_frequencies'][1-label], 'frequency interval')
    elif task == 'chromatic_increment':
        difference = np.array(m['frame_colors'][label])-np.array(m['frame_colors'][1-label])
        require(np.allclose(difference,np.array(m['axis_linear_rgb'])*m['chromatic_increment'],atol=1e-7), 'chromatic interval')
    elif task == 'natural_spectrum': require(m['betas_by_frame'][label]<m['betas_by_frame'][1-label], 'spectrum interval')
    elif task == 'motion_direction': require(m['direction']==TASKS[task]['labels'][label], 'motion direction label')
    elif task == 'motion_duration_cued':
        counts = np.array([np.bincount(d,minlength=4) for d in m['directions_by_patch']])
        require(counts.tolist()==m['duration_counts_by_patch'] and np.all(counts.sum(1)==8), 'native eight-transition counts')
        require(all(list(row).count(max(row))==1 for row in counts), 'duration ties')
        require(counts.argmax(1).tolist()==m['winner_by_patch'] and int(counts[m['target_location']].argmax())==label, 'duration winner')
    elif task == 'spatial_binding':
        sample, probe = np.array(m['sample_angles_radians']), np.array(m['probe_angles_radians'])
        pair = m['swapped_locations']
        require(len(pair)==2 and len(set(pair))==2 and label==int(m['target_location'] in pair), 'swap label/pair')
        expected = sample.copy(); expected[pair] = sample[pair[::-1]]
        require(np.array_equal(expected,probe) and np.count_nonzero(sample!=probe)==2, 'exactly one exchanged pair')
    elif task == 'krauzlis_cued_motion':
        require(label==int(m['event_type']=='target'), 'target/foil/catch label')
        require(m['signed_change_degrees']==(0 if m['event_type']=='catch' else m['event_sign']*m['event_magnitude_degrees']), 'signed event')
        require(episode['timing']['native_frame_ms']==10, 'Krauzlis clock')
    elif task == 'image_recognition':
        probes, study = m['probe_frames'], m['study_frames']
        require(all(np.array_equal(raw[probes[0]],raw[i]) for i in probes), 'repeated probes changed')
        hashes = [sha(raw[i].tobytes()) for i in study]
        probe = sha(raw[probes[0]].tobytes())
        require(hashes==m['study_raster_sha256'] and probe==m['probe_raster_sha256'], 'native membership raster hashes')
        require(len(set(hashes))==len(study) and (probe in hashes)==bool(label), 'exact raster membership')
        require(len(study)>0 or label==0, 'empty set positive')
        if label: require(hashes[m['seen_study_index']]==probe, 'serial position mismatch')
    require(all(p.startswith('BSDS500/train/') for p in episode['photo_ids']), 'non-train photo identity')
    return dict(png_exact=True, gif_timeline_exact=True, native_semantics=True,
                frames=n, encoded_gif_frames=len(mapping))


def validate_manifest(manifest, root=DEMO, native=False):
    import hashlib
    import re
    import numpy as np
    root=Path(root).resolve()
    def require(ok,message):
        if not ok: raise ValueError(message)
    result=validate_inventory(manifest)
    episodes=manifest['episodes']; by_id={e['id']:e for e in episodes}
    require(len(by_id)==len(episodes),'Duplicate episode IDs')
    require(all(re.fullmatch(r'[a-zA-Z0-9_-]+',e['id']) for e in episodes),'Unsafe episode ID')
    referenced=[]
    for cell in manifest['cells']:
        require(cell['episode_ids'] and len(set(cell['episode_ids']))==len(cell['episode_ids']),'Empty/duplicate cell bank')
        require(cell['showcase_id'] in cell['episode_ids'],'Showcase not in cell bank')
        for key in cell['episode_ids']:
            require(key in by_id,'Unknown referenced episode')
            episode=by_id[key]
            require((episode['task_id'],episode['condition_id'])==(cell['task_id'],cell['condition_id']),'Episode assigned to wrong cell')
            referenced.append(key)
    require(set(referenced)==set(by_id) and len(referenced)==len(by_id),'Orphaned/multiply referenced episode')
    coverage=validate_coverage(episodes)
    require(json.loads((root/'artifacts/coverage.json').read_text())==coverage,'Saved coverage map mismatch')
    repo=DEMO.parents[2]; provenance=manifest['provenance']
    def snapshot():
        return {relative:hashlib.sha256((repo/relative).read_bytes()).hexdigest() for relative in provenance['source_sha256']}
    before=snapshot()
    require(before==provenance['source_sha256'],'Source hash mismatch')
    dataset=repo/'PreAttentiveVision/data/bsds500/manifest.json'
    require(hashlib.sha256(dataset.read_bytes()).hexdigest()==provenance['dataset_manifest_sha256'],'Dataset manifest hash mismatch')
    photos={r['base_id']:r for r in json.loads(dataset.read_text())['images'] if r['split']=='train'}
    used_photos={p for e in episodes for p in e['photo_ids']}
    require(used_photos.issubset(photos),'Unknown/non-train source identity')
    for photo in used_photos:
        record=photos[photo]
        require(hashlib.sha256((dataset.parent/record['file']).read_bytes()).hexdigest()==record['sha256'],'Source photo bytes changed')
    require(provenance['device']=='cpu' and provenance['threads']<=2 and provenance['torch_threads']<=2 and provenance['torch_interop_threads']<=2,'CPU thread boundary')
    require(provenance['training_updates']==0 and provenance['model_operations']==0,'Model-operation boundary')
    require(set(provenance['import_trace']).issubset({p for p in provenance['source_sha256'] if p.endswith('.py')}),'Unexpected imported project code')
    checks=[validate_episode(e,root) for e in episodes]
    equal_count=0
    if native:
        # Sequential fresh direct-native replay; no model and no live state.
        import fcntl
        from . import export_assets as export
        lock=(DEMO/'verification/export-render.lock').open('a+')
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for cell in manifest['cells']:
                task=cell['task_id']; selected=[by_id[i] for i in cell['episode_ids']]
                seed=selected[0]['seed']
                require(all(e['seed']==seed for e in selected),'Multiple seed identities per cell')
                require(seed==export.DemoStream(task,cell['condition_id']).seed,'Demo seed namespace mismatch')
                cls=export.TaskStream if TASKS[task]['group']=='sensory' else export.VariantStream if task=='orientation_ring' else export.SpatialBatteryStream
                direct=cls(seed,'train'); wanted={e['native_ordinal']:e for e in selected}
                for ordinal in range(max(wanted)+1):
                    args=(1,task) if TASKS[task]['group']=='sensory' else (1,task,cell['kwargs'])
                    x,y,metadata=direct.batch(*args)
                    if ordinal in wanted:
                        episode=wanted[ordinal]
                        require(np.array_equal(x[0].numpy(),np.load(root/episode['raw_path'],allow_pickle=False)),episode['id']+': direct-native raster mismatch')
                        require(int(y[0])==episode['label'] and metadata[0]==episode['metadata'],episode['id']+': direct-native label/metadata mismatch')
                        equal_count+=1
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN); lock.close()
    require(before==snapshot(),'Source changed during validation')
    assets={p for e in episodes for p in e['frames']+[e['gif'],e['poster'],e['frame_zip'],e['metadata_path']]}
    require(len({e['gif'] for e in episodes})==len(episodes),'Duplicate GIF paths')
    result.update(status='passed',episodes=len(episodes),gifs=len(episodes),png_frames=sum(c['frames'] for c in checks),
                  encoded_gif_frames=sum(c['encoded_gif_frames'] for c in checks),
                  frame_zips=len({e['frame_zip'] for e in episodes}),raw_arrays=len({e['raw_path'] for e in episodes}),
                  coverage_required=coverage['required_count'],coverage_covered=coverage['covered_count'],coverage_missing=coverage['missing'],
                  native_equal_episodes=equal_count,native_replay_requested=native,
                  source_immutable=True,train_source_photos=len(used_photos),asset_bytes=sum((root/'dist'/p).stat().st_size for p in assets),
                  uint8_conversion_max_abs=max(e['conversion_error']['max_abs'] for e in episodes),
                  gif_quantization_max_abs=max(e['gif_quantization_error']['max_abs'] for e in episodes),
                  gif_quantization_worst_episode_mean_abs=max(e['gif_quantization_error']['mean_abs'] for e in episodes),
                  models_imported=0,model_operations=0)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all',action='store_true',required=True)
    parser.add_argument('--native',action='store_true',help='Also replay all selected episodes directly from native CPU streams')
    parser.add_argument('--root',type=Path,default=DEMO)
    args=parser.parse_args()
    args.root.resolve().relative_to(DEMO)
    manifest=json.loads((args.root/'artifacts/manifest.json').read_text())
    result=validate_manifest(manifest,args.root,native=args.native)
    path=args.root/'verification'/('export-validation-native.json' if args.native else 'export-validation.json')
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
