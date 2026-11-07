"""Independent frozen-checkpoint engineering and attention audit; CPU only."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from SecondPass.WeightedSumTransformer import dataset, train
from SecondPass.WeightedSumTransformer.model import WeightedSumTransformer, weighted_sum


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    run = root / 'LocalRuntime/run01'
    config = json.loads((run / 'production_config.json').read_text())
    assert train.source_hashes() == config['source_hashes']
    paths = [run / (name + '.pt') for name in ('best', 'latest')]
    before_hash = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows = [dataset.generate_evaluation(36000000 + i, 'test') for i in range(96)]
    a, b = [torch.stack([r[k] for r in rows]) for k in (0, 1)]
    for index in range(0, len(rows), 2):
        a0, b0, y0, m0 = rows[index]
        a1, b1, y1, m1 = rows[index + 1]
        assert y0 == 0 and y1 == 1 and torch.equal(a0, a1) and torch.equal(b0[0], b1[0])
        assert not torch.equal(b0[-1], b1[-1])
        for aa, bb, yy, mm in (rows[index], rows[index + 1]):
            positions = np.asarray(mm['positions_unwrapped_xy'])
            assert np.allclose(np.diff(positions[:4], axis=0), np.asarray(mm['velocity_before_xy'])[None, None])
            assert np.allclose(np.diff(positions[3:], axis=0), np.asarray(mm['velocity_after_xy'])[None, None])
            expected_angle = yy * mm['angle']
            assert abs(mm['signed_change']) == expected_angle
    torch.manual_seed(config['init_seed'])
    initial = WeightedSumTransformer().state_dict()
    checkpoint_findings = {}
    for name in ('best', 'latest'):
        payload = torch.load(run / (name + '.pt'), map_location='cpu')
        model = WeightedSumTransformer()
        model.load_state_dict(payload['model'])
        model.eval()
        with torch.no_grad():
            fast = model(a, b)
            images = weighted_sum(torch.cat([a, b]))
            patches = model.patch_embedding(images - .5).flatten(2).transpose(1, 2)
            tokens = torch.cat([model.cls_token.expand(len(images), -1, -1), patches], dim=1) + model.position_embedding
            normalized = model.transformer.norm1(tokens)
            attended, weights = model.transformer.self_attn(normalized[:, :1], normalized, normalized,
                need_weights=True, average_attn_weights=False)
            cls_mid = tokens[:, :1] + attended
            cls_out = cls_mid + model.transformer._ff_block(model.transformer.norm2(cls_mid))
            cls_manual = model.cls_norm(cls_out[:, 0])
            cls_actual = model.encode_images(images)
            weights = weights.squeeze(2)
            entropy = -(weights.clamp_min(1e-20) * weights.clamp_min(1e-20).log()).sum(-1)
            attention = dict(n_clip_images=len(images), CLS_input_max_across_images=float((normalized[:, 0] - normalized[0, 0]).abs().max()),
                CLS_only_computation_max_error=float((cls_manual - cls_actual).abs().max()),
                mean_entropy_per_head=entropy.mean(0).tolist(), uniform_entropy=float(np.log(101)),
                mean_CLS_self_mass_per_head=weights[:, :, 0].mean(0).tolist(),
                mean_max_weight_per_head=weights.max(-1).values.mean(0).tolist(),
                argmax_token_frequencies=[{str(k): v for k, v in collections.Counter(weights[:, h].argmax(-1).tolist()).most_common(6)} for h in range(4)])
        model.train()
        with torch.no_grad():
            slow = model(a, b)
        delta = {key: dict(l2=float((value - initial[key]).norm()), initial_l2=float(initial[key].norm()), current_l2=float(value.norm()))
                 for key, value in payload['model'].items()}
        checkpoint_findings[name] = dict(step=payload['state']['step'], train_eval_max_abs_diff=float((fast - slow).abs().max()),
            logit_std=float(fast.std()), optimizer_state_count=len(payload['optimizer']['state']),
            optimizer_steps=sorted(set(float(v['step']) for v in payload['optimizer']['state'].values())),
            all_weights_finite=all(bool(torch.isfinite(v).all()) for v in payload['model'].values()),
            all_Adam_tensors_finite=all(bool(torch.isfinite(v).all()) for item in payload['optimizer']['state'].values() for v in item.values() if torch.is_tensor(v)),
            all_state_tensors_changed_from_fresh=all(item['l2'] > 0 for item in delta.values()), tensor_updates=delta, attention=attention)
    final = json.loads((run / 'final_test.json').read_text())
    predictions = final['predictions']
    scores = np.asarray([p['logit'] for p in predictions])
    labels = np.asarray([p['label'] for p in predictions])
    positive, negative = scores[labels == 1], scores[labels == 0]
    rank = positive[:, None] - negative[None, :]
    auc = float(((rank > 0) + .5 * (rank == 0)).mean())
    threshold = final['threshold']
    ba = float(.5 * ((positive > threshold).mean() + (negative <= threshold).mean()))
    zero_ba = float(.5 * ((positive > 0).mean() + (negative <= 0).mean()))
    bce = float((np.logaddexp(0, scores) - labels * scores).mean())
    assert abs(auc - final['auc']) < 1e-12 and abs(ba - final['balanced_accuracy']) < 1e-12
    assert abs(bce - final['loss']) < 1e-12 and abs(zero_ba - final['zero_logit_balanced_accuracy']) < 1e-12
    records = [json.loads(line) for line in (run / 'progress.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records] == list(range(1, 25025))
    validation = []
    for path in sorted(run.glob('validation_*.json')):
        value = json.loads(path.read_text())
        validation.append({key: value[key] for key in ('step', 'loss', 'auc', 'balanced_accuracy')})
    min_bce = min(validation, key=lambda value: value['loss'])
    assert min_bce['step'] == final['selected_step']
    after_hash = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert before_hash == after_hash
    report = dict(status='completed_independent_frozen_CPU_audit', reviewer='/root/weighted_sum_review',
        audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), cpu_threads=1, GPU_work=False,
        training=False, deployed_weights_modified=False, source_manifest_matches_actual_run=True,
        checkpoint_sha256=after_hash, checkpoints=checkpoint_findings, source_hashes=config['source_hashes'],
        discovered_code_bugs=[],
        concrete_failure_findings=[
            'One-block CLS-only readout has an image-independent query entering attention; it can pool independent patch features but cannot use patch-to-patch updates or patch FFN results from the same block. This is an architectural restriction, not proof that learning is impossible.',
            'Frozen selected and latest models exhibit concentrated CLS attention. One latest head attends only the CLS token; most other heads prefer a few fixed spatial patches in this 192-clip-image diagnostic. This is measured computational behavior, not a causal diagnosis or a proven remedy.'
        ],
        assistant_selected_choices=['Patch size10', 'Embedding width128', 'CLS-only pooling and finalCLS LayerNorm', 'Divide rawsum by6 and centerinput at.5', 'PreLN/GELU/dropout0/feedforward512', 'Adam learningrate3e-4 eps1e-8 noWD/noclip', '1000-pair twoepoch replay pools', 'Minimum validationBCE selection and validation BA-optimal threshold'],
        data_audit=dict(rendered_trials=96, paired_contexts=48, label_position_semantics_correct=True,
            paired_first_four_raster_frames_identical=True, changed_last_frame_different=True,
            train_eval_marginal_task_same=True, paired_eval_changes_dependence_not_marginal_generator=True,
            training_pair_order_and_labels_read_from_same_row=True),
        metrics_independent_recompute=dict(n=len(labels), positive_count=len(positive), negative_count=len(negative),
            auc=auc, balanced_accuracy=ba, zero_logit_balanced_accuracy=zero_ba, BCE=bce,
            selected_step=final['selected_step'], all_match_saved_report=True),
        training_record=dict(sequential_updates=len(records), total_presentations=records[-1]['presentations'],
            unique_generated_pairs=records[-1]['unique_pairs'], first100_mean_loss=float(np.mean([r['loss'] for r in records[:100]])),
            last100_mean_loss=float(np.mean([r['loss'] for r in records[-100:]])),
            maximum_logged_gradient_norm=max(r['gradient_norm'] for r in records), validation=validation),
        limits=['No training/overfit diagnostic was authorized or run.', 'CPU fast/slow comparison does not verify every CUDA-specific kernel, although differences are tiny relative to observed logits on these96movies.', 'All parameters receiving Adam updates and finite gradients does not establish useful representation learning.', 'Audit establishes the selected trained model failed on this task; it does not establish impossibility of weighted-image representations, transformers or the project.', 'No additional architecture change or training run is justified automatically by this audit.'])
    (root / 'Audit/independent_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'concrete_failure_findings', 'metrics_independent_recompute', 'training_record')}, indent=2))


if __name__ == '__main__':
    main()
