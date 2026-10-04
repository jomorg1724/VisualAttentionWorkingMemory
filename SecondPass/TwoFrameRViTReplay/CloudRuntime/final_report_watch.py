"""One-shot local final reporting from already mirrored artifacts; no GPU/API calls."""
import datetime as dt
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUTPUT = ROOT.parent / 'FINAL_REPORT.md'


def read(path):
    return json.loads(path.read_text())


def publish():
    receipt_path = ROOT / 'retrieval_verified.json'
    if not receipt_path.exists():
        return False
    receipt = read(receipt_path)
    if not receipt.get('complete'):
        return False
    checks = receipt.get('cpu_checkpoint_reload', {})
    if set(checks) != {'selected', 'terminal'} or not all(v.get('verified') for v in checks.values()):
        return False
    artifacts = ROOT / 'artifacts'
    report = read(artifacts / 'report.json')
    if report.get('failure') is not None or not report.get('final_coverage_complete'):
        return False
    progress = [json.loads(line) for line in (artifacts / 'progress.jsonl').read_text().splitlines() if line.strip()]
    recent = progress[-100:]
    loss = sum(row['loss'] for row in recent) / len(recent)
    utc = dt.datetime.now(dt.timezone.utc).isoformat()
    lines = ['# Older cloud RViT — final report', '', f'Reported {utc}.', '',
             f"Completed {report['terminal_step']:,} updates / {report['episodes']:,} presentations; "
             f"validation selected update {report['selected_step']:,}. "
             f"Training loss averaged **{loss:.5f}** over the final {len(recent)} updates.", '',
             'Fresh final tests use 200 trials per condition. These are independent of training and validation.', '',
             '| Checkpoint | Condition | Balanced accuracy | AUC | Target hit | Foil false alarm | Catch false alarm |',
             '|---|---|---:|---:|---:|---:|---:|']
    results = {}
    for role in ('selected', 'terminal'):
        data = read(artifacts / ('test_' + role + '.json'))
        data = data.get('results', data)
        if not data.get('complete'):
            return False
        results[role] = data
        for row in data['cells']:
            lines.append(f"| {role} | {row['cell']} | {row['balanced_accuracy']:.1%} | {row['auc']:.4f} | "
                         f"{row['target_hit_rate']:.1%} | {row['foil_false_alarm_rate']:.1%} | "
                         f"{row['catch_false_positive_rate']:.1%} |")
    lines += ['', 'Training within each repeated pool:', '',
              '| Pool | Unique movies | First epoch mean loss | Last epoch mean loss |',
              '|---|---:|---:|---:|']
    for pool in sorted({row['pool_index'] for row in progress}):
        rows = [row for row in progress if row['pool_index'] == pool]
        first_epoch, last_epoch = min(row['epoch'] for row in rows), max(row['epoch'] for row in rows)
        first = [row['loss'] for row in rows if row['epoch'] == first_epoch]
        last = [row['loss'] for row in rows if row['epoch'] == last_epoch]
        lines.append(f'| {pool+1} | 1,000 | {sum(first)/len(first):.5f} | {sum(last)/len(last):.5f} |')
    acquired = all(row['balanced_accuracy'] >= .70 for row in results['selected']['cells'])
    lines += ['', 'The within-pool losses measure fitting reused training movies; fresh test scores measure generalization.', '',
              ('The selected model meets the recorded acquisition criterion of at least 70% balanced accuracy in every condition.'
               if acquired else 'The selected model does not meet the recorded acquisition criterion of at least 70% balanced accuracy in every condition.'), '',
              'A longer continuation is a possible next experiment. This report does not extend the existing budget or launch another run.', '',
              '[Original final evidence](CloudRuntime/artifacts/REPORT.md) · '
              '[Verified retrieval](CloudRuntime/retrieval_verified.json)', '']
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(OUTPUT)
    journal = REPO / 'LabJournal/krauzlis-rvit-replay.md'
    note = f"\n\n**Final results recorded {utc}:** {report['terminal_step']:,} updates; "
    note += f"final 100-update mean training loss {loss:.5f}. "
    for role, data in results.items():
        cells = data['cells']
        ba = sum(row['balanced_accuracy'] for row in cells) / len(cells)
        auc = sum(row['auc'] for row in cells) / len(cells)
        note += f'{role.capitalize()} fresh mean BA {ba:.2%}, AUC {auc:.4f}. '
    note += '[Final report](../SecondPass/TwoFrameRViTReplay/FINAL_REPORT.md).\n'
    journal.write_text(journal.read_text() + note)
    with (ROOT / 'final_report_status.json').open('w') as stream:
        json.dump(dict(state='reported', utc=utc, report=str(OUTPUT), training_extended=False), stream, indent=2)
    return True


def main():
    with (ROOT / 'final_report_claim.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), started=time.time(), no_training_or_cloud_calls=True), stream)
    deadline = read(ROOT / 'budget.json')['hard_deadline'] + 180
    while time.time() < deadline:
        if publish():
            return
        time.sleep(30)
    (ROOT / 'final_report_status.json').write_text(json.dumps(
        dict(state='awaiting_complete_verified_final_artifacts', report=str(OUTPUT), no_extension=True), indent=2))


if __name__ == '__main__':
    main()
