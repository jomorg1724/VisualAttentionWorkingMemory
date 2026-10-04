"""Pod-local eight-hour guard: verified finalization stops compute, retains disk.
No restart/resume/delete. Publication is attempted before stop, never blocks it.
"""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request

ROOT = Path('/workspace/vawm_sequence_kda16_01')
PUBLIC_KEY = 'VAWM_PUBLIC_STATUS'
MAX_PUBLIC = 25000


def read_json(path):
    try:
        with path.open('rb') as stream:
            raw = stream.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            return {}
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, RecursionError):
        return {}


def process_absent(pid):
    if not isinstance(pid,int) or pid<=0: return False
    try:
        os.kill(pid,0)
        return False  # Even an unreaped zombie or reused PID fails closed.
    except ProcessLookupError: return True
    except (PermissionError,OSError): return False


def verified_finalization(run):
    """Disk evidence AND both execution owners gone; no retrieval dependency."""
    try:
        complete=read_json(run/'cloud_completion.json')
        if complete.get('status')!='complete': return False
        supervisor=read_json(run/'supervisor_result.json')
        owner=read_json(run/'owner_run.json')
        if supervisor.get('returncode')!=0 or supervisor.get('status')!='complete' or owner.get('status')!='complete': return False
        if not all(process_absent(p) for p in (supervisor.get('worker_pid'),supervisor.get('supervisor_pid'),owner.get('pid'))): return False
        rows=complete.get('artifact_manifest',[])
        required={'report.json','test_terminal.json','test_selected.json','selected.pt','supervisor_result.json','owner_run.json'}
        names={r['relative_path'] for r in rows}
        if not rows or len(names)!=len(rows) or not required<=names or not ({'terminal.pt','terminal_validated.pt'}&names): return False
        for r in rows:
            rel=Path(r['relative_path']); p=run/rel
            if rel.is_absolute() or '..' in rel.parts or p.is_symlink() or not p.resolve().is_relative_to(run.resolve()): return False
            if not p.is_file() or p.stat().st_size!=r['bytes']: return False
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
            if h.hexdigest()!=r['sha256']: return False
        report=read_json(run/'report.json')
        if report.get('failure') is not None or report.get('final_coverage_complete') is not True: return False
        step=report['terminal_step']
        pinned=report['config']['max_steps']
        if not isinstance(step,int) or not isinstance(pinned,int) or not 0<step<=pinned: return False
        if step!=complete['actual_updates'] or pinned!=complete['pinned_updates']: return False
        if step<pinned:
            if report.get('stop_reason')!='wall_budget_reserve' or report.get('cap_limited') is not True or report.get('exposure_completed') is not False: return False
        if not isinstance(report.get('selected_step'),int) or not 0<report['selected_step']<=step: return False
        for role in ('terminal','selected'):
            result=read_json(run/('test_'+role+'.json')); result=result.get('results',result)
            cells=result.get('cells',[])
            if result.get('complete') is not True or len(cells)!=3 or {r['cell'] for r in cells}!={'B12','B20','B28'}: return False
            for r in cells:
                if r['task']!='krauzlis_cued_motion' or r['n']!=200: return False
                if {e:r['events'][e]['n'] for e in ('target','foil','catch')}!={'target':114,'foil':58,'catch':28}: return False
        return True
    except (OSError,ValueError,KeyError,TypeError): return False


def stop_reason(now, deadline, boot, run, self_test=False):
    if now>=deadline: return 'wall_deadline'
    failed=read_json(run/'DEPLOYMENT_FAILED')
    if failed.get('stop') is True and failed.get('reason')=='setup_or_worker_failure': return 'setup_or_worker_failure'
    if verified_finalization(run):
        # A short bounded opportunity for the already-running off-pod mirror.
        # Retrieval never extends the hard creation deadline or requires a restart.
        marker=read_json(run/'retrieval_verified.json')
        completion_sha=hashlib.sha256((run/'cloud_completion.json').read_bytes()).hexdigest()
        if marker.get('complete') is True and marker.get('completion_sha256')==completion_sha:
            return 'verified_finalization_retrieved'
        pending=read_json(run/'completion_retrieval_grace.json')
        if not pending:
            try:
                atomic_json(run/'completion_retrieval_grace.json',dict(first_observed=now,grace_seconds=120))
            except OSError:
                return 'verified_finalization_disk_retained'
            return None
        if now>=pending.get('first_observed',now)+120:
            return 'verified_finalization_disk_retained'
        return None
    owner=read_json(run/'owner_claim.json')
    if owner and process_absent(owner.get('pid')):
        # The one-shot owner disappeared without verifiable finalization.
        return 'setup_or_worker_failure'
    # Boot setup is bounded even if the laptop/deployer disappears.
    if now>=boot+900 and not (run/'owner_claim.json').is_file(): return 'setup_or_worker_failure'
    return None


def clean(value, secrets=(), depth=0):
    """Bound public data and redact credential-like fields and error text."""
    if depth > 7:
        return '[depth omitted]'
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, '[redacted]')
        value = re.sub(r'(?i)bearer\s+[^\s\"\',;]+', 'Bearer [redacted]', value)
        value = re.sub(r'(?i)((?:api[_-]?key|token|password|secret|authorization)[\"\']?\s*[:=]\s*)[^\s,;]+',
                       r'\1[redacted]', value)
        return value[:500]
    if isinstance(value, dict):
        return {clean(str(k), secrets): clean(v, secrets, depth + 1)
                for k, v in list(value.items())[:60]
                if not re.search(r'(?i)key|token|secret|password|authorization|env|manifest|traceback|command', str(k))}
    if isinstance(value, list):
        return [clean(v, secrets, depth + 1) for v in value[:30]]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def fields(value, names):
    return {key: value[key] for key in names.split() if key in value}


def snapshot(run, now, state, reason, deadline, secrets=()):
    live = read_json(run / 'live_status.json')
    checkpoint = read_json(run / 'latest_checkpoint.json')
    complete = read_json(run / 'cloud_completion.json')
    failure = read_json(run / 'failure.json')
    supervisor = read_json(run / 'supervisor_result.json')
    if not supervisor:
        supervisor = read_json(run / 'production_supervisor_result.json')
    if not supervisor:
        supervisor = read_json(run / 'production_supervisor.json')
    if not supervisor:
        supervisor = (read_json(run / 'run_supervisor_result.json') or read_json(run / 'run_supervisor.json')
                      or read_json(run / 'profile/profile_supervisor_result.json') or read_json(run / 'profile/profile_supervisor.json'))
    if not supervisor:
        activation = read_json(run / 'activation.json')
        supervisor = dict(status='activated' if activation else 'awaiting_activation',
                          **fields(activation, 'supervisor_pid'))
    evaluations = {}
    groups = {'validation': sorted(run.glob('validation_*.json'), reverse=True),
              'test_selected': [run / 'test_selected.json'],
              'test_terminal': [run / 'test_terminal.json']}
    for role, paths in groups.items():
        for path in paths:
            if 'partial' in path.name:
                continue
            result = read_json(path)
            result = result.get('results', result)
            if isinstance(result, dict) and result.get('complete') is True:
                evaluations[role] = dict(file=path.name, complete=True,
                    cell_count=len(result.get('cells', [])), summary=result.get('summary'))
                break
    data = dict(schema=1, observed=now,
        observed_utc=datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat(),
        guard=dict(state=state, reason=reason, deadline=deadline),
        live_status=fields(live, 'phase pid utc architecture initialization step episodes optimizer_seconds deadline best_step'),
        latest_checkpoint=fields(checkpoint, 'path bytes sha256 verified step'),
        completion=fields(complete, 'status actual_updates target_updates pinned_updates cap_limited exposure_completed error finished_utc'),
        evaluations=evaluations,
        failure=fields(failure, 'type message error_type error utc'),
        supervisor=fields(supervisor, 'status returncode timed_out pid supervisor_pid worker_pid deadline error outcomes'))
    data = clean(data, secrets)
    # Preserve each section even if unusual summaries exceed the provider budget.
    for section, budget in [('live_status', 2500), ('latest_checkpoint', 1500),
                            ('completion', 2500), ('failure', 1500), ('supervisor', 3000)]:
        if len(json.dumps(data[section])) > budget:
            data[section] = {'truncated': True, **fields(data[section], 'status step type message actual_updates error finished_utc')}
    for evaluation in data['evaluations'].values():
        if len(json.dumps(evaluation.get('summary'))) > 3500:
            evaluation['summary'] = {'truncated': True}
    raw = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(raw) > MAX_PUBLIC:
        raise ValueError('Public snapshot budget exceeded')
    return data, raw


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Provider:
    """Key originates only from the pod's existing environment, never logs."""
    def __init__(self, pod, key, template_id):
        if not all(re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in (pod, template_id)):
            raise ValueError('Invalid provider ID')
        self.template_id = template_id
        self.status_url = 'https://api.runpod.io/v2/templates/' + template_id
        self.base = 'https://api.runpod.io/v2/pods/' + pod
        self.stop_url = self.base + '/action'
        self.headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                        'User-Agent': 'runpodctl'}

    def request(self, url, method, body=None):
        request = urllib.request.Request(url, headers=self.headers, method=method,
            data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(request, timeout=10) as response:
            if method == 'GET':
                return json.load(response)
            return response.status

    def get(self):
        return self.request(self.base, 'GET')

    def get_status(self):
        value = self.request(self.status_url, 'GET')
        if value.get('id') != self.template_id:
            raise ValueError('Status template identity mismatch')
        return value

    def patch_status(self, body):
        # This template is PRIVATE and NEVER DEPLOYED. Never PATCH a pod:
        # changing a pod environment would restart its training container.
        return self.request(self.status_url, 'PATCH', body)

    def stop(self):
        return self.request(self.stop_url, 'POST', {'action': 'stop'})


class Guard:
    def __init__(self, root, pod, deadline, provider, boot=None, secrets=()):
        if not math.isfinite(deadline):
            raise ValueError('Deadline must be finite')
        self.root, self.run = Path(root), Path(root) / 'run'
        self.pod, self.deadline, self.provider = pod, deadline, provider
        self.boot = time.time() if boot is None else boot
        self.secrets = secrets
        self.checked = False
        self.last_publish = None
        self.publication_verified = False
        self.publication_error_type = None

    def record(self, now, state, reason, **extra):
        try:
            atomic_json(self.root / 'billing_guard.json', dict(pod=self.pod, pid=os.getpid(),
                boot=self.boot, deadline=self.deadline, updated=now,
                authenticated_read=self.checked, state=state, reason=reason,
                publication_verified=self.publication_verified,
                publication_error_type=self.publication_error_type, **extra))
        except Exception as exc:
            # Disk failure must not disable the deadline. Never print exception text.
            print('billing_guard_write_failed:' + type(exc).__name__, flush=True)

    def get(self):
        pod = self.provider.get()
        if pod.get('id') != self.pod:
            raise ValueError('Pod identity mismatch')
        self.checked = True
        return pod

    def publish(self, now, state, reason):
        self.last_publish = now
        self.publication_verified = False
        self.publication_error_type = None
        try:
            if not self.checked:
                self.get()
            public, raw = snapshot(self.run, now, state, reason, self.deadline, self.secrets)
            step = public['latest_checkpoint'].get('step', public['live_status'].get('step', 0))
            if not isinstance(step, int):
                step = 0
            name = ('vawm-sequence-kda16-01-' + state + ' ' + (reason or 'monitoring') + ' step=' + str(step))[:100]
            public_env = {PUBLIC_KEY: raw}
            self.provider.patch_status(dict(name=name, public=False, env=public_env))
            readback = self.provider.get_status()
            if (readback.get('env') != public_env or readback.get('name') != name
                    or readback.get('public') is not False):
                raise ValueError('Public status readback mismatch')
            self.publication_verified = True
        except Exception as exc:
            self.publication_error_type = type(exc).__name__
        return self.publication_verified

    def tick(self, now):
        reason = stop_reason(now, self.deadline, self.boot, self.run)
        state = 'stop_requested' if reason else 'armed'
        self.record(now, state if self.checked else 'api_retry', reason)
        if reason:
            # Deadline/failure stop has priority over status networking. A slow or
            # unavailable status service must never delay the provider stop call.
            try:
                status = self.provider.stop()
                self.record(now, 'stop_accepted', reason, http_status=status)
            except Exception as exc:
                self.record(now, 'stop_retry', reason, error_type=type(exc).__name__)
        if reason or self.last_publish is None or now - self.last_publish >= 60:
            self.publish(now, state, reason)
        if not reason:
            self.record(now, state if self.checked else 'api_retry', reason)
        return reason


def reap_children():
    """PID1 reaps exited adopted owners; live PID checks stay fail-closed."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def main():
    key = os.environ.pop('VAWM_STOP_API_KEY')
    pod = os.environ['RUNPOD_POD_ID']
    deadline = float(os.environ['VAWM_HARD_DEADLINE'])
    guard = Guard(ROOT, pod, deadline, Provider(pod, key, os.environ['VAWM_STATUS_TEMPLATE_ID']), secrets=(key,))
    # Preserve the existing startup path, without exporting the stop key.
    try:
        subprocess.Popen(['bash', '/start.sh'], env=dict(os.environ))
    except Exception as exc:
        print('start_sh_failed:' + type(exc).__name__, flush=True)
    while True:
        reap_children()
        now = time.time()
        try:
            reason = guard.tick(now)
        except Exception as exc:
            guard.record(now, 'guard_retry', None, error_type=type(exc).__name__)
            reason = None
        # Never exit merely because the provider accepted a stop; retry until off.
        time.sleep(5 if reason else min(10, max(0.1, deadline - time.time())))


if __name__ == '__main__':
    main()
