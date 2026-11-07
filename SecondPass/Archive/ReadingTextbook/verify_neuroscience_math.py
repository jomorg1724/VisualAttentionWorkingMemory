"""Synthetic algebra checks for the reading; no checkpoint or task evaluation."""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(28471)
results = {}
def check(name, a, b, tol=1e-9):
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert err < tol, (name, err)
    results[name] = {'max_absolute_error': err, 'tolerance': tol}

def unit():
    v = rng.normal(size=8)
    return v / np.linalg.norm(v)

def update(S, k, v, alpha, beta):
    retained = alpha[:, None] * S
    error = v - retained.T @ k
    return retained + beta * np.outer(k, error)

S = rng.normal(size=(8,16)); k = unit(); q = unit(); v = rng.normal(size=16)
alpha = rng.uniform(.4,.99,size=8); beta=.37
D = np.diag(alpha); retained=D@S
new = update(S,k,v,alpha,beta)
A=(np.eye(8)-beta*np.outer(k,k))@D
check('ordered_transition_identity',new,A@S+beta*np.outer(k,v))
check('same_key_interpolation',new.T@k,(1-beta)*retained.T@k+beta*v)
check('cross_query_interference',(new-retained).T@q,beta*(k@q)*(v-retained.T@k))
# Independent finite-difference gradient of the stated objective.
def loss(M): return .5*np.sum((v-M.T@k)**2)
grad = np.empty_like(S); eps=1e-6
for idx in np.ndindex(S.shape):
    d=np.zeros_like(S);d[idx]=eps
    grad[idx]=(loss(retained+d)-loss(retained-d))/(2*eps)
check('gradient_step',new,retained-beta*grad,1e-7)
# Ordered unrolling on a changing feature/gate sequence.
state=np.zeros_like(S); history=[]
for _ in range(11):
    ki=unit();vi=rng.normal(size=16);al=rng.uniform(.4,.99,size=8);be=float(rng.uniform(.1,.9))
    Ai=(np.eye(8)-be*np.outer(ki,ki))@np.diag(al)
    history.append((Ai,be,ki,vi));state=update(state,ki,vi,al,be)
P=np.eye(8); expanded=np.zeros_like(S); out=np.zeros(16)
for Ai,be,ki,vi in reversed(history):
    expanded+=np.outer(P@(be*ki),vi)
    out+=(be*q@P@ki)*vi
    P=P@Ai
check('matrix_unrolling',state,expanded)
check('implicit_read_coefficients',state.T@q,out)
# Repeated-key predictions, including nonzero decay.
state=np.zeros_like(S)
for n in range(1,31):
    state=update(state,k,v,np.ones(8),beta)
    check(f'repetition_no_decay_step_{n}',state.T@k,(1-(1-beta)**n)*v)
state=np.zeros_like(S);a=.9
for _ in range(300): state=update(state,k,v,np.full(8,a),beta)
check('decayed_fixed_point',state.T@k,beta/(1-a*(1-beta))*v)
# A matrix difference invisible to one query remains recoverable from another.
hidden=unit();orth=q-hidden*(hidden@q);orth/=np.linalg.norm(orth)
Delta=np.outer(hidden,v)
check('query_nullspace',Delta.T@orth,np.zeros(16))
check('query_recovery',Delta.T@hidden,v)
# Full gated Jacobian formula, using a small dense analogue with reset gate.
n=5; Wz=rng.normal(size=(n,n));Wr=rng.normal(size=(n,n));Wh=rng.normal(size=(n,n))
bz=rng.normal(size=n);br=rng.normal(size=n);bc=rng.normal(size=n);h=rng.normal(size=n)
sig=lambda x:1/(1+np.exp(-x))
def gru(h):
    z=sig(Wz@h+bz);r=sig(Wr@h+br);c=np.tanh(Wh@(r*h)+bc)
    return (1-z)*h+z*c
z=sig(Wz@h+bz);r=sig(Wr@h+br);c=np.tanh(Wh@(r*h)+bc)
Zh=np.diag(z*(1-z))@Wz; Rh=np.diag(r*(1-r))@Wr
Ch=np.diag(1-c*c)@Wh@(np.diag(r)+np.diag(h)@Rh)
J=np.diag(1-z)+np.diag(c-h)@Zh+np.diag(z)@Ch
Jnum=np.column_stack([(gru(h+eps*np.eye(n)[i])-gru(h-eps*np.eye(n)[i]))/(2*eps) for i in range(n)])
check('full_gru_jacobian',J,Jnum,1e-7)
angles=rng.uniform(-np.pi,np.pi,size=(2,200));sample,probe=angles
check('axial_comparison',np.sin(2*(probe-sample)),np.sin(2*probe)*np.cos(2*sample)-np.cos(2*probe)*np.sin(2*sample))
Path('neuroscience_math_verification.json').write_text(json.dumps({'scope':'Synthetic algebra only, not trained-model evidence','checks':results},indent=2)+'\n')
print(f'{len(results)} synthetic algebra checks passed; no model loaded.')
