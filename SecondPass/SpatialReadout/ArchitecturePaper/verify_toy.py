"""Exact rational teaching examples only; no model, inference, or training."""
from fractions import Fraction as F
from pathlib import Path
import json

def matmul(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def trans(a): return [list(x) for x in zip(*a)]
def add(a,b): return [[x+y for x,y in zip(r,s)] for r,s in zip(a,b)]
def scale(c,a): return [[c*x for x in r] for r in a]
def outer(a,b): return [[x*y for y in b] for x in a]
def sub(a,b): return add(a,scale(-1,b))
def col(v): return [[x] for x in v]
def flat(a): return [r[0] for r in a]
def show(a):
    if isinstance(a,list): return [show(x) for x in a]
    return str(a)
I=[[F(1),F(0)],[F(0),F(1)]]
state=[[F(0),F(0)],[F(0),F(0)]]
steps=[([F(1),F(0)],[F(1),F(0)],[F(2),F(4)],[F(1,2),F(3,4)],F(1,2)),
       ([F(3,5),F(4,5)],[F(0),F(1)],[F(2),F(0)],[F(1,2),F(3,4)],F(1,2)),
       ([F(0),F(1)],[F(-1),F(0)],[F(1),F(-1)],[F(4,5),F(3,5)],F(1,4))]
rows=[]; transitions=[]; writes=[]
for t,(k,q,v,alpha,beta) in enumerate(steps):
    assert sum(x*x for x in k)==1 and sum(x*x for x in q)==1
    assert all(0<x<1 for x in alpha) and 0<beta<1
    d=[[alpha[0],F(0)],[F(0),alpha[1]]]
    decayed=matmul(d,state)
    prediction=flat(matmul(trans(decayed),col(k)))
    error=[x-y for x,y in zip(v,prediction)]
    correction=scale(beta,outer(k,error))
    updated=add(decayed,correction)
    read=flat(matmul(trans(updated),col(q)))
    a=matmul(sub(I,scale(beta,outer(k,k))),d)
    b=scale(beta,outer(k,v))
    assert updated==add(matmul(a,state),b)
    transitions.append(a);writes.append(b)
    transport=I; recon=[[F(0),F(0)],[F(0),F(0)]]; coefficients={}
    for tau in range(t,-1,-1):
        recon=add(recon,matmul(transport,writes[tau]))
        kt,qt,vt,at,bt=steps[tau]
        coefficients[str(tau)]=bt*matmul(trans(col(q)),matmul(transport,col(kt)))[0][0]
        transport=matmul(transport,transitions[tau])
    assert recon==updated
    weighted=[sum(coefficients[str(tau)]*steps[tau][2][j] for tau in range(t+1)) for j in range(2)]
    assert weighted==read
    rows.append(dict(t=t,k=show(k),q=show(q),v=show(v),alpha=show(alpha),beta=show(beta),previous=show(state),decayed=show(decayed),prediction=show(prediction),error=show(error),correction=show(correction),updated=show(updated),read=show(read),A=show(a),J=show(b),coefficients={x:show(y) for x,y in coefficients.items()}))
    state=updated
assert rows[1]['updated']==[['101/100','41/50'],['17/25','-6/25']]
assert rows[1]['read']==['17/25','-6/25']
gru_reset=F(1,4)*F(4,5)
gru_update=(1-F(1,4))*F(4,5)+F(1,4)*F(-2,5)
assert gru_reset==F(1,5) and gru_update==F(1,2)
result=dict(scope='Hand-selected 2x2 pedagogical arithmetic, not an actual model result.',exact_rational=True,steps=rows,checks=['normalized keys and queries','gates strictly between zero and one','expanded A recurrence equals direct correction','three-step transport sum equals recurrence','transposed coefficient/value sum equals read'],signed_example=dict(beta='1/2',q=['-1','0'],k=['1','0'],current_coefficient='-1/2',sum_coefficients='-1/2'),gru_example=dict(reset_product=show(gru_reset),updated=show(gru_update)),passed=True)
Path(__file__).with_name('toy_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
