/* Deterministic teaching controls. No checkpoint values or measured performance. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const ink = '#172d36', teal = '#087f81', coral = '#c66249';
  function label(c, s, x, y, size=22) {c.fillStyle=ink;c.font=`${size}px sans-serif`;c.fillText(s,x,y);}
  function clear(id) {const c=$(id).getContext('2d');c.clearRect(0,0,c.canvas.width,c.canvas.height);return c;}
  function softmaxDemo() {
    const gap=+$('pixels-score-gap').value, tau=+$('pixels-temperature').value;
    const right=1/(1+Math.exp(-gap/tau)), left=1-right, mean=3*(right-left);
    const entropy=-(left*Math.log(Math.max(left,1e-30))+right*Math.log(Math.max(right,1e-30)));
    const c=clear('pixels-softmax-canvas');
    [[left,'−3 pixels: left',teal],[right,'+3 pixels: right',coral]].forEach(([p,s,color],i)=>{
      label(c,s,24,48+i*92);c.fillStyle='#ecece6';c.fillRect(245,22+i*92,520,42);
      c.fillStyle=color;c.fillRect(245,22+i*92,520*p,42);label(c,`${(100*p).toFixed(1)}%`,790,51+i*92);
    });
    label(c,`Population mean: ${mean.toFixed(2)} pixels · two alternatives retained`,24,229);
    $('pixels-softmax-readout').textContent=`Score difference ${gap.toFixed(3)} · temperature ${tau.toFixed(3)} · right/left odds ${Math.exp(gap/tau).toPrecision(3)} · mean ${mean.toFixed(3)} pixels · entropy ${entropy.toFixed(3)} / ${Math.log(2).toFixed(3)} nats. These are two-candidate teaching values.`;
  }
  function traceDemo() {
    const rho=+$('pixels-trace-rate').value, r=+$('pixels-trace-reliability').value, h=rho*r;
    let z=Array(81).fill(1/81);const delta=[];
    for(let t=0;t<50;t++) {const p=Array(81).fill(0);p[t<16?42:58]=1;
      delta.push(p.reduce((s,v,k)=>s+Math.abs(v-z[k]),0));z=z.map((v,k)=>(1-h)*v+h*p[k]);}
    const c=clear('pixels-trace-canvas'), x=t=>75+t*17.6, y=v=>238-v*93;
    [0,1,2].forEach(v=>{c.strokeStyle='#d9ded5';c.beginPath();c.moveTo(75,y(v));c.lineTo(938,y(v));c.stroke();label(c,String(v),40,y(v)+7,18);});
    [0,10,20,30,40,49].forEach(t=>label(c,String(t),x(t)-8,262,18));
    c.strokeStyle=coral;c.setLineDash([6,5]);c.beginPath();c.moveTo(x(16),35);c.lineTo(x(16),238);c.stroke();c.setLineDash([]);
    c.strokeStyle=teal;c.lineWidth=3;c.beginPath();delta.forEach((v,t)=>t?c.lineTo(x(t),y(v)):c.moveTo(x(t),y(v)));c.stroke();c.lineWidth=1;
    label(c,'Difference from old trace (sum of absolute differences)',75,23,20);
    label(c,'Prescribed change',x(16)+9,50,18);label(c,'Observation index',735,290,18);
    const half=h===0?'infinite':(Math.log(.5)/Math.log(1-h)).toFixed(2);
    $('pixels-trace-readout').textContent=`Base rate ${rho.toFixed(2)} × reliability ${r.toFixed(2)} = effective write ${h.toFixed(3)} · retention ${(1-h).toFixed(3)} · half-life ${half} updates · initial mismatch ${delta[0].toFixed(3)} · mismatch at observation 49 ${delta[49].toFixed(3)}. Prescribed population.`;
  }
  ['pixels-score-gap','pixels-temperature'].forEach(id=>$(id).addEventListener('input',softmaxDemo));
  ['pixels-trace-rate','pixels-trace-reliability'].forEach(id=>$(id).addEventListener('input',traceDemo));
  softmaxDemo();traceDemo();
  function revealReference() {
    const id=decodeURIComponent(location.hash.slice(1));if(!id)return;
    const target=document.getElementById(id),details=target?.closest('details');
    if(details&&!details.open){details.open=true;target.scrollIntoView();}
  }
  window.addEventListener('hashchange',revealReference);revealReference();
})();
