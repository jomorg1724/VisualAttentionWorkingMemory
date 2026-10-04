'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const file = path.join(__dirname, '../web/atlas.js');
const load = () => fs.existsSync(file) ? require(file) : {};
const episode = (n=2) => ({id:'example', frames:Array.from({length:n},(_,i)=>`frame-${i}.png`), frame_count:n, timing:{frame_ms:300}, phases:Array.from({length:n},(_,index)=>({index,phase:'sample',cue_visible:false}))});

test('lossless player exposes every source index and explicitly marks two-frame restart', () => {
  const {Player} = load();
  assert.equal(typeof Player,'function','Player contract must exist');
  const seen=[]; const p=new Player(episode(), {onChange:state=>seen.push(state.index)});
  p.play(); p.tick(0); p.tick(300); assert.equal(p.index,1);
  p.tick(600); assert.equal(p.index,0); assert.equal(p.loops,1);
  p.pause(); p.tick(900); assert.equal(p.index,0);
  p.step(-1); assert.equal(p.index,1); assert.equal(p.playing,false);
  p.seek(100); assert.equal(p.index,1); p.replay(); assert.equal(p.index,0);
  assert.ok(seen.includes(0) && seen.includes(1));
});

test('one scheduler advances all visible players, freezes hidden pages, and honors reduced motion', () => {
  const {Player, Scheduler} = load(); assert.equal(typeof Scheduler,'function');
  const callbacks=[]; const s=new Scheduler({request:fn=>{callbacks.push(fn);return callbacks.length;},cancel:()=>{},reducedMotion:true});
  const a=s.add(new Player(episode(28))); const b=s.add(new Player(episode(7)));
  assert.equal(a.playing,false); s.playAll(); s.tick(0); s.tick(300);
  assert.equal(a.index,1); assert.equal(b.index,1);
  b.setVisible(false); s.tick(600); assert.equal(a.index,2); assert.equal(b.index,1);
  s.setHidden(true); s.tick(90000); assert.equal(a.index,2);
  s.setHidden(false); s.tick(90001); assert.equal(a.index,2); s.tick(90301); assert.equal(a.index,3);
  s.pauseAll(); s.tick(100000); assert.equal(a.index,3); assert.equal(b.index,1);
  s.setRate(2); assert.equal(a.rate,2); assert.equal(b.rate,2);
  s.start(); s.start(); assert.equal(callbacks.length,1,'only one RAF scheduled'); s.destroy();
});

test('long blanks and identical probe rasters retain each original semantic index', () => {
  const {Player}=load(); const e=episode(31); e.frames.fill('identical.png'); const seen=[];
  const p=new Player(e,{onChange:p=>seen.push(p.index)}); p.play(); p.tick(0);
  for(let i=1;i<31;i++)p.tick(i*300);
  assert.deepEqual([...new Set(seen)],Array.from({length:31},(_,i)=>i));
  p.tick(999999); assert.equal(p.index,0,'slow browser never skips several source frames');
});

test('decode backpressure and rate changes cannot display a missing or skipped frame', () => {
  const {Player}=load(); let ready=false; const p=new Player(episode(4),{isReady:()=>ready});
  p.play(); p.tick(0); p.tick(900); assert.equal(p.index,0);
  ready=true; p.tick(1000); assert.equal(p.index,1);
  p.setRate(2); p.tick(1100); p.tick(1250); assert.equal(p.index,2);
  assert.throws(()=>p.setRate(0));
});
