/* Native PNG atlas. No stimulus rendering, interpolation or remote requests. */
(function (root) {
  'use strict';
  class Player {
    constructor(episode, options = {}) {
      if (!episode.frames?.length || episode.frames.length !== episode.frame_count) throw new Error('Incomplete native frame sequence');
      this.episode = episode; this.index = 0; this.loops = 0;
      this.playing = false; this.visible = true; this.rate = 1; this.lastTime = null; this.elapsed = 0;
      this.onChange = options.onChange || (() => {});
      this.isReady = options.isReady || (() => true);
    }
    setVisible(value) { this.visible = Boolean(value); this.resetClock(); }
    setRate(value) { if (!(Number(value) > 0) || !Number.isFinite(Number(value))) throw new Error('Playback rate must be positive'); this.rate = Number(value); this.resetClock(); }
    notify() { this.onChange(this); }
    resetClock() { this.lastTime = null; this.elapsed = 0; }
    play() { this.playing = true; this.resetClock(); this.notify(); }
    pause() { this.playing = false; this.resetClock(); this.notify(); }
    seek(index) { this.index = Math.max(0, Math.min(this.episode.frame_count - 1, Math.trunc(Number(index) || 0))); this.resetClock(); this.notify(); }
    step(delta) { this.pause(); this.seek((this.index + delta + this.episode.frame_count) % this.episode.frame_count); }
    replay() { this.loops = 0; this.seek(0); this.play(); }
    tick(now) {
      if (!this.playing || !this.visible) { this.resetClock(); return; }
      if (this.lastTime === null) { this.lastTime = now; return; }
      this.elapsed += Math.max(0, now - this.lastTime); this.lastTime = now;
      const interval = (this.episode.timing?.frame_ms || 300) / this.rate;
      if (this.elapsed < interval) return;
      const next = (this.index + 1) % this.episode.frame_count;
      if (!this.isReady(next)) return;
      // Never skip a semantic source frame to catch up after a slow decode.
      this.elapsed = 0; this.index = next; if (next === 0) this.loops += 1;
      this.notify();
    }
  }
  class Scheduler {
    constructor({request = fn => root.requestAnimationFrame(fn), cancel = id => root.cancelAnimationFrame(id), reducedMotion = false} = {}) {
      this.players = new Set(); this.request = request; this.cancel = cancel; this.hidden = false;
      this.playing = !reducedMotion; this.rate = 1; this.handle = null;
    }
    add(player) { this.players.add(player); player.setRate(this.rate); if (this.playing) player.play(); return player; }
    clear() { this.players.clear(); }
    playAll() { this.playing = true; this.players.forEach(p => p.play()); }
    pauseAll() { this.playing = false; this.players.forEach(p => p.pause()); }
    setRate(rate) { this.players.forEach(p => p.setRate(rate)); this.rate = Number(rate); }
    setHidden(hidden) { this.hidden = hidden; this.players.forEach(p => p.resetClock()); }
    tick(now) { if (!this.hidden) this.players.forEach(p => p.tick(now)); }
    start() { if (this.handle !== null) return; const loop = now => { this.tick(now); this.handle = this.request(loop); }; this.handle = this.request(loop); }
    destroy() { if (this.handle !== null) this.cancel(this.handle); this.handle = null; this.clear(); }
  }
  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const words = value => String(value ?? '').replace(/_/g, ' ');
  const directions = ['right', 'up', 'left', 'down'];
  const canReveal = options => Boolean(options.answers && (!options.trying || options.revealed));
  const canExplain = options => Boolean(canReveal(options) && options.explanation);
  const episodeById = (data, id) => data.manifest.episodes.find(e => e.id === id);
  const taskCells = (data, id) => data.manifest.cells.filter(c => c.task_id === id);
  const catalogTask = (data, id) => data.manifest.catalog.tasks.find(t => t.id === id);
  const format = value => typeof value === 'object' ? JSON.stringify(value) : String(value ?? 'not recorded');
  function variantLabel(tag) {
    const parts=String(tag).split('/'), attribute=parts[2];
    if(!attribute) return words(tag);
    let value=parts.slice(3).join('/'); try {value=JSON.parse(value);} catch { /* Preserve unknown future coverage tokens. */ }
    if(attribute==='contrast_pair'&&Array.isArray(value))return `Increment ${value[0]} · pedestal ${value[1]}`;
    const labels={displacement_pixels:'Displacement (px)',orientation_magnitude:'Magnitude (°)',frequency_octave_increment:'Frequency increment (octaves)',chromatic_increment:'Chromatic increment',alignment_jitter_degrees:'Alignment jitter SD (°)',beta_delta:'Spectral β difference',rotation_magnitude:'Rotation magnitude (°)',target_location:'Target location',cue_sign:'Cue sign',step_pixels:'Motion step (px)',event_type:'Event type',event_magnitude:'Event magnitude (°)',signed_event:'Event sign',orientation_case:'Orientation outcome',membership_position:'Membership position',last_differs:'Last direction differs from winner',foil_differs:'Foil winner differs from target',base_color:'Distinct base-color coverage',base_id:'Distinct source-photo coverage'};
    return `${labels[attribute] || words(attribute)} · ${format(value)}`;
  }
  function detailLink(episode, reader = false) {
    const query = new URLSearchParams({task:episode.task_id, condition:episode.condition_id, example:episode.id});
    if (reader) query.set('mode', 'reader');
    return '#' + query.toString();
  }
  function resolveRoute(hash, data) {
    if (hash === '#conditions') return {view:'conditions'};
    if (!hash || hash === '#' || hash === '#explore') return {view:'gallery'};
    const query = new URLSearchParams(hash.replace(/^#/, ''));
    const task = data.tasks.find(t => t.id === query.get('task'));
    if (!task) return {view:'not-found'};
    const cells = taskCells(data, task.id);
    const cell = query.has('condition') ? cells.find(c => c.condition_id === query.get('condition')) : cells[0];
    if (!cell) return {view:'not-found'};
    const id = query.get('example') || cell.showcase_id;
    const episode = cell.episode_ids.includes(id) && episodeById(data, id);
    if (!episode) return {view:'not-found'};
    return {view:'detail', task, cell, episode, reader:query.get('mode') === 'reader'};
  }
  function phaseGroups(episode, explain = false) {
    const groups = [];
    for (const row of episode.phases) {
      // Never announce an invisible event to an observer, including catch trials.
      let phase = row.phase;
      if (!explain && episode.task_id === 'krauzlis_cued_motion' && /baseline|post.?event|post.?change|event|motion/i.test(phase)) phase = 'motion';
      const previous = groups[groups.length - 1];
      if (previous && previous.phase === phase && previous.cue === Boolean(row.cue_visible) && previous.end + 1 === row.index) previous.end = row.index;
      else groups.push({phase, start:row.index, end:row.index, cue:Boolean(row.cue_visible)});
    }
    return groups;
  }
  function durationEvidence(episode, index) {
    const m = episode.metadata || {}, schedule = m.directions_by_patch?.[m.target_location] || [];
    const count = (m.moving_frames || []).filter(frame => frame <= index).length;
    const counts = [0,0,0,0]; schedule.slice(0,count).forEach(d => { if (counts[d] !== undefined) counts[d] += 1; });
    const totals = [0,0,0,0]; schedule.forEach(d => { if (totals[d] !== undefined) totals[d] += 1; });
    return {counts, transitions:Math.min(count,schedule.length), total:schedule.length, winner:totals.indexOf(Math.max(...totals)), finalDirection:schedule.at(-1)};
  }
  function citationText(text) { return esc(text).replace(/(?<![\w])\[([a-zA-Z0-9_.:-]+)\]/g, (_, id) => `<a class="citation" href="#ref-${esc(id)}" data-reference="${esc(id)}">[${esc(id)}]</a>`); }
  const paragraphs = list => (Array.isArray(list) ? list : [list]).filter(Boolean).map(p => `<p>${citationText(p)}</p>`).join('');
  function nativeImage(episode, index = 0, extra = '') {
    return `<img class="native-image" src="${esc(episode.frames[index])}" width="100" height="100" alt="Native stimulus, frame ${index}; no annotations" ${extra}>`;
  }
  function playerMarkup(episode, options = {}, compact = false) {
    const phases = phaseGroups(episode, canExplain(options));
    return `<section class="player ${compact ? 'compact-player' : 'detail-player'}" data-player="${esc(episode.id)}" aria-label="Native frame player">
      <div class="scene-stage"><div class="native-frame">${nativeImage(episode, 0, 'data-scene loading="lazy" decoding="async"')}</div></div>
      <div class="scene-caption"><span>100 × 100 · lossless PNG</span><span data-loop>Native trial · first pass</span></div>
      <div class="transport"><button type="button" data-action="toggle-player" aria-label="Play or pause this trial">Pause</button>${compact ? '' : '<button type="button" data-action="replay" title="Replay from frame zero">↺ Replay</button><button type="button" data-action="previous-frame" aria-label="Previous frame">←</button><button type="button" data-action="next-frame" aria-label="Next frame">→</button>'}<output data-counter>Frame 0 / ${episode.frame_count - 1}</output></div>
      ${compact ? '' : `<label class="frame-slider">Frame <input type="range" data-frame min="0" max="${episode.frame_count - 1}" step="1" value="0" aria-label="Native frame index"></label><div class="phase-readout"><span data-phase>${esc(words(phases[0]?.phase))}</span><span data-cue>${phases[0]?.cue ? 'Cue visible' : 'Cue absent'}</span></div><div class="timeline" aria-label="Native phase timeline">${phases.map(p => `<button type="button" data-seek="${p.start}" data-end="${p.end}" style="flex-grow:${p.end - p.start + 1}" title="${esc(words(p.phase))}: frames ${p.start}–${p.end}${p.cue ? '; cue visible' : ''}"><span>${esc(words(p.phase))}</span><small>${p.start === p.end ? p.start : p.start+'–'+p.end}${p.cue ? ' · cue' : ''}</small></button>`).join('')}</div><p class="microcopy">Indices are zero-based; every blank and repeated frame remains an input. Loop restart is a new presentation, not an extra motion transition.</p>`}
    </section>`;
  }
  function renderGallery(data, options = {}, wall = false) {
    const groups = [{id:'sensory',title:'Sensory discrimination',description:'Two ordered observations. One precise difference.'},{id:'selection',title:'Spatial selection & comparison',description:'Which location matters changes what must be compared.'},{id:'retention',title:'Retention & recognition',description:'Keep the relevant evidence across time.'}];
    const groupFor = task => groups.some(g => g.id === task.group) ? task.group : (task.id === 'image_recognition' || task.id === 'spatial_binding' ? 'retention' : 'selection');
    const tiles = task => (wall ? taskCells(data,task.id) : taskCells(data,task.id).slice(0,1)).map(cell => {
      const e = episodeById(data,cell.showcase_id); if (!e) return '';
      const linkEpisode = {...e,task_id:task.id,condition_id:cell.condition_id};
      return `<article class="task-tile"><div class="tile-kicker"><span>${esc(cell.condition_id)}</span><span>${e.frame_count} native frames</span><a href="${esc(e.poster)}" download>Static poster ↓</a></div>${playerMarkup(e,options,true)}<div class="tile-copy"><h3><a href="${esc(detailLink(linkEpisode))}">${esc(task.title)}</a></h3><p>${esc(task.question)}</p><div class="tile-bottom"><a class="text-link" href="${esc(detailLink(linkEpisode))}">Explore task <span aria-hidden="true">↗</span></a><span class="tile-answer">${canReveal(options) ? 'Answer: '+esc(e.label_meaning) : 'Answer hidden'}</span></div></div></article>`;
    }).join('');
    return `<header class="page-intro"><p class="eyebrow">A field guide to visual computation</p><h1>${wall ? 'Every native condition.' : 'Seeing, selecting & remembering.'}</h1><div class="intro-bottom"><p>${wall ? `All ${data.manifest.cells.length} native conditions, each with its own real sequence. Delay cells are independent example streams—not paired versions of one trial.` : `Explore ${data.tasks.length} experiments in seeing, selecting and remembering. These are the native inputs—not illustrations of a model’s internal state or evidence of learned performance.`}</p><a class="text-link" href="${wall ? '#explore' : '#conditions'}">${wall ? 'Back to the task atlas' : `View all ${data.manifest.cells.length} conditions`} <span aria-hidden="true">↗</span></a></div></header>
      <div class="field-note"><span class="note-mark" aria-hidden="true">↳</span><p>All visible scenes can play together. Playback is slowed for inspection; it is not calibrated experimental timing. Open a task to step through every native frame.</p></div>
      ${groups.filter(g => data.tasks.some(t => groupFor(t) === g.id)).map(g => `<section class="gallery-section" aria-labelledby="group-${g.id}"><div class="section-heading"><h2 id="group-${g.id}">${g.title}</h2><p>${g.description}</p></div><div class="task-grid">${data.tasks.filter(t => groupFor(t) === g.id).map(tiles).join('')}</div></section>`).join('')}`;
  }
  function facts(rows) { return `<dl class="evidence-facts">${rows.filter(([,value]) => value !== undefined).map(([key,value]) => `<div><dt>${esc(key)}</dt><dd>${esc(format(value))}</dd></div>`).join('')}</dl>`; }
  function locationSchematic(m, index) {
    if (m.target_location === undefined) return '';
    const xy = m.positions_xy || [[27,27],[73,27],[27,73],[73,73]];
    const cueVisible = (m.cue_frames || []).includes(index);
    return `<div class="location-schematic"><svg viewBox="0 0 140 125" role="img" aria-label="External location schematic; selected location ${m.target_location}">${xy.map(([x,y],i) => `<circle cx="${x+20}" cy="${y+10}" r="14" class="${i === m.target_location ? 'selected-location' : ''}"/><text x="${x+20}" y="${y+15}" text-anchor="middle">${i}</text>`).join('')}</svg><p>Selected location <strong>${m.target_location}</strong>${m.cue_sign !== undefined ? ` · sign <strong>${m.cue_sign > 0 ? '+' : '−'}</strong>` : ''}<br><span data-analysis-cue>${cueVisible ? 'Native cue visible now' : 'Native cue absent now'}</span><br><small>Schematic only. The true cue is in the unmodified scene.</small></p></div>`;
  }
  function renderAnalysis(episode, index = 0) {
    const m = episode.metadata || {}, id = episode.task_id;
    let body = '';
    if (episode.frame_count === 2) {
      body = `<p class="microcopy">A/B analysis display, not simultaneous model input. Native replay above presents these two observations in order.</p><div class="ab-filmstrip">${[0,1].map(i => `<figure>${nativeImage(episode,i,'loading="lazy"')}<figcaption>${i === 0 ? 'A' : 'B'} · frame ${i}</figcaption></figure>`).join('')}</div>`;
      const evidence = {
        motion_direction:[['Direction',episode.label_meaning],['Displacement (pixels)',m.displacement_pixels]],
        orientation:[['Signed axial change (degrees)',m.signed_orientation_degrees],['Base angle (degrees)',m.base_orientation_degrees]],
        contrast:[['Frame amplitudes [A, B]',m.frame_contrasts],['Increment',m.contrast_increment],['Pedestal',m.pedestal]],
        spatial_frequency:[['Frame frequencies',m.frame_frequencies],['Octave increment',m.frequency_octave_increment]],
        chromatic_increment:[['Positive increment',m.chromatic_increment],['Linear-RGB axis',m.axis_linear_rgb],['Patch RGB values [A, B]',m.frame_colors],['Numerical luminances [A, B]',m.frame_luminances]],
        contour:[['Structured interval (zero-based)',episode.label],['Alignment jitter SD (degrees)',m.alignment_jitter_degrees]],
        natural_spectrum:[['Beta delta',m.beta_delta],['Frame betas [A, B]',m.betas_by_frame],['Shared RMS',m.actual_common_rms],['Source photograph',m.base_id]]
      };
      const equations = {orientation:'Δθ = ½ atan2(sin 2(θB − θA), cos 2(θB − θA))',spatial_frequency:'f_high / f_low = 2^δ',chromatic_increment:'axis ∝ (0.7152, −0.2126, 0); luminance weights · axis = 0',contrast:'Compare modulation amplitude; do not normalize each frame.',contour:'Compare the arrangement, not a guessed path drawn over pixels.',natural_spectrum:'Smaller spectral β → greater relative high-frequency detail.'};
      body += facts(evidence[id] || []) + (id === 'motion_direction' ? `<div class="direction-diagram" aria-label="External displacement schematic">A <span>${['→','↑','←','↓'][episode.label]}</span> B</div>` : `<p class="equation">${esc(equations[id] || '')}</p>`);
      const chosen=Number(episode.label), other=1-chosen, chosenName=chosen===0?'A':'B', otherName=other===0?'A':'B';
      const numeric=value=>value === undefined ? 'not recorded' : Number(value).toPrecision(6).replace(/0+$/,'').replace(/\.$/,'');
      const reasoning={
        motion_direction:`Surviving dot identities move ${episode.label_meaning} by ${format(m.displacement_pixels)} pixels. The alternatives specify a different displacement vector; reborn dots are nuisance variation, not a second direction to report.`,
        orientation:`The recorded axial change is ${format(m.signed_orientation_degrees)}°. Its sign is ${m.signed_orientation_degrees>0?'positive':'negative'}, not ${m.signed_orientation_degrees>0?'negative':'positive'}. Independent carrier phases do not change this sign rule.`,
        contrast:`${chosenName} has amplitude ${numeric(m.frame_contrasts?.[chosen])}; ${otherName} has ${numeric(m.frame_contrasts?.[other])}. Choose the greater amplitude, not the pedestal alone or the phase.`,
        spatial_frequency:`${chosenName} has ${numeric(m.frame_frequencies?.[chosen])} cycles/image versus ${numeric(m.frame_frequencies?.[other])} in ${otherName}. The higher frequency is the answer; the envelope size does not change.`,
        chromatic_increment:`${chosenName} receives the positive ${format(m.chromatic_increment)} increment along the recorded linear-RGB axis; ${otherName} is the base color. Numerical luminance matching does not make the chromatic difference disappear, and it is not observer-calibrated isoluminance.`,
        contour:`${chosenName} is generated with the structured seven-element path; ${otherName} reassigns the orientation multiset. The other interval may contain accidental alignment, but it was not generated as the structured interval. No guessed contour is drawn over the inputs.`,
        natural_spectrum:`${chosenName} has β = ${numeric(m.betas_by_frame?.[chosen])}, smaller than ${numeric(m.betas_by_frame?.[other])} in ${otherName}. That is greater relative high-frequency weighting, not a different scene or greater overall RMS contrast.`
      };
      body += `<p class="worked-reason">${esc(reasoning[id] || '')}</p>`;
    } else if (id === 'motion_duration_cued') {
      const d = durationEvidence(episode,index);
      body = locationSchematic(m,index) + `<p>Selected-patch evidence through the current frame: <strong data-transition-count>${d.transitions}</strong> / ${d.total} transitions.</p><div class="count-bars">${directions.map((name,i) => `<div><span>${name}</span><meter data-direction-meter="${i}" min="0" max="${d.total || 8}" value="${d.counts[i]}" aria-label="${name} transition count"></meter><strong data-direction-count="${i}">${d.counts[i]}</strong></div>`).join('')}</div><p>Full-trial duration winner: <strong>${esc(directions[d.winner])}</strong>. Final direction: <strong>${esc(directions[d.finalDirection])}</strong>. ${d.winner !== d.finalDirection ? 'The last direction is not the answer.' : 'They happen to agree in this example; the rule is still the count winner.'}</p><p class="microcopy">Counts are derived from the native schedule, not decoded neural evidence. Foil patches have their own schedules.</p>`;
    } else if (id === 'krauzlis_cued_motion') {
      body = locationSchematic(m,index) + `<div class="event-line" aria-label="Explanation-only event line"><span>Reference<br>frame ${esc(m.reference_frame)}</span><span class="event-marker">${m.event_type === 'catch' ? 'Virtual event boundary' : 'First post-change frame'}<br><strong>${esc(m.first_postchange_frame ?? m.virtual_event_frame)}</strong></span><span>Report<br>frame ${esc(m.report_frame)}</span></div>` + facts([['Event type',m.event_type],['Target patch',m.target_location],['Changed patch',m.changed_patch],['Signed event (degrees)',m.signed_change_degrees]]) + '<p>Only a target event is positive. A real foil event and a catch are both negative. These curated examples do not estimate the native 57 / 29 / 14 mixture.</p>';
    } else if (id === 'spatial_binding') {
      const pair = m.swapped_locations || [];
      body = locationSchematic(m,index) + `<p>Exactly one pair exchanges orientations; the inventory is preserved. Location <strong>${esc(m.target_location)}</strong> is queried only after retention.</p><div class="binding-map" aria-label="Revealed sample to probe correspondence">${(m.sample_angles_radians || []).map((angle,i) => { const destination = pair.includes(i) ? pair.find(p => p !== i) : i; return `<div class="${pair.includes(i) ? 'exchanged' : ''}"><span>Sample site ${i}<small>${(angle*180/Math.PI).toFixed(1)}°</small></span><span aria-hidden="true">${pair.includes(i) ? '⇢' : '→'}</span><span>Probe site ${destination}</span></div>`; }).join('')}</div><p>Swapped locations: <strong>${esc(pair.join(' ↔ '))}</strong>. ${pair.includes(m.target_location) ? 'The queried item participates, so this is a target exchange.' : 'Only foils exchange; the queried item stays at its location. This is not a no-change trial.'}</p>`;
    } else if (id === 'image_recognition') {
      const study = m.study_frames || [], probes = m.probe_frames || [], repeat = probes.filter(i => i <= index).length;
      body = `<p>${study.length ? `All ${study.length} study items in their native order.` : 'Empty study set: no item can be a member.'} Matching is revealed below; this filmstrip is an analysis display, never an extra input.</p><div class="study-filmstrip">${study.map((frame,i) => `<figure class="${i === m.seen_study_index ? 'matching-item' : ''}">${nativeImage(episode,frame,'loading="lazy"')}<figcaption>Study ${i+1}${i === m.seen_study_index ? ' · match' : ''}</figcaption></figure>`).join('')}</div><div class="recognition-sequence"><span>${study.length} study frames</span><span>→ ${m.blank_frames?.length ?? 3} blank frames →</span><span>Identical probe <strong data-probe-counter>${repeat}</strong> / ${probes.length}</span></div>${facts([['Membership',episode.label ? `Exact raster match at study position ${Number(m.seen_study_index)+1}` : 'Probe absent from study set'],['Repeated probe exposures',probes.length]])}<p>Repeated probes are the same observation, not new study items or independent trials. ${study.length === 0 ? 'For N0, use specificity and false-positive rate; BA/AUC are not defined.' : ''}</p>`;
    } else {
      const delta = m.rotations_degrees?.[m.target_location];
      body = locationSchematic(id === 'orientation_ring' ? {...m,cue_sign:undefined} : m,index) + facts([['Target signed change (degrees)',delta],['Sample angles (radians)',m.sample_angles_radians],['Probe angles (radians)',m.probe_angles_radians]]) + `<p>${id === 'orientation_cued' ? `The rule combines the location and cue sign: y = 1[c × Δθ_target > 0]. Here ${esc(m.cue_sign)} × ${esc(delta)} ${m.cue_sign * delta > 0 ? '> 0: aligned.' : '≤ 0: unchanged or opposite.'}` : 'Report the numeric sign of the ring-selected change. Independent foil rotations are not the answer; there is no additional sign instruction.'}</p>`;
    }
    return `<section class="analysis-panel" aria-label="External explanation"><p class="eyebrow">Explanation · ground truth, not model state</p>${body}</section>`;
  }
  const sectionNames = {appears:'What appears',sequence:'Sequence & timing',rules:'Decision rule',ignore:'What to ignore',implementation:'How it is generated',neuroscience:'Neuroscience motivation',network:'What a network must preserve',boundary:'Interpretation boundary',metrics:'Metrics—not measurements',adaptation:'Adapted here'};
  function renderReferences(task, sources) {
    return `<details class="essay-section" data-section="references"><summary>Sources & claim boundaries</summary><div class="section-body">${(task.references || []).map(id => { const s = sources.find(s => s.id === id); if (!s) return `<p>Source unavailable: ${esc(id)}</p>`; return `<article class="reference" id="ref-${esc(id)}"><h3>${esc(s.title)}</h3><p>${esc(format(s.authors))} · ${esc(s.year)} · ${esc(s.type)}</p><p>${s.url || s.doi ? `<a href="${esc(s.url || ('https://doi.org/'+s.doi))}" target="_blank" rel="noopener noreferrer">Source record ↗</a> <span class="microcopy">External link; the atlas itself works offline.</span>` : `<span class="microcopy">Repository provenance · ${esc(s.repository_path || 'local executable source')} · source hashes in manifest</span>`}</p>${facts([['Verification',s.verification],['Supporting location',s.support_location],['Supports',s.supports],['Does not support',s.does_not_support]])}</article>`; }).join('')}</div></details>`;
  }
  function renderDetail(data, route, options = {}) {
    const {task,episode:e,cell,reader} = route, show = canReveal(options), explain = canExplain(options);
    const cells = taskCells(data,task.id), examples = cell.episode_ids.map(id => episodeById(data,id)).filter(Boolean);
    const labels = catalogTask(data,task.id)?.labels || [];
    const sectionMarkup = Object.entries(sectionNames).filter(([key]) => task.sections?.[key]).map(([key,title]) => `<details class="essay-section" data-section="${key}" ${reader || options.full ? 'open' : ''}><summary>${title}</summary><div class="section-body">${paragraphs(task.sections[key])}</div></details>`).join('');
    const variantTags = [...new Set(examples.flatMap(item => item.covered_variants || []))].filter(tag=>!['primary','label'].includes(String(tag).split('/')[2]));
    const answer = show ? `<div class="answer-box" role="status"><p class="eyebrow">Generator’s answer</p><h3>${esc(e.label_meaning)}</h3><p>Class ${esc(e.label)}${options.response !== undefined ? ` · ${Number(options.response) === Number(e.label) ? 'Your judgment agrees.' : 'Your judgment differs.'}` : ''}</p>${!explain ? '<p>Switch to Explanation for the recorded evidence and worked reasoning.</p>' : ''}</div>` : '<p class="answer-withheld">The answer and trial-specific metadata are hidden.</p>';
    return `<div class="detail-view ${reader ? 'reader-view' : ''}"><nav class="breadcrumbs" aria-label="Breadcrumb"><a href="#explore">Atlas</a><span>/</span><a href="#conditions">Conditions</a><span>/</span><span>${esc(task.title)}</span></nav><header class="task-heading"><p class="eyebrow">${esc(words(task.group))} · ${esc(cell.condition_id)}</p><h1>${esc(task.title)}</h1><p class="task-question">${esc(task.question)}</p><div class="view-tabs" aria-label="Reading mode"><a ${!reader ? 'aria-current="page"' : ''} href="${esc(detailLink(e))}">Explore</a><a ${reader ? 'aria-current="page"' : ''} href="${esc(detailLink(e,true))}">Learn / Inspect</a></div></header>
      <div class="detail-layout"><div class="experiment-column"><div class="selection-controls"><label>Primary condition<select data-condition>${cells.map(c => `<option value="${esc(c.condition_id)}" ${c.condition_id === cell.condition_id ? 'selected' : ''}>${esc(c.condition_id)}</option>`).join('')}</select></label><label>Example trial<select data-example>${examples.map((item,i) => `<option value="${i}" ${item.id === e.id ? 'selected' : ''}>Example ${i+1}${show ? ' · '+esc(item.label_meaning) : ''}</option>`).join('')}</select></label></div>
      <div class="example-navigation"><button data-action="previous-example">← Previous example</button><span>${examples.findIndex(item => item.id === e.id)+1} / ${examples.length}</span><button data-action="next-example">Next example →</button></div>
      ${show && variantTags.length ? `<label class="variant-picker">Stimulus variants <select data-variant><option value="">Choose a covered example…</option>${variantTags.map(tag => `<option value="${esc(tag)}">${esc(variantLabel(tag))}</option>`).join('')}</select></label>` : '<p class="microcopy">Variant and outcome labels stay hidden until answers are revealed.</p>'}
      ${playerMarkup(e,options)}<p class="mobile-task-question">${esc(task.question)}</p><div class="local-controls"><div class="segmented" aria-label="Scene view"><button data-action="observer" aria-pressed="${!options.explanation}">Observer</button><button data-action="explanation" aria-pressed="${Boolean(options.explanation)}" ${options.trying && !options.revealed ? 'disabled' : ''}>Explanation</button></div><label>Demo speed<select data-local-rate>${[0.25,0.5,1,2,4].map(rate => `<option value="${rate}" ${Number(options.rate || 1) === rate ? 'selected' : ''}>${rate}×</option>`).join('')}</select></label></div>
      <p class="timing-note">Illustration clock: ${(e.timing?.frame_ms || 300)} ms per frame at 1×. ${task.id === 'krauzlis_cued_motion' ? 'Native source clock: 100 Hz (10 ms/frame). This browser playback is slowed, not a calibrated 100 Hz presentation.' : 'Native delays are frame counts; no biological duration is assigned.'}</p>
      <p class="microcopy">Display mapping: clip to [0,1], round(value × 255). No per-frame normalization or gamma correction. Browser colors are not calibrated; PNGs preserve display rasters, while GIF previews are quantized.</p>
      <div data-analysis>${explain ? renderAnalysis(e,0) + (task.worked_examples?.[e.id] ? `<section class="analysis-panel"><p class="eyebrow">This example, worked through</p>${paragraphs(task.worked_examples[e.id])}</section>` : '') : options.explanation ? '<p class="field-note">Reveal answers to show the external analysis. No explanatory graphics are drawn into the native pixels.</p>' : ''}</div>
      <div class="downloads"><span>Take this trial</span>${options.trying && !options.revealed ? '<p>Downloads available after judgment reveal.</p>' : `<a href="${esc(e.gif)}" download>Animated GIF ↓</a><a href="${esc(e.frame_zip)}" download>PNG frames ↓</a>${show ? `<a href="${esc(e.metadata_path)}" download>Trial metadata ↓</a>` : '<span class="microcopy">Metadata download withheld with answers.</span>'}<a href="${esc(e.poster)}" download>Poster ↓</a>`}</div></div>
      <aside class="reading-column"><p class="lead">${citationText(task.intro)}</p><div class="judgment"><button class="primary-button" data-action="try">${options.trying && !options.revealed ? 'Restart judgment' : 'Try the judgment'}</button><p class="microcopy">Informal exploration, not a calibrated psychophysical test. No response-time measurement.</p>${options.trying && !options.revealed ? `<fieldset><legend>Your response</legend>${labels.map((label,i) => `<button data-judgment="${i}">${esc(label)}</button>`).join('')}<button class="text-button" data-action="reveal">Reveal without a response</button><button class="text-button" data-action="cancel-judgment">Leave judgment</button></fieldset>` : ''}${answer}</div>
      <div class="reading-controls"><button data-action="overview">Overview</button><button data-action="expand">Full description / Expand all</button><button data-action="print">Print</button></div>${sectionMarkup}${task.equations?.length ? `<details class="essay-section" data-section="equations" ${reader || options.full ? 'open' : ''}><summary>Mathematical rule</summary><div class="section-body">${task.equations.map(eq => `<p class="equation">${esc(eq)}</p>`).join('')}</div></details>` : ''}${renderReferences(task,data.sources).replace('<details ',`<details ${reader || options.full ? 'open ' : ''}`)}</aside></div>
      <section class="condition-section"><div class="section-heading"><h2>Primary conditions</h2><p>Independent streams, not matched or paired trials.</p></div><div class="condition-matrix">${cells.map(c => { const item=episodeById(data,c.showcase_id); return `<a class="condition-link" ${c.condition_id === cell.condition_id ? 'aria-current="true"' : ''} href="${esc(detailLink(item,reader))}"><strong>${esc(c.condition_id)}</strong><span>${item.frame_count} frames · ${c.episode_ids.length} examples</span></a>`; }).join('')}</div></section>
      <section class="technical-section"><h2>Under the surface</h2><p>Task properties and source provenance. These descriptions, labels and analysis metadata are not model inputs; task identity selects an external supervised head.</p><details class="essay-section" data-section="properties" ${reader || options.full ? 'open' : ''}><summary>Implementation properties</summary><div class="table-scroll"><table><caption>Native generator properties; continuous ranges are not a finite variant inventory.</caption><thead><tr><th scope="col">Property</th><th scope="col">Value / units</th><th scope="col">Sampling & role</th><th scope="col">Visibility</th><th scope="col">Code provenance</th></tr></thead><tbody>${(task.properties || []).map(p => `<tr><th scope="row">${esc(p.name)}</th><td>${esc(format(p.value))}<small>${esc(p.units)}</small></td><td>${esc(format(p.sampling))}<small>${esc(p.role)}</small></td><td>${esc(p.visibility)}</td><td>${esc(format(p.provenance))}</td></tr>`).join('')}</tbody></table></div></details>
      ${show ? `<details class="essay-section" data-section="metadata" ${reader ? 'open' : ''}><summary>Current trial · native metadata & provenance</summary><div class="section-body"><p>Renderer values below are unchanged. Worked graphics outside the scene are analyst-derived. These are curated illustrations, not evaluation data.</p>${facts([['Example',e.id],['Demo seed',e.seed],['Native trial',e.native_trial_id],['Source split',e.source_split],['Curation reason',e.curation_reason],['Covered variants',e.covered_variants],['Float SHA256',e.float_sha256]])}<pre>${esc(JSON.stringify(e.metadata,null,2))}</pre></div></details>` : '<p class="metadata-withheld">Current-trial technical metadata is withheld while answers are hidden.</p>'}</section></div>`;
  }
  function transition(state, action, value) {
    const next = {...state};
    if (action === 'try' || (action === 'new-example' && state.trying)) Object.assign(next,{trying:true,revealed:false,answers:false,explanation:false,response:undefined});
    else if (action === 'cancel-judgment') Object.assign(next,{trying:false,revealed:false,answers:false,explanation:false,response:undefined});
    else if (action === 'new-example') Object.assign(next,{revealed:false,response:undefined});
    else if (action === 'judgment' || action === 'reveal') Object.assign(next,{revealed:true,answers:true,explanation:true,response:action === 'judgment' ? Number(value) : undefined});
    else if (action === 'answers' && !(state.trying && !state.revealed)) next.answers = !state.answers;
    else if (action === 'observer') next.explanation = false;
    else if (action === 'explanation' && !(state.trying && !state.revealed)) { next.explanation = true; next.answers = true; }
    return next;
  }
  function boot(data, document) {
    const app = document.getElementById('app');
    if (!data?.manifest?.episodes?.length || !data?.tasks?.length) {
      app.innerHTML = '<section class="load-error"><h1>The local example bank is missing.</h1><p>Open the built atlas folder with its data.js and assets alongside index.html. The authored web template is not a standalone build.</p></section>'; return;
    }
    let state = {answers:false,explanation:false,trying:false,revealed:false,full:false,rate:1};
    let route, observer, nodes = new Map(), sectionStates = new Map();
    const scheduler = new Scheduler({reducedMotion:root.matchMedia?.('(prefers-reduced-motion: reduce)').matches || false});
    const cache = new Map();
    function getImage(path) {
      if (cache.has(path)) { const entry = cache.get(path); cache.delete(path); cache.set(path,entry); return entry; }
      const image = new root.Image(); const entry = {image,ready:false,error:false}; cache.set(path,entry);
      image.onload = () => { entry.ready = true; nodes.forEach((player,node) => { if (player.episode.frames[player.index] === path) paint(node,player); }); };
      image.onerror = () => { entry.error = true; nodes.forEach((player,node) => { if (player.episode.frames[player.index] === path || player.episode.frames[(player.index+1)%player.episode.frame_count] === path) { player.pause(); const status=node.querySelector('[data-loop]'); status.textContent='Frame missing — check local assets'; status.classList.add('scene-error'); } }); };
      image.src = path;
      if (image.complete && image.naturalWidth) entry.ready = true;
      if (cache.size > 256) cache.delete(cache.keys().next().value);
      return entry;
    }
    function paint(node, player) {
      if (!node.isConnected) return;
      const image = node.querySelector('[data-scene]'), frame = player.episode.frames[player.index];
      // Loading/decode never creates intermediate or reconstructed evidence.
      if (player.visible) {
        const current = getImage(frame); image.style.visibility = current.ready ? 'visible' : 'hidden';
        if (current.ready && image.getAttribute('src') !== frame) image.src = frame;
        image.alt = `Native stimulus, frame ${player.index}; no annotations`;
        getImage(player.episode.frames[(player.index + 1) % player.episode.frame_count]);
      }
      node.querySelector('[data-counter]').textContent = `Frame ${player.index} / ${player.episode.frame_count - 1}`;
      const button=node.querySelector('[data-action="toggle-player"]'); button.textContent=player.playing ? 'Ⅱ Pause' : '▶ Play'; button.setAttribute('aria-pressed',String(player.playing));
      const loop=node.querySelector('[data-loop]'); loop.textContent=player.index === 0 && player.loops ? `↺ Trial restart · pass ${player.loops+1}` : `Native trial · pass ${player.loops+1}`;
      const slider=node.querySelector('[data-frame]'); if (slider) slider.value=player.index;
      const phases=phaseGroups(player.episode,canExplain(state)), phase=phases.find(p=>player.index>=p.start && player.index<=p.end);
      const phaseNode=node.querySelector('[data-phase]'); if(phaseNode) phaseNode.textContent=words(phase?.phase);
      const cue=node.querySelector('[data-cue]'); if(cue) cue.textContent=phase?.cue ? 'Cue visible' : 'Cue absent';
      node.querySelectorAll('[data-seek]').forEach(button=>button.setAttribute('aria-current',String(player.index>=Number(button.dataset.seek)&&player.index<=Number(button.dataset.end))));
      if (!node.classList.contains('detail-player') || !canExplain(state)) return;
      const analysis=app.querySelector('[data-analysis]'); if(!analysis) return;
      analysis.querySelectorAll('[data-analysis-cue]').forEach(el=>el.textContent=phase?.cue ? 'Native cue visible now' : 'Native cue absent now');
      if(player.episode.task_id==='motion_duration_cued') {
        const evidence=durationEvidence(player.episode,player.index);
        analysis.querySelector('[data-transition-count]').textContent=evidence.transitions;
        evidence.counts.forEach((count,i)=>{analysis.querySelector(`[data-direction-meter="${i}"]`).value=count; analysis.querySelector(`[data-direction-count="${i}"]`).textContent=count;});
      }
      if(player.episode.task_id==='image_recognition') analysis.querySelector('[data-probe-counter]').textContent=(player.episode.metadata.probe_frames || []).filter(i=>i<=player.index).length;
      analysis.querySelectorAll('.ab-filmstrip figure').forEach((figure,i)=>figure.setAttribute('aria-current',String(i===player.index)));
    }
    function updateGlobals() {
      const answers=document.querySelector('[data-global="answers"]');
      answers.textContent=state.answers ? 'Hide answers' : 'Show answers'; answers.setAttribute('aria-pressed',String(state.answers)); answers.disabled=Boolean(state.trying&&!state.revealed);
      document.querySelector('[data-playback-status]').textContent=scheduler.playing ? 'Visible scenes playing · illustration speed' : 'Paused · every frame remains inspectable';
      document.querySelector('[data-global-rate]').value=state.rate;
      document.querySelectorAll('[data-nav]').forEach(a=>a.setAttribute('aria-current',(a.dataset.nav==='conditions')===(route?.view==='conditions')?'page':'false'));
    }
    function render({preserve=true, restart=false} = {}) {
      const snapshots=new Map();
      if(preserve) nodes.forEach(player=>snapshots.set(player.episode.id,{index:player.index,loops:player.loops,playing:player.playing,rate:player.rate}));
      app.querySelectorAll('details[data-section]').forEach(el=>sectionStates.set(el.dataset.section,el.open));
      observer?.disconnect(); nodes.clear(); scheduler.clear();
      app.innerHTML=route.view==='detail' ? renderDetail(data,route,state) : route.view==='not-found' ? '<section class="load-error"><h1>This atlas address is not in the native inventory.</h1><p>No substitute condition or episode has been selected.</p><a href="#explore">Return to the atlas →</a></section>' : renderGallery(data,state,route.view==='conditions');
      if(preserve&&!route.reader) app.querySelectorAll('details[data-section]').forEach(el=>{if(sectionStates.has(el.dataset.section))el.open=sectionStates.get(el.dataset.section);});
      observer = root.IntersectionObserver ? new root.IntersectionObserver(entries=>entries.forEach(entry=>{const player=nodes.get(entry.target); if(player){player.setVisible(entry.isIntersecting);if(entry.isIntersecting)paint(entry.target,player);}}),{rootMargin:'0px',threshold:0}) : null;
      app.querySelectorAll('[data-player]').forEach(node=>{
        const e=episodeById(data,node.dataset.player);
        const player=new Player(e,{onChange:p=>paint(node,p),isReady:i=>getImage(e.frames[i]).ready});
        player.visible=false; nodes.set(node,player); scheduler.add(player);
        const snapshot=snapshots.get(e.id); if(snapshot&&!restart){player.index=snapshot.index;player.loops=snapshot.loops;player.playing=snapshot.playing;player.setRate(snapshot.rate);}
        if(restart){player.index=0;player.loops=0;player.playing=true;}
        paint(node,player); if(observer)observer.observe(node);else{player.setVisible(true);paint(node,player);}
      });
      document.title=route.task ? `${route.task.title} · Visual Task Atlas` : `${route.view==='conditions'?'All conditions':'Explore'} · Visual Task Atlas`;
      updateGlobals();
    }
    function navigateEpisode(episode) { root.location.hash=detailLink(episode,route.reader); }
    function nextExample(delta) { const ids=route.cell.episode_ids; const index=ids.indexOf(route.episode.id); navigateEpisode(episodeById(data,ids[(index+delta+ids.length)%ids.length])); }
    app.addEventListener('click',event=>{
      const ref=event.target.closest('[data-reference]');
      if(ref){event.preventDefault(); const target=document.getElementById('ref-'+ref.dataset.reference); if(target){target.closest('details').open=true;target.scrollIntoView({block:'start'});}return;}
      const button=event.target.closest('button'); if(!button||button.disabled)return;
      const node=button.closest('[data-player]'), player=nodes.get(node), action=button.dataset.action;
      if(button.hasAttribute('data-seek')){player.pause();player.seek(button.dataset.seek);return;}
      if(button.hasAttribute('data-judgment')){state=transition(state,'judgment',button.dataset.judgment);render();return;}
      if(action==='toggle-player'){player.playing?player.pause():player.play();return;}
      if(action==='replay'){player.replay();return;}
      if(action==='previous-frame'||action==='next-frame'){player.step(action==='next-frame'?1:-1);return;}
      if(action==='next-example'||action==='previous-example'){nextExample(action==='next-example'?1:-1);return;}
      if(action==='expand'||action==='overview'){state.full=action==='expand'; app.querySelectorAll('details[data-section]').forEach(el=>el.open=state.full);return;}
      if(action==='print'){app.querySelectorAll('details[data-section]').forEach(el=>el.open=true);root.print();return;}
      if(['try','reveal','observer','explanation','cancel-judgment'].includes(action)){state=transition(state,action);render({restart:action==='try'});}
    });
    app.addEventListener('input',event=>{if(event.target.matches('[data-frame]')){const index=event.target.value;const player=nodes.get(event.target.closest('[data-player]'));player.pause();player.seek(index);}});
    app.addEventListener('change',event=>{
      const el=event.target;
      if(el.matches('[data-condition]')){const cell=taskCells(data,route.task.id).find(c=>c.condition_id===el.value);navigateEpisode(episodeById(data,cell.showcase_id));}
      if(el.matches('[data-example]'))navigateEpisode(episodeById(data,route.cell.episode_ids[Number(el.value)]));
      if(el.matches('[data-variant]')&&el.value){const episode=route.cell.episode_ids.map(id=>episodeById(data,id)).find(e=>(e.covered_variants||[]).includes(el.value));if(episode)navigateEpisode(episode);}
      if(el.matches('[data-local-rate]')){state.rate=Number(el.value);scheduler.setRate(state.rate);updateGlobals();}
    });
    document.querySelector('.global-controls').addEventListener('click',event=>{
      const button=event.target.closest('[data-global]');if(!button||button.disabled)return;
      const action=button.dataset.global;
      if(action==='play-all')scheduler.playAll();
      if(action==='pause-all')scheduler.pauseAll();
      if(action==='answers'){state=transition(state,'answers');render();}
      updateGlobals();
    });
    document.querySelector('[data-global-rate]').addEventListener('change',event=>{state.rate=Number(event.target.value);scheduler.setRate(state.rate);const local=app.querySelector('[data-local-rate]');if(local)local.value=state.rate;});
    document.addEventListener('visibilitychange',()=>scheduler.setHidden(document.hidden));
    document.addEventListener('keydown',event=>{
      if(route.view!=='detail'||event.ctrlKey||event.metaKey||event.altKey||/INPUT|SELECT|TEXTAREA|BUTTON|A|SUMMARY/.test(event.target.tagName))return;
      const player=nodes.values().next().value;if(!player)return;
      if(event.code==='Space'){event.preventDefault();player.playing?player.pause():player.play();}
      if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();player.step(event.key==='ArrowRight'?1:-1);}
      if(event.key==='Home'){event.preventDefault();player.pause();player.seek(0);}
      if(event.key==='End'){event.preventDefault();player.pause();player.seek(player.episode.frame_count-1);}
    });
    root.addEventListener('hashchange',()=>{
      if(root.location.hash==='#app'){app.focus();return;}
      const next=resolveRoute(root.location.hash,data), same=route?.episode?.id&&route.episode.id===next.episode?.id;
      if(!same){state=transition(state,'new-example');sectionStates.clear();}
      if(next.view!=='detail'&&state.trying)state=transition(state,'cancel-judgment');
      route=next;render({preserve:Boolean(same)});if(!same)root.scrollTo(0,0);
    });
    root.addEventListener('beforeprint',()=>app.querySelectorAll('details[data-section]').forEach(el=>el.open=true));
    root.addEventListener('pagehide',()=>scheduler.setHidden(true));
    root.addEventListener('pageshow',()=>scheduler.setHidden(document.hidden));
    route=resolveRoute(root.location.hash,data);render({preserve:false});scheduler.setHidden(document.hidden);scheduler.start();
    return {scheduler,getState:()=>({...state}),getRoute:()=>route};
  }
  const api = {Player, Scheduler, resolveRoute, detailLink, phaseGroups, durationEvidence, renderGallery, renderDetail, renderAnalysis, canReveal, canExplain, transition, variantLabel, boot};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.AtlasUI = api;
  if (root.document?.getElementById('app')) root.atlasApplication = boot(root.ATLAS,root.document);
})(typeof window !== 'undefined' ? window : globalThis);
