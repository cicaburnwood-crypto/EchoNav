'use strict';
const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const menu = $('.menu-toggle');
menu.addEventListener('click', () => {
  const open = menu.getAttribute('aria-expanded') !== 'true';
  menu.setAttribute('aria-expanded', String(open));
  menu.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
  $('#main-nav').classList.toggle('open', open);
});
$$('#main-nav a').forEach(link => link.addEventListener('click', () => {
  menu.setAttribute('aria-expanded', 'false');
  menu.setAttribute('aria-label', 'Open navigation');
  $('#main-nav').classList.remove('open');
}));
const benchmarks = {
  sequential: { label:'SIMULATION · SUCCESS RATE', title:'Reach both goals, in sequence.', description:'1,000 shared routes across 308 held-out scenes. The second destination is revealed only after the first is reached.', delta:'+32.2', deltaLabel:'percentage points\nover OccAnt-RGBD', unit:'Success rate (%) · higher is better', rows:[['EchoNav',86.3,true],['OccAnt-RGBD',54.1,false]], note:'The strongest evaluated baseline is shown here. Success requires reaching both destinations in order without collision. Full comparisons: Section IV-D.' },
  physical: { label:'PHYSICAL ROBOT · SUCCESS RATE', title:'Navigate real indoor scenes.', description:'60 trials per method in 20 scenes, with three matched trials per scene. Success requires reaching within 30 cm of the goal.', delta:'+40.0', deltaLabel:'percentage points\nover X-NavDP', unit:'Success rate (%) · higher is better', rows:[['EchoNav',73.3,true],['X-NavDP',33.3,false],['OmniVLA',26.7,false]], note:'Quantitative evaluation uses a Mecanum-wheeled robot. A trial fails on collision or when the planner cannot find a feasible path. Source: Section IV-E.' },
  fusion: { label:'ABLATION · OVERALL mIoU', title:'Select history, cell by cell.', description:'190 examples from held-out HM3D and ProcTHOR scenes. All variants use the same checkpoint and produce dense predictions.', delta:'+10.84', deltaLabel:'mIoU percentage points\nover no history fusion', unit:'Overall mIoU (%) · higher is better', rows:[['Full selective fusion',58.43,true],['Equal-weight fusion',56.53,false],['No history fusion',47.59,false]], note:'The anchor-only baseline still retains historical context from joint encoding. Full fusion also reaches 48.95% Recovery mIoU and 50.26% Hidden mIoU. Source: Table II.' }
};
function showBenchmark(key, focus = false) {
  const data = benchmarks[key];
  if (!data) return;
  const labels = {'result-label':'label','result-title':'title','result-description':'description','result-delta':'delta','result-delta-label':'deltaLabel','chart-unit':'unit','chart-note':'note'};
  Object.entries(labels).forEach(([id, field]) => { document.getElementById(id).textContent = data[field]; });
  $('#result-delta-label').style.whiteSpace = 'pre-line';
  const bars = $('#chart-bars'); bars.replaceChildren();
  data.rows.forEach(([name,value,ours]) => {
    const row = document.createElement('div'); row.className = 'bar-row' + (ours ? ' ours' : '');
    const label = document.createElement('div'); label.className = 'bar-label';
    const title = document.createElement('span'); title.textContent = name;
    const number = document.createElement('strong'); number.textContent = value.toFixed(key === 'fusion' ? 2 : 1) + '%';
    label.append(title, number);
    const track = document.createElement('div'); track.className = 'bar-track'; track.setAttribute('aria-hidden','true');
    const fill = document.createElement('span'); fill.style.width = value + '%'; track.append(fill); row.append(label, track); bars.append(row);
  });
  $$('.result-tabs button').forEach(tab => { const active = tab.dataset.benchmark === key; tab.setAttribute('aria-selected',String(active)); tab.tabIndex = active ? 0 : -1; if (active && focus) tab.focus(); });
  $('#result-panel').setAttribute('aria-labelledby','tab-' + key);
}
const tabs = $$('.result-tabs button');
tabs.forEach((tab,i) => {
  tab.addEventListener('click', () => showBenchmark(tab.dataset.benchmark));
  tab.addEventListener('keydown', event => {
    let target = i;
    if (event.key === 'ArrowRight') target = (i + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') target = (i + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') target = 0;
    else if (event.key === 'End') target = tabs.length - 1;
    else return;
    event.preventDefault(); showBenchmark(tabs[target].dataset.benchmark, true);
  });
});
const dialog = $('#figure-dialog'); let lastFigureButton;
$$('[data-zoom]').forEach(button => button.addEventListener('click', () => {
  lastFigureButton = button;
  $('#enlarged-figure').src = button.dataset.zoom;
  $('#enlarged-figure').alt = button.querySelector('img').alt;
  $('#figure-caption').textContent = button.dataset.caption;
  dialog.showModal(); document.body.classList.add('dialog-open');
}));
$('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) { const r=dialog.getBoundingClientRect(); if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom) dialog.close(); } });
dialog.addEventListener('close', () => { document.body.classList.remove('dialog-open'); lastFigureButton?.focus(); });
$('#copy-citation').addEventListener('click', async () => {
  const text = $('#citation').textContent;
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else { const field=document.createElement('textarea'); field.value=text; field.style.position='fixed'; field.style.left='-10000px'; document.body.append(field); field.select(); const ok=document.execCommand('copy'); field.remove(); if(!ok) throw new Error('Copy unavailable'); }
    $('#copy-status').textContent='Citation copied.';
  } catch (_) { $('#copy-status').textContent='Select the citation above to copy it.'; }
});
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => { if (entry.isIntersecting) $$('#main-nav a[href^="#"]').forEach(link => link.classList.toggle('current',link.hash === '#' + entry.target.id)); });
  }, { rootMargin:'-15% 0px -65% 0px', threshold:0 });
  ['overview','method','comparisons','humanoid-trials','quadruped-trials','results'].forEach(id => observer.observe(document.getElementById(id)));
}

// A scene always has exactly one trial for each method.
const sceneSelect = $('#scene-select');
const trialMethods = ['ours', 'xdp', 'omnivla'];
const trialPlayers = trialMethods.map(method => $('#trial-' + method));
const trialScenes = window.ECHONAV_SCENES || [];
let sceneGeneration = 0;
let groupPlaying = false;
const trialPlayButton = $('#play-trials');
function stopTrials() {
  groupPlaying = false;
  trialPlayers.forEach(player => player.pause());
  trialPlayButton.textContent = 'Play all';
}
function chooseScene(index) {
  sceneGeneration++;
  stopTrials();
  const scene = trialScenes[index];
  if (!scene) return;
  $('#scene-description').textContent = scene.description;
  trialPlayers.forEach((player, i) => {
    const clip = scene.videos[trialMethods[i]];
    const speedBadge = document.getElementById('speed-' + trialMethods[i]);
    speedBadge.textContent = (clip.speed ? clip.speed.toFixed(2) : '1') + '× speed';
    speedBadge.title = clip.originalDuration ? `Original: ${clip.originalDuration.toFixed(2)} s; shown in 30 s` : 'Original recording speed';
    const badge = document.getElementById('outcome-' + trialMethods[i]);
    badge.textContent = clip.outcome === 'success' ? 'Success' : 'Failure';
    badge.className = 'trial-outcome ' + clip.outcome;
    badge.setAttribute('aria-label', ['EchoNav','X-NavDP','OmniVLA'][i] + ' outcome: ' + badge.textContent);
    player.src = clip.src;
    player.poster = clip.poster;
    player.playbackRate = 1;
    player.setAttribute('aria-label', `${['EchoNav','X-NavDP','OmniVLA'][i]} — ${scene.title}`);
    player.load();
  });
  $('#trial-status').textContent = `${scene.label} · Ready to play three trials.`;
}
trialScenes.forEach((scene, i) => {
  const option = document.createElement('option');
  option.value = i; option.textContent = `${scene.label} — ${scene.title}`;
  sceneSelect.append(option);
});
sceneSelect.addEventListener('change', () => chooseScene(Number(sceneSelect.value)));
function readyToPlay(player) {
  if (player.readyState >= 3) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const finish = (error) => { clearTimeout(timer); player.removeEventListener('canplay',ok); player.removeEventListener('error',fail); error ? reject(error) : resolve(); };
    const ok = () => finish();
    const fail = () => finish(new Error('Video could not load'));
    const timer = setTimeout(() => finish(new Error('Loading timed out')),30000);
    player.addEventListener('canplay',ok,{once:true}); player.addEventListener('error',fail,{once:true});
    player.preload = 'auto';
  });
}
trialPlayButton.addEventListener('click', async () => {
  if (groupPlaying) { stopTrials(); $('#trial-status').textContent = 'Trials paused.'; return; }
  const generation = sceneGeneration;
  trialPlayButton.disabled = true;
  $('#trial-status').textContent = 'Loading all three trials…';
  try {
    await Promise.all(trialPlayers.map(readyToPlay));
    if (generation !== sceneGeneration) return;
    // Restart a finished player; otherwise resume each trial from its paused position.
    trialPlayers.forEach(player => { if (player.ended) player.currentTime = 0; });
    document.querySelectorAll('.platform-player').forEach(player => player.pause());
    await Promise.all(trialPlayers.map(player => player.play()));
    if (generation !== sceneGeneration) { stopTrials(); return; }
    groupPlaying = true; trialPlayButton.textContent = 'Pause all';
    $('#trial-status').textContent = 'Playing three trials. Each stops at its own end.';
  } catch (_) {
    if (generation === sceneGeneration) { stopTrials(); $('#trial-status').textContent = 'Could not start all trials. Use the individual video controls or try again.'; }
  } finally { trialPlayButton.disabled = false; }
});
$('#restart-trials').addEventListener('click', () => {
  sceneGeneration++; stopTrials();
  trialPlayers.forEach(player => { if (player.readyState >= 1) player.currentTime = 0; });
  $('#trial-status').textContent = 'All trials reset. Press Play all to start together.';
});
trialPlayers.forEach(player => player.addEventListener('ended', () => {
  if (trialPlayers.every(p => p.ended || p.paused)) { groupPlaying=false;trialPlayButton.textContent='Play all';$('#trial-status').textContent='Playback finished.'; }
}));

if (trialScenes.length) chooseScene(0);

const platformCases = {
 humanoid: {
  glass: {file:'humanoid-glass', description:'Glass-door navigation: detect the door and infer the free space behind it.'},
  elevator: {file:'humanoid-elevator', description:'Exit the elevator and dynamically avoid a person.'}
 },
 quadruped: {
  memory: {file:'quadruped-memory-30s', poster:'quadruped-memory', speed:1.2288889, description:'Use historical observations to explore alternative routes around indoor obstacles.'}
 }
};
Object.entries(platformCases).forEach(([platform,cases]) => {
 const player = document.getElementById(platform+'-video');
 const select = document.getElementById(platform+'-select');
 const play = document.getElementById(platform+'-play');
 const status = document.getElementById(platform+'-status');
 function loadCase() {
  player.pause(); const chosen=cases[select.value];
  player.src='assets/platforms/'+chosen.file+'.mp4';
  player.poster='assets/platforms/'+(chosen.poster || chosen.file)+'.jpg';
  player.playbackRate=1;
  document.getElementById(platform+'-speed').textContent=(chosen.speed ? chosen.speed.toFixed(2) : '1')+'× speed';
  player.load(); play.textContent='Play';
  document.getElementById(platform+'-description').textContent=chosen.description;
  status.textContent='Ready to play.';
 }
 select.addEventListener('change',loadCase);
 play.addEventListener('click',async()=>{
  if(!player.paused){player.pause();return;}
  try {await player.play();}catch(_){status.textContent='Use the video controls to start playback.';}
 });
 document.getElementById(platform+'-restart').addEventListener('click',()=>{
  player.pause();if(player.readyState>=1)player.currentTime=0;status.textContent='Reset. Press Play to start.';
 });
 player.addEventListener('play',()=>{
  stopTrials();
  document.querySelectorAll('.platform-player').forEach(other=>{if(other!==player)other.pause();});
  play.textContent='Pause';status.textContent='Playing: '+select.selectedOptions[0].textContent;
 });
 player.addEventListener('pause',()=>{play.textContent='Play';status.textContent=player.ended?'Playback finished.':'Paused.';});
 player.addEventListener('ended',()=>{play.textContent='Play';status.textContent='Playback finished.';});
 player.addEventListener('error',()=>{status.textContent='Video could not load. Please reload the page.';});
 loadCase();
});
