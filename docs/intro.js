'use strict';
(() => {
  const panel = document.getElementById('intro-overlay');
  const composer = document.getElementById('intro-composer');
  const typed = document.getElementById('intro-typed');
  const messages = document.getElementById('intro-messages');
  const thinking = document.getElementById('intro-thinking');
  const result = document.getElementById('intro-result');
  const resultText = document.getElementById('intro-result-text');
  const pause = document.getElementById('intro-pause');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const title = 'EchoNav: Evidential BEV Completion from Historical Observations for Visual Navigation';
  const questions = ['Can robots remember the paths they have traveled?', 'Can robots predict the space behind obstacles?'];
  let run = 0;
  let paused = false;
  const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
  // Size against the complete title so its single line never jumps while typing.
  const measure = document.createElement('canvas').getContext('2d');
  function fitTitle() {
    const style = getComputedStyle(resultText);
    measure.font = `${style.fontWeight} 22px ${style.fontFamily}`;
    const available = Math.min(1000, panel.clientWidth - (innerWidth <= 700 ? 32 : 48)) - 42;
    resultText.style.fontSize = Math.min(22, available / measure.measureText(title).width * 22) + 'px';
  }
  new ResizeObserver(fitTitle).observe(panel);
  async function type(text, target, delay, token) {
    target.textContent = '';
    for (const char of text) {
      while ((paused || document.hidden) && token === run) await wait(100);
      if (token !== run) return false;
      target.textContent += char;
      await wait(delay);
    }
    return token === run;
  }
  async function hold(ms, token) {
    for (let remaining = ms; remaining > 0;) {
      if (token !== run) return false;
      await wait(100);
      if (!paused && !document.hidden) remaining -= 100;
    }
    return token === run;
  }
  async function start() {
    const token = ++run;
    if (reduced.matches) {
      messages.replaceChildren(); composer.hidden = thinking.hidden = true;
      result.hidden = false; resultText.textContent = title; pause.hidden = true;
      fitTitle(); return;
    }
    pause.hidden = false;
    while (token === run) {
      messages.replaceChildren(); typed.textContent = ''; resultText.textContent = '';
      composer.hidden = false; thinking.hidden = result.hidden = true;
      for (const question of questions) {
        if (!await type(question, typed, 38, token)) return;
        composer.classList.add('is-sending');
        if (!await hold(220, token)) return;
        const bubble = document.createElement('div');
        bubble.className = 'intro-bubble'; bubble.textContent = question;
        messages.append(bubble); typed.textContent = '';
        composer.classList.remove('is-sending');
        if (!await hold(400, token)) return;
      }
      composer.hidden = true; thinking.hidden = false;
      if (!await hold(3000, token)) return;
      thinking.hidden = true; result.hidden = false;
      fitTitle();
      if (!await type(title, resultText, 24, token)) return;
      if (!await hold(4000, token)) return;
    }
  }
  pause.addEventListener('click', () => {
    paused = !paused;
    panel.classList.toggle('is-paused', paused);
    pause.textContent = paused ? 'Resume animation' : 'Pause animation';
    pause.setAttribute('aria-pressed', String(paused));
  });
  reduced.addEventListener('change', start);
  start();
})();
