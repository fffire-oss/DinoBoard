export function createModal() {
  const backdrop = document.createElement('div');
  backdrop.className = 'general-modal-backdrop hidden';
  backdrop.innerHTML = `
    <div class="general-modal-panel">
      <div class="general-modal-title"></div>
      <div class="general-modal-text"></div>
      <div style="display:flex; gap:8px; justify-content:flex-end; flex-wrap:wrap;">
        <button class="general-modal-ok-btn" data-modal="replay" data-mobile-hide style="background:#0ea5e9;">查看录像</button>
        <button class="general-modal-ok-btn" data-modal="restart" style="background:#16a34a;">再来一局</button>
        <button class="general-modal-ok-btn" data-modal="ok">确定</button>
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);

  const titleEl = backdrop.querySelector('.general-modal-title');
  const textEl = backdrop.querySelector('.general-modal-text');
  const replayBtn = backdrop.querySelector('[data-modal="replay"]');
  const restartBtn = backdrop.querySelector('[data-modal="restart"]');
  let onReplay = null;
  let onRestart = null;

  backdrop.querySelector('[data-modal="ok"]').addEventListener('click', () => {
    backdrop.classList.add('hidden');
  });
  replayBtn.addEventListener('click', () => {
    backdrop.classList.add('hidden');
    if (onReplay) onReplay();
  });
  restartBtn.addEventListener('click', () => {
    backdrop.classList.add('hidden');
    if (onRestart) onRestart();
  });

  return {
    show(title, text, showReplay = true) {
      titleEl.textContent = title;
      textEl.textContent = text;
      replayBtn.style.display = showReplay ? '' : 'none';
      backdrop.classList.remove('hidden');
    },
    hide() { backdrop.classList.add('hidden'); },
    onReplay(fn) { onReplay = fn; },
    onRestart(fn) { onRestart = fn; },
  };
}
