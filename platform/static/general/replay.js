import { apiGet, API_BASE } from './api.js';

export function createReplayController(infoCol, config) {
  const panel = document.createElement('div');
  panel.className = 'analysis-panel';
  panel.hidden = true;
  panel.innerHTML = `
    <div class="analysis-controls">
      <button class="replay-btn" data-action="first" type="button">回到最初</button>
      <button class="replay-btn" data-action="play" type="button">播放</button>
      <button class="replay-btn" data-action="prev" type="button">上一步</button>
      <button class="replay-btn" data-action="next" type="button">下一步</button>
      <button class="replay-btn" data-action="warn" type="button">跳到失误</button>
      <button class="replay-btn" data-action="blunder" type="button">跳到严重失误</button>
    </div>
    <div class="analysis-step-card" id="analysis-step">--</div>
    <div class="analysis-list" id="analysis-list"></div>
  `;
  infoCol.appendChild(panel);

  let frames = null;
  let step = 0;
  let lastRenderedStep = -1;
  let playing = false;
  let timer = null;
  let onRenderFrame = null;
  let onExit = null;
  let resolveActor = null;
  // When alwaysVisible is true, the panel stays mounted even without a
  // loaded replay — controls are disabled and placeholder text is shown.
  // This is a dev-debug convenience: styles for this panel used to be
  // only visible after finishing a full game and opening its replay.
  let alwaysVisible = false;

  const stepEl = panel.querySelector('#analysis-step');
  const listEl = panel.querySelector('#analysis-list');
  const btns = {};
  panel.querySelectorAll('[data-action]').forEach(b => { btns[b.dataset.action] = b; });

  function stopTimer() {
    if (timer) { clearInterval(timer); timer = null; }
    playing = false;
  }

  // Build the 3 fixed lines used both by the current-frame card and each
  // list item. Missing analysis/suggestion slots render as "—" so every
  // card has the same line count — the CSS height is then a simple
  // constant and doesn't jitter between frames.
  //   Line 1: 帧数-玩家-行动（合并在一行）
  //   Line 2: 胜率-掉点
  //   Line 3: 智能推荐行动
  function frameLines(frame) {
    const total = frames.length;
    const idx = frames.indexOf(frame);

    if (frame.__terminal_display) {
      return {
        line1: '帧 ' + (idx + 1) + '/' + total + ' · 终局画面',
        line2: '—',
        line3: '—',
      };
    }

    const actorName = resolveActor ? resolveActor(frame.actor) : frame.actor;
    const actor = frame.actor === 'start' ? '' : actorName;

    // frame.actor is "player_<N>" or "start"; parse the numeric seat for
    // games whose formatMove needs actor (e.g. Love Letter self-target
    // no-op fallback detection).
    const actorIdx = (typeof frame.actor === 'string' && frame.actor.startsWith('player_'))
      ? parseInt(frame.actor.slice('player_'.length), 10) : null;
    const moveText = config.formatOpponentMove
      ? config.formatOpponentMove(frame.action_info, frame.action_id, actorIdx)
      : (frame.action_id !== null && frame.action_id !== undefined ? '动作 ' + frame.action_id : '开局');
    const move = frame.actor === 'start' ? '开局' : moveText;

    // tail-solve flag lives on ai_stats of the AI frame. We use it as a
    // line-1 badge on the AI frame ("AI 这步用了 tail solver"), and we
    // *propagate the verdict* to line 2 of the IMMEDIATELY PRECEDING human
    // frame — the human's move set up the proven losing position, so
    // "胜率 X%（残局已求解）" belongs on their analysis row. The AI frame
    // itself doesn't need a line-2 winrate; it's not an interesting datum
    // ("AI proved itself wins" is what the badge already says), and a "0%"
    // there read confusingly as "AI's winrate is 0%". Keep AI line-2 as "—".
    const aiStats = frame.ai_stats || null;
    const tailSolved = !!(aiStats && aiStats.tail_solved);

    // Lookahead: if the next frame is an AI tail-solve adoption, surface
    // the verdict on this (human) frame. This is the only place we need
    // the next-frame info; computing it once here keeps the rule local.
    const nextFrame = (idx + 1 < frames.length) ? frames[idx + 1] : null;
    const nextAiStats = nextFrame ? (nextFrame.ai_stats || null) : null;
    const nextTailSolved = !!(nextAiStats && nextAiStats.tail_solved);
    const nextTailOutcome = nextAiStats ? nextAiStats.tail_solve_outcome : 0;

    const parts1 = ['帧 ' + (idx + 1) + '/' + total];
    if (actor) parts1.push(actor);
    parts1.push(frame.is_terminal ? move + ' · 终局' : move);
    if (tailSolved) parts1.push('残局已求解');
    const line1 = parts1.join(' · ');

    const a = frame.analysis;
    let line2 = '—';
    let line3 = '—';
    if (nextTailSolved && !tailSolved) {
      // Human frame preceding an AI tail-solve adoption: the AI proved a
      // forced outcome from this position. Translate the AI-POV outcome
      // into the human's win rate so the column reads consistently with
      // pipeline-derived analysis rows.
      let humanWrPct;
      let suffix;
      if (nextTailOutcome === 1) {
        // ProvenWin (AI) → human loses. AI only adopts on this branch in
        // the live path, so this is the dominant case.
        humanWrPct = '0.0%';
        suffix = '残局已求解';
      } else if (nextTailOutcome === 2) {
        humanWrPct = '100.0%';
        suffix = '残局已求解';
      } else if (nextTailOutcome === 3) {
        humanWrPct = '50.0%';
        suffix = '残局已求解：平局';
      } else {
        humanWrPct = '--';
        suffix = '残局已求解';
      }
      line2 = '胜率 ' + humanWrPct + '（' + suffix + '）';
    } else if (a) {
      const wr = (a.best_win_rate * 100).toFixed(1) + '%';
      const drop = a.drop_score.toFixed(1) + '%';
      line2 = '胜率 ' + wr + ' · 掉点 ' + drop;
      if (frame.action_id !== a.best_action && a.best_action_info) {
        const bestText = config.formatSuggestedMove
          ? config.formatSuggestedMove(a.best_action_info, a.best_action, actorIdx)
          : '动作 ' + a.best_action;
        line3 = '推荐 ' + bestText;
      } else if (frame.action_id === a.best_action) {
        line3 = '（最优着法）';
      }
    }
    return { line1, line2, line3 };
  }

  function renderPanel() {
    if (!frames || !frames.length) return;
    const frame = frames[step];
    const total = frames.length;

    const lines = frameLines(frame);
    stepEl.textContent = lines.line1 + '\n' + lines.line2 + '\n' + lines.line3;

    btns.first.disabled = step <= 0;
    btns.prev.disabled = step <= 0;
    btns.next.disabled = step >= total - 1;
    // Re-enable play — if the empty-state (alwaysVisible, no replay) ran
    // earlier it would have disabled every button, and only first/prev/
    // next/warn/blunder get re-enabled below. Without this, play stays
    // disabled after loading a replay file.
    btns.play.disabled = false;
    btns.play.textContent = playing ? '暂停' : '播放';

    let hasBlunder = false, hasWarn = false;
    for (const f of frames) {
      if (f.analysis) {
        if (f.analysis.drop_score >= 10) hasBlunder = true;
        if (f.analysis.drop_score >= 5) hasWarn = true;
      }
    }
    btns.blunder.disabled = !hasBlunder;
    btns.warn.disabled = !hasWarn;

    listEl.innerHTML = '';
    for (let i = 0; i < frames.length; i++) {
      const f = frames[i];
      const item = document.createElement('div');
      item.className = 'analysis-item' + (i === step ? ' current' : '');

      const a = f.analysis;
      if (a) {
        const sev = a.drop_score >= 10 ? 'blunder' : (a.drop_score >= 5 ? 'warn' : '');
        if (sev) item.classList.add(sev);
      }

      const ln = frameLines(f);
      // Always 3 rows — missing stats/suggest slots show "—". Keeps every
      // item the same height so the visible window is always exactly
      // 2 items tall.
      item.innerHTML =
        '<div>' + ln.line1 + '</div>' +
        '<div>' + ln.line2 + '</div>' +
        '<div>' + ln.line3 + '</div>';

      item.addEventListener('click', () => {
        stopTimer();
        step = i;
        renderPanel();
      });
      listEl.appendChild(item);
    }

    const currentItem = listEl.querySelector('.current');
    if (currentItem) currentItem.scrollIntoView({ block: 'nearest', behavior: 'smooth' });

    if (onRenderFrame) {
      // Pass prev frame for the step-transition so the board can animate
      // (fly tiles between prev → current frame). Sequential-step changes
      // animate naturally; jumps (e.g. click on a distant item) pass the
      // last-rendered step too — animation will just look instant but
      // the rendered frame is still correct.
      const prevFrame = (lastRenderedStep >= 0 && lastRenderedStep < frames.length)
          ? frames[lastRenderedStep] : null;
      onRenderFrame(frames[step], frames, prevFrame);
      lastRenderedStep = step;
    }
  }

  function jumpToSeverity(minDrop) {
    if (!frames) return;
    const start = step + 1;
    const len = frames.length;
    for (let offset = 0; offset < len; offset++) {
      const i = (start + offset) % len;
      if (frames[i].analysis && frames[i].analysis.drop_score >= minDrop) {
        stopTimer();
        step = i;
        renderPanel();
        return;
      }
    }
  }

  btns.first.addEventListener('click', () => { stopTimer(); step = 0; renderPanel(); });
  btns.prev.addEventListener('click', () => { stopTimer(); step = Math.max(0, step - 1); renderPanel(); });
  btns.next.addEventListener('click', () => { stopTimer(); step = Math.min((frames || []).length - 1, step + 1); renderPanel(); });
  btns.play.addEventListener('click', () => {
    if (playing) { stopTimer(); renderPanel(); return; }
    playing = true;
    renderPanel();
    timer = setInterval(() => {
      const max = (frames || []).length - 1;
      if (step >= max) { stopTimer(); renderPanel(); return; }
      step++;
      renderPanel();
    }, 800);
  });
  btns.blunder.addEventListener('click', () => jumpToSeverity(10));
  btns.warn.addEventListener('click', () => jumpToSeverity(5));

  function renderEmpty() {
    stepEl.textContent = '暂未加载录像';
    listEl.innerHTML = '';
    const hint = document.createElement('div');
    hint.className = 'analysis-item';
    hint.style.cursor = 'default';
    hint.style.opacity = '0.65';
    hint.textContent = '对局结束后在弹窗选择“查看录像”，或从侧边栏加载录像文件';
    listEl.appendChild(hint);
    for (const k of Object.keys(btns)) btns[k].disabled = true;
    btns.play.textContent = '播放';
  }

  function applyVisibility() {
    if (frames && frames.length) {
      panel.hidden = false;
      return;
    }
    if (alwaysVisible) {
      panel.hidden = false;
      renderEmpty();
    } else {
      panel.hidden = true;
    }
  }

  // Append a synthetic terminal-display frame so the user can scrub past the
  // last move and actually see the final position. The replay's display
  // logic (in app.js) renders `frames[idx-1]` as the board state, which means
  // clicking the original last frame shows the position BEFORE the final
  // move, with the analysis card describing that move. The terminal layout
  // (winner reveal, last-card showdown, etc.) never gets a frame of its own.
  // This wrapper appends one extra entry that points at the real last frame
  // — when displayed via `displayFrame = allFrames[idx-1]`, it resolves to
  // the actual terminal state, which is exactly the picture we want to show.
  function withTerminalFrame(rawFrames) {
    if (!rawFrames || !rawFrames.length) return rawFrames;
    const last = rawFrames[rawFrames.length - 1];
    if (!last || !last.is_terminal) return rawFrames;
    const tail = {
      ...last,
      __terminal_display: true,
      analysis: null,
      action_info: null,
      action_id: null,
      actor: 'start',
      ply_index: (typeof last.ply_index === 'number') ? last.ply_index + 1 : last.ply_index,
    };
    return rawFrames.concat([tail]);
  }

  return {
    async enter(sessionId) {
      const data = await apiGet(API_BASE + '/' + sessionId + '/replay');
      frames = withTerminalFrame(data.frames);
      // Land on the last analyzed move (pre-final-position with analysis),
      // not the synthetic terminal-only frame appended after it. The user
      // typically wants to see "your last move's analysis" first; the
      // terminal picture is one click away via "next".
      const hasTerminal = frames.length && frames[frames.length - 1].__terminal_display;
      step = Math.max(0, frames.length - (hasTerminal ? 2 : 1));
      lastRenderedStep = -1;  // no prior animation frame reference
      playing = false;
      panel.hidden = false;
      renderPanel();
      return data;
    },
    enterWithFrames(framesData) {
      frames = withTerminalFrame(framesData);
      step = 0;
      lastRenderedStep = -1;
      playing = false;
      panel.hidden = false;
      renderPanel();
    },
    exit() {
      stopTimer();
      frames = null;
      playing = false;
      applyVisibility();
    },
    // Dev toggle: keep the panel mounted during live play so its styles
    // render without needing a finished game. When no replay is loaded,
    // the panel shows a placeholder with disabled controls.
    setAlwaysVisible(flag) {
      alwaysVisible = !!flag;
      applyVisibility();
    },
    isActive() { return frames !== null; },
    onRenderFrame(fn) { onRenderFrame = fn; },
    onExit(fn) { onExit = fn; },
    setResolveActor(fn) { resolveActor = fn; },
    getPanel() { return panel; },
  };
}
