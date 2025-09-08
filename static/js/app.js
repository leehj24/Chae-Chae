/* static/js/app.js
   - index.html용 대화 상호작용
   - 점수 버튼/테마/기간/이동수단 동작 + 일정 생성 트리거
*/
(() => {
  if (window.__CHAT_INIT__) return;
  window.__CHAT_INIT__ = true;

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const state = ($('.chat-window')?.dataset.state) || '지역';

  /* ── 점수 선택 (관광지수 / 인기도지수) ───────────────── */
  if (state === '점수') {
    const form = $('#scoreForm');
    const input = $('#scoreInput');
    $$('.score-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        $$('.score-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        input.value = btn.dataset.value;   // "관광지수" | "인기도지수"
        form.submit();
      });
    });
  }

  /* ── 테마 선택 (최대 3개) ─────────────────────────────── */
  if (state === '테마') {
    const form = $('#themeForm');
    const input = $('#themesInput');
    const submit = $('#themeSubmit');
    const clear  = $('#themeClear');
    const MAX = 3;
    let picked = [];

    const refresh = () => {
      // 버튼 selection
      $$('.theme-btn').forEach(b => {
        const v = b.dataset.value;
        b.classList.toggle('selected', picked.includes(v));
      });
      // hidden input & submit 버튼
      input.value = picked.join(',');
      submit.disabled = picked.length === 0;
      submit.textContent = `선택 완료 (${picked.length}/${MAX})`;
    };

    $$('.theme-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const v = btn.dataset.value;
        if (picked.includes(v)) {
          picked = picked.filter(x => x !== v);
        } else {
          if (picked.length >= MAX) return; // 3개 초과 방지
          picked.push(v);
        }
        refresh();
      });
    });

    clear?.addEventListener('click', () => { picked = []; refresh(); });
    submit?.addEventListener('click', () => form.submit());
    refresh();
  }

  /* ── 기간 (날짜 범위) ─────────────────────────────────── */
  if (state === '기간') {
    const form = $('#rangeForm');
    const s = $('#startDate');
    const e = $('#endDate');
    const submit = $('#rangeSubmit');
    const daysPrev = $('#daysPreview');
    const openStart = $('#openStart');
    const openEnd   = $('#openEnd');

    const calcDays = () => {
      const sv = s.value, ev = e.value;
      if (!sv || !ev) { daysPrev.textContent = '—'; submit.disabled = true; return; }
      const sd = new Date(sv + 'T00:00:00');
      const ed = new Date(ev + 'T00:00:00');
      const diff = Math.round((ed - sd) / 86400000) + 1;
      if (isNaN(diff) || diff < 1 || diff > 100) {
        daysPrev.textContent = '범위 오류';
        submit.disabled = true;
      } else {
        daysPrev.textContent = `${diff}일`;
        submit.disabled = false;
      }
    };

    s?.addEventListener('change', calcDays);
    e?.addEventListener('change', calcDays);
    openStart?.addEventListener('click', () => s?.showPicker?.());
    openEnd?.addEventListener('click', () => e?.showPicker?.());
    form?.addEventListener('submit', (ev) => {
      // 브라우저 기본검사 외에도 버튼 disabled 방지
      if (submit.disabled) ev.preventDefault();
    });
    calcDays();
  }

  /* ── 이동수단 (걷기/대중교통) ─────────────────────────── */
  if (state === '이동수단') {
    const form = $('#transportForm');
    const input = $('#transportInput');
    $$('.transport-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        $$('.transport-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        input.value = btn.dataset.value; // "walk" | "transit"
        form.submit();
      });
    });
  }

  /* ── 실행중: 일정 생성 트리거 호출 ────────────────────── */
  if (state === '실행중') {
    // 서버 세션에 pending_job=True 일 때만 성공.
    // 성공/실패와 관계없이 완료되면 새로고침해 렌더 상태(완료/오류)를 반영.
    fetch('/do_generate', { method: 'POST' })
      .then(() => window.location.reload())
      .catch(() => window.location.reload());
  }
})();
