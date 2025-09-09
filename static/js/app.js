/*
  static/js/app.js
  - Handles user interactions for chat input steps.
  - (The logic for rendering results is in a <script> tag in index.html)
*/
(() => {
  if (window.chatInputHandlerInitialized) return;
  window.chatInputHandlerInitialized = true;

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  
  const chatWindow = document.getElementById('chat-window');
  if (!chatWindow) return;

  const state = chatWindow.dataset.state || '지역';

  // --- Auto-scroll to the bottom ---
  chatWindow.scrollTop = chatWindow.scrollHeight;

  // --- Score Selection ---
  if (state === '점수') {
    const form = $('#scoreForm');
    const input = $('#scoreInput');

    $$('.score-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (!input || !form) return;
        input.value = btn.dataset.value; // "관광지수" | "인기도지수"
        form.submit();
      });
    });
  }

  // --- Theme Selection (Max 3) ---
  if (state === '테마') {
    const form = $('#themeForm');
    const input = $('#themesInput');
    const submitBtn = $('#themeSubmit');
    const clearBtn = $('#themeClear');
    const MAX_THEMES = 3;
    let pickedThemes = [];

    const refreshThemeUI = () => {
      // Update button selection state
      $$('.theme-btn').forEach(b => {
        b.classList.toggle('selected', pickedThemes.includes(b.dataset.value));
      });
      
      // Update hidden input and submit button state
      if(input) input.value = pickedThemes.join(',');
      if(submitBtn) {
        submitBtn.disabled = pickedThemes.length === 0;
        submitBtn.textContent = `선택 완료 (${pickedThemes.length}/${MAX_THEMES})`;
      }
    };

    $$('.theme-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const value = btn.dataset.value;
        if (pickedThemes.includes(value)) {
          pickedThemes = pickedThemes.filter(x => x !== value);
        } else {
          if (pickedThemes.length >= MAX_THEMES) return; // Prevent exceeding max
          pickedThemes.push(value);
        }
        refreshThemeUI();
      });
    });

    clearBtn?.addEventListener('click', () => {
      pickedThemes = [];
      refreshThemeUI();
    });

    submitBtn?.addEventListener('click', () => {
      if(form) form.submit();
    });

    refreshThemeUI(); // Initial UI setup
  }

  // --- Date Range Selection ---
  if (state === '기간') {
    const form = $('#rangeForm');
    const startDateInput = $('#startDate');
    const endDateInput = $('#endDate');
    const submitBtn = $('#rangeSubmit');
    const daysPreview = $('#daysPreview');
    const openStartBtn = $('#openStart');
    const openEndBtn = $('#openEnd');

    const calculateDays = () => {
      if (!startDateInput || !endDateInput || !daysPreview || !submitBtn) return;
      
      const startVal = startDateInput.value;
      const endVal = endDateInput.value;

      if (!startVal || !endVal) {
        daysPreview.textContent = '—';
        submitBtn.disabled = true;
        return;
      }

      const startDate = new Date(startVal);
      const endDate = new Date(endVal);
      const diffTime = endDate - startDate;
      
      if (diffTime < 0) {
        daysPreview.textContent = '날짜 오류';
        submitBtn.disabled = true;
        return;
      }

      const diffDays = Math.round(diffTime / (1000 * 60 * 60 * 24)) + 1;

      if (isNaN(diffDays) || diffDays < 1 || diffDays > 100) {
        daysPreview.textContent = '범위 오류';
        submitBtn.disabled = true;
      } else {
        daysPreview.textContent = `${diffDays}일`;
        submitBtn.disabled = false;
      }
    };

    startDateInput?.addEventListener('change', calculateDays);
    endDateInput?.addEventListener('change', calculateDays);
    openStartBtn?.addEventListener('click', () => startDateInput?.showPicker?.());
    openEndBtn?.addEventListener('click', () => endDateInput?.showPicker?.());

    form?.addEventListener('submit', (event) => {
      if (submitBtn && submitBtn.disabled) {
        event.preventDefault();
      }
    });
    
    calculateDays(); // Initial calculation
  }

  // --- Transport Mode Selection ---
  if (state === '이동수단') {
    const form = $('#transportForm');
    const input = $('#transportInput');
    $$('.transport-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (!form || !input) return;
        input.value = btn.dataset.value; // "walk" | "transit"
        form.submit();
      });
    });
  }

  // --- Trigger generation on 'executing' state ---
  if (state === '실행중') {
    // This POST request tells the server to start the heavy computation
    fetch('/do_generate', { method: 'POST' })
      .then(res => {
        if (!res.ok) {
            // Handle error if needed, e.g., show an error message
            console.error('Generation failed on the server.');
        }
      })
      .catch(err => console.error('Network error during generation:', err))
      .finally(() => {
        // Reload the page to show results or an error message from the server
        window.location.reload();
      });
  }
})();
