/* static/js/home.js — 홈(랜딩) 그리드 + 캐러셀 */

(() => {
  if (window.homePageInitialized) return;
  window.homePageInitialized = true;

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const explore = $('.explore');
  if (!explore) return;

  const API = explore.dataset.api || '/api/places';
  const grid = $('#grid');
  const pageList = $('#pageList');
  const prevPagesBtn = $('#prevPages');
  const nextPagesBtn = $('#nextPages');

  const sidoSel = $('#sido-filter');
  const cat1Sel = $('#cat1-filter');
  const cat3Sel = $('#cat3-filter');
  const searchInput = $('#search-input');

  const sortDropdown = $('#sortDropdown');
  const sortTrigger = sortDropdown?.querySelector('.sort-trigger');
  const sortMenu = sortDropdown?.querySelector('.sort-menu');

  // -------------------------------------------------
  // 캐러셀 유틸
  // -------------------------------------------------
  // 이미지 에러 → 슬라이드 제거 → 상태 갱신
  function handleCardImgError(e) {
    const img = e.target;
    const carousel = img.closest('.carousel');
    if (!carousel) return;

    const slides = carousel.querySelector('.slides');
    if (!slides) return;

    // 해당 <img> 제거
    img.remove();

    const imgs = slides.querySelectorAll('img');
    const left = imgs.length;

    if (left === 0) {
      // 전부 사라졌으면 “이미지 없음”
      const frame = carousel.parentElement;
      if (frame) {
        frame.innerHTML = `<div class="noimage">이미지 없음</div>`;
      }
      return;
    }

    // 인덱스/버튼 갱신
    const cur = parseInt(carousel.dataset.slide || '0', 10);
    updateCarouselState(carousel, Math.min(cur, left - 1));
  }

  // 끝/처음에서 버튼 숨기기 포함 캐러셀 상태 갱신
  function updateCarouselState(carousel, newIndex) {
    const slidesEl = carousel.querySelector('.slides');
    if (!slidesEl) return;

    const imgs = Array.from(slidesEl.querySelectorAll('img'));
    const n = imgs.length;
    if (n === 0) return;

    const i = Math.max(0, Math.min(newIndex, n - 1));
    carousel.dataset.slide = String(i);
    slidesEl.style.transform = `translateX(-${i * 100}%)`;

    const prevBtn = carousel.querySelector('.cbtn.prev');
    const nextBtn = carousel.querySelector('.cbtn.next');
    if (prevBtn) prevBtn.hidden = (i <= 0);
    if (nextBtn) nextBtn.hidden = (i >= n - 1);
  }

  // 카드 한 장의 캐러셀 초기화
  function setupCarousel(container, images, title) {
    const list = (images || []).filter(Boolean);
    if (list.length === 0) {
      container.innerHTML = `<div class="noimage">이미지 없음</div>`;
      return;
    }

    const slides = list.map(src => {
      const proxied = src.startsWith('/uploads') ? src : `/img-proxy?u=${encodeURIComponent(src)}`;
      return `<img src="${proxied}" alt="${title} 이미지" loading="lazy" referrerpolicy="no-referrer">`;
    }).join('');

    const showNav = list.length > 1;
    const nav = showNav
      ? `<button class="cbtn prev" type="button" aria-label="이전">‹</button>
         <button class="cbtn next" type="button" aria-label="다음">›</button>`
      : '';

    container.innerHTML = `
      <div class="carousel" data-slide="0" data-count="${list.length}">
        ${nav}
        <div class="slides">${slides}</div>
      </div>
    `;

    const carousel = container.querySelector('.carousel');
    // 에러 핸들러 바인딩 (위임 대신 직접)
    carousel.querySelectorAll('img').forEach(img => {
      img.addEventListener('error', handleCardImgError, { once: true });
    });

    // 초기 상태 (첫 장 → prev 숨김)
    updateCarouselState(carousel, 0);

    if (showNav) {
      carousel.addEventListener('click', (e) => {
        const prev = e.target.closest('.cbtn.prev');
        const next = e.target.closest('.cbtn.next');
        if (!prev && !next) return;
        const cur = parseInt(carousel.dataset.slide || '0', 10);
        const count = carousel.querySelectorAll('.slides img').length; // 남은 개수로 계산
        if (prev) updateCarouselState(carousel, cur - 1);
        if (next) updateCarouselState(carousel, Math.min(cur + 1, count - 1));
      });
    }
  }

  // -------------------------------------------------
  // 렌더링
  // -------------------------------------------------
  function cardHTML(item) {
    const { rank, title, addr1, cat1, cat3, images } = item;
    const badge = (cat1 || '').trim();
    return `
      <article class="place-card">
        <div class="rank">#${rank}</div>
        ${badge ? `<div class="badge">${badge}</div>` : ''}

        <div class="carousel" aria-label="${title} 이미지 프레임">
          <div class="slides"></div>
        </div>

        <div class="meta">
          <h3 class="title">${title}</h3>
          <div class="addr">${addr1 || ''}</div>
          <div class="tags">
            ${cat1 ? `<span class="chip">${cat1}</span>` : ''}
            ${cat3 ? `<span class="chip">${cat3}</span>` : ''}
          </div>
        </div>
      </article>
    `;
  }

  function renderGrid(items) {
    if (!grid) return;
    if (!items || items.length === 0) {
      grid.innerHTML = `<div class="empty">검색 결과가 없습니다.</div>`;
      return;
    }

    const html = items.map(cardHTML).join('');
    grid.innerHTML = html;

    // 각 카드 캐러셀 구성
    const cards = $$('.place-card', grid);
    items.forEach((item, idx) => {
      const card = cards[idx];
      const frame = card.querySelector('.carousel');
      setupCarousel(frame, item.images || [], item.title || '');
    });
  }

  function renderPagination(page, totalPages) {
    if (!pageList) return;
    pageList.innerHTML = '';

    const blockSize = 10;
    const currentBlock = Math.floor((page - 1) / blockSize);
    const start = currentBlock * blockSize + 1;
    const end = Math.min(start + blockSize - 1, totalPages);

    prevPagesBtn.disabled = start <= 1;
    nextPagesBtn.disabled = end >= totalPages;

    for (let p = start; p <= end; p++) {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.textContent = String(p);
      if (p === state.page) btn.setAttribute('aria-current', 'page');
      btn.addEventListener('click', () => {
        if (state.page === p) return;
        state.page = p;
        load();
      });
      li.appendChild(btn);
      pageList.appendChild(li);
    }

    prevPagesBtn.onclick = () => {
      state.page = Math.max(1, start - blockSize);
      load();
    };
    nextPagesBtn.onclick = () => {
      state.page = Math.min(totalPages, end + 1);
      load();
    };
  }

  // -------------------------------------------------
  // 상태 & 로딩
  // -------------------------------------------------
  const state = {
    page: 1,
    per_page: 40,
    sort: sortDropdown?.dataset.current || 'review',
    sido: 'all',
    cat1: 'all',
    cat3: 'all',
    q: ''
  };

  async function fetchFilterOptions() {
    try {
      const res = await fetch('/api/filter-options');
      const json = await res.json();
      if (!json.ok) return;

      // 시/도
      (json.options.sidos || []).forEach(s => {
        const o = document.createElement('option');
        o.value = s;
        o.textContent = s;
        sidoSel?.appendChild(o);
      });

      // 대분류
      (json.options.cat1s || []).forEach(c => {
        const o = document.createElement('option');
        o.value = c;
        o.textContent = c;
        cat1Sel?.appendChild(o);
      });

      // 소분류
      (json.options.cat3s || []).forEach(c => {
        const o = document.createElement('option');
        o.value = c;
        o.textContent = c;
        cat3Sel?.appendChild(o);
      });

    } catch (e) {
      // 필터 옵션 실패는 조용히 무시
      console.warn('filter-options fetch failed', e);
    }
  }

  async function load() {
    try {
      const params = new URLSearchParams({
        page: String(state.page),
        per_page: String(state.per_page),
        sort: state.sort,
      });
      if (state.sido && state.sido !== 'all') params.set('sido', state.sido);
      if (state.cat1 && state.cat1 !== 'all') params.set('cat1', state.cat1);
      if (state.cat3 && state.cat3 !== 'all') params.set('cat3', state.cat3);
      if (state.q) params.set('q', state.q);

      grid.innerHTML = `<div class="empty">불러오는 중…</div>`;

      const res = await fetch(`${API}?${params.toString()}`);
      const json = await res.json();

      if (!json.ok) {
        grid.innerHTML = `<div class="error">목록을 불러오지 못했어요.</div>`;
        return;
      }

      renderGrid(json.items || []);
      renderPagination(json.page, json.total_pages);
    } catch (e) {
      grid.innerHTML = `<div class="error">네트워크 오류가 발생했어요.</div>`;
    }
  }

  // -------------------------------------------------
  // 이벤트 바인딩
  // -------------------------------------------------
  // 정렬 드롭다운
  if (sortTrigger && sortMenu) {
    sortTrigger.addEventListener('click', () => {
      sortDropdown.classList.toggle('open');
      const expanded = sortTrigger.getAttribute('aria-expanded') === 'true';
      sortTrigger.setAttribute('aria-expanded', String(!expanded));
    });

    sortMenu.querySelectorAll('[role="option"]').forEach(opt => {
      opt.addEventListener('click', () => {
        const val = opt.dataset.value;
        sortMenu.querySelectorAll('[role="option"]').forEach(o => o.setAttribute('aria-selected', 'false'));
        opt.setAttribute('aria-selected', 'true');
        sortDropdown.dataset.current = val;
        sortTrigger.querySelector('.label').textContent = opt.textContent.trim();
        state.sort = val;
        state.page = 1;
        sortDropdown.classList.remove('open');
        sortTrigger.setAttribute('aria-expanded', 'false');
        load();
      });
    });

    // 바깥 클릭 닫기
    document.addEventListener('click', (e) => {
      if (!sortDropdown.contains(e.target)) {
        sortDropdown.classList.remove('open');
        sortTrigger.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // 필터 & 검색
  sidoSel?.addEventListener('change', () => { state.sido = sidoSel.value; state.page = 1; load(); });
  cat1Sel?.addEventListener('change', () => { state.cat1 = cat1Sel.value; state.page = 1; load(); });
  cat3Sel?.addEventListener('change', () => { state.cat3 = cat3Sel.value; state.page = 1; load(); });
  let searchTimer = null;
  searchInput?.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.q = (searchInput.value || '').trim();
      state.page = 1;
      load();
    }, 250);
  });

  // 초기 로드
  fetchFilterOptions().finally(load);
})();
