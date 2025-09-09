/* static/js/home.js
   - 이미지 임베드 차단 우회: 프록시 사용 + no-referrer
   - 실패 이미지 슬라이드/점 제거
   - 서버 페이지네이션 & 정렬 드롭다운
*/
(() => {
  if (window.__HOME_INIT__) return;
  window.__HOME_INIT__ = true;

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const exploreSection = $('.explore');
  const grid         = $('#grid');
  const pageList     = $('#pageList');
  const prevPagesBtn = $('#prevPages');
  const nextPagesBtn = $('#nextPages');

  // 커스텀 드롭다운 요소
  const sortDd       = $('#sortDropdown');
  const sortTrigger  = $('.sort-trigger', sortDd);
  const sortLabelEl  = $('.sort-trigger .label', sortDd);
  const sortMenu     = $('.sort-menu', sortDd);

  const API = exploreSection?.dataset.api || window.__EXPLORE_API__ || '/api/places';

  const PAGE_SIZE   = 40;  // 4 x 10
  const PAGE_WINDOW = 10;  // 10페이지 단위 창

  // --- 이미지 프록시 사용 (항상 true 권장)
  const useProxy = true;
  const proxied = (u) => useProxy ? `/img-proxy?u=${encodeURIComponent(u)}` : u;

  let items = [];        // 현재 페이지 데이터
  let currentPage = 1;
  let totalPages  = 1;
  let sortParam   = (sortDd?.dataset.current) || 'review'; // review | tour

  const isNonEmpty = (v) => {
    if (v === null || v === undefined) return false;
    const s = String(v).trim();
    if (!s) return false;
    if (['nan','none','null','undefined'].includes(s.toLowerCase())) return false;
    return true;
  };

  const escapeHtml = (str) =>
    String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

  const pageWindowRange = (page, total) => {
    const start = Math.floor((page - 1) / PAGE_WINDOW) * PAGE_WINDOW + 1;
    const end   = Math.min(start + PAGE_WINDOW - 1, total);
    return [start, end];
  };

  const buildUrl = (page) => {
    const q = new URLSearchParams({
      sort: sortParam,
      page: String(page),
      per_page: String(PAGE_SIZE),
    });
    return `${API}?${q.toString()}`;
  };

  const fetchPage = async (page = 1) => {
    if (!grid) return;
    grid.innerHTML = `<div class="empty">불러오는 중…</div>`;
    try {
      const res = await fetch(buildUrl(page), { headers: { 'Accept':'application/json' } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      const rows = Array.isArray(data?.items) ? data.items
                 : Array.isArray(data?.rows)  ? data.rows
                 : Array.isArray(data)        ? data : [];

      items       = rows.map(sanitizeItem);
      currentPage = Number(data?.page) || page;
      totalPages  = Number(data?.total_pages) || 1;

      renderGrid();
      renderPagination();
    } catch (err) {
      grid.innerHTML = `
        <div class="error">
          데이터를 불러오지 못했습니다. (${escapeHtml(String(err))})<br/>
          API(<code>${escapeHtml(API)}</code>)를 확인해 주세요.
        </div>`;
      if (pageList) pageList.innerHTML = '';
    }
  };

  // JSON을 그대로 보존
  const sanitizeItem = (r) => {
    const images = (Array.isArray(r.images) ? r.images : [r.firstimage, r.firstimage2]).filter(isNonEmpty);
    return {
      rank: Number(r.rank ?? 0),
      title: r.title ?? '',
      addr1: r.addr1 ?? '',
      cat1:  r.cat1  ?? '',
      cat3:  r.cat3  ?? '',
      tour_score:   r.tour_score,
      review_score: r.review_score,
      images,
    };
  };

  const splitCat3 = (s) => {
    if (!isNonEmpty(s)) return [];
    return String(s).split(/[\/,\|·∙]/g).map(x => x.trim()).filter(Boolean).slice(0, 3);
  };

  /* ── 칩(Chip) 톤 자동 배정 ───────────────────────────────── */
  const PASTELS = ['#BDE0FE','#C9E4CA','#F9E0AE','#FFD6E7','#CDE7F0','#EAD7F4','#F7D794','#D6F5E5','#E7E3FF','#FDE2D2'];
  const hash = (s) => { let h = 0; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0; return h; };
  const hexToRgb = (hex) => { const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex); return m ? { r: parseInt(m[1],16), g: parseInt(m[2],16), b: parseInt(m[3],16) } : { r:0,g:0,b:0 }; };
  const yiqBrightness = ({r,g,b}) => (r*299 + g*587 + b*114) / 1000;
  const chipStyleFor = (label) => {
    const bg = PASTELS[hash(label) % PASTELS.length];
    const rgb = hexToRgb(bg);
    const bright = yiqBrightness(rgb) >= 160;
    const fg = bright ? '#111' : '#fff';
    const border = bright ? 'rgba(0,0,0,.12)' : 'rgba(255,255,255,.18)';
    return { bg, fg, border };
  };
  const renderChip = (text) => {
    const { bg, fg, border } = chipStyleFor(text);
    return `<span class="chip" style="background:${bg};color:${fg};border-color:${border}">${escapeHtml(text)}</span>`;
  };

  // UI 표시용: 원본 점수 → ×100 → 소수점 1자리
  const formatBadge = (raw) => {
    if (!isNonEmpty(raw)) return '-';
    const n = Number(String(raw).replace(/,/g, ''));
    if (!Number.isFinite(n)) return '-';
    const val = n * 100;
    return `${(Math.round(val * 10) / 10).toFixed(1)}점`;
  };

  const badgeFromRow = (row) => {
    const primary  = (sortParam === 'tour') ? row.tour_score : row.review_score;
    const fallback = (sortParam === 'tour') ? row.review_score : row.tour_score;
    return isNonEmpty(primary) ? formatBadge(primary)
         : isNonEmpty(fallback) ? formatBadge(fallback)
         : '-';
  };

  const updateArrows = (carousel) => {
    const prev = $('.cbtn.prev', carousel);
    const next = $('.cbtn.next', carousel);
    if (!prev || !next) return;
    const i = parseInt(carousel.dataset.slide || '0', 10);
    const count = parseInt(carousel.dataset.count || '0', 10);
    prev.hidden = (i <= 0);
    next.hidden = (i >= count - 1);
  };

  // 실패 이미지 처리
  window.handleImgError = function(e){
    const img = e.target;
    const slides = img.closest('.slides');
    const carousel = img.closest('.carousel');
    const dotsWrap = $('.dots', carousel);
    if (!slides || !carousel) return;

    // 해당 이미지 제거
    try { img.remove(); } catch(_) {}

    // 점(마지막 점) 하나 제거
    const dotList = $$('.dots .dot', carousel);
    if (dotList.length) {
      dotList[dotList.length - 1].remove();
    }

    // 남은 슬라이드 수 갱신
    const remain = slides.children.length;
    carousel.dataset.count = String(remain);

    if (remain <= 1) {
      const prev = $('.cbtn.prev', carousel);
      const next = $('.cbtn.next', carousel);
      if (prev) prev.hidden = true;
      if (next) next.hidden = true;
      if (dotsWrap) dotsWrap.remove();
    }

    if (remain === 0) {
      slides.insertAdjacentHTML('afterend', '<div class="noimage">이미지 없음</div>');
      slides.remove();
    }
  };

  const renderGrid = () => {
    if (!grid) return;
    if (!items.length) {
      grid.innerHTML = `<div class="empty">표시할 결과가 없습니다.</div>`;
      return;
    }

    const cards = items.map((row) => {
      const rank  = row.rank || 0;
      const badgeText = badgeFromRow(row);

      const title = row.title || '';
      const addr1 = row.addr1 || '';
      const cat1  = row.cat1  || '';
      const cat3s = splitCat3(row.cat3);
      const imgs  = (row.images || []).filter(isNonEmpty);
      const imgP  = imgs.map(proxied);
      const imgCount = imgP.length;

      const showNav = imgCount > 1;          // 1장이면 화살표/점 숨김
      const hasImage = imgCount > 0;

      const dots = showNav
        ? `<div class="dots" role="tablist" aria-label="이미지 선택 점">
             ${imgP.map((_, di) => `<button type="button" class="dot" data-action="dot" data-i="${di}" aria-label="${di+1}번째 이미지"></button>`).join('')}
           </div>` : '';

      const slides = hasImage
        ? `<div class="slides" style="transform:translateX(0%)">
             ${imgP.map((src, si) => `
                <img src="${src}"
                     alt="${escapeHtml(title)} 이미지 ${si+1}"
                     loading="lazy"
                     referrerpolicy="no-referrer"
                     crossorigin="anonymous"
                     onerror="handleImgError(event)">
             `).join('')}
           </div>`
        : `<div class="noimage">이미지 없음</div>`;

      const carousel = `
        <div class="carousel" data-slide="0" data-count="${imgCount}">
          ${showNav ? `<button class="cbtn prev" type="button" aria-label="이전 이미지" data-action="prev" hidden>‹</button>` : ''}
          ${showNav ? `<button class="cbtn next" type="button" aria-label="다음 이미지" data-action="next">›</button>` : ''}
          ${slides}
          ${dots}
        </div>`;

      const tags = `
        <div class="tags">
          ${isNonEmpty(cat1) ? renderChip(cat1) : ''}
          ${cat3s.map(renderChip).join('')}
        </div>`;

      return `
        <article class="place-card">
          <span class="rank">#${rank}</span>
          <span class="badge">${badgeText}</span>
          ${carousel}
          <div class="meta">
            <h3 class="title">${escapeHtml(title)}</h3>
            <div class="addr">${escapeHtml(addr1)}</div>
            ${tags}
          </div>
        </article>`;
    });

    grid.innerHTML = cards.join('');

    // 초기 dot/arrow 상태 세팅
    $$('[data-action="dot"]', grid).forEach((btn) => {
      if (Number(btn.dataset.i) === 0) btn.setAttribute('aria-current', 'true');
    });
    $$('.carousel', grid).forEach(updateArrows);
  };

  const renderPagination = () => {
    if (!pageList) return;
    const [startW, endW] = pageWindowRange(currentPage, totalPages);
    const btns = [];
    for (let p = startW; p <= endW; p++) {
      const cur = p === currentPage ? ' aria-current="page"' : '';
      btns.push(`<li><button type="button" data-page="${p}"${cur}>${p}</button></li>`);
    }
    pageList.innerHTML = btns.join('');
    if (prevPagesBtn) prevPagesBtn.disabled = startW <= 1;
    if (nextPagesBtn) nextPagesBtn.disabled = endW >= totalPages;
  };

  // ── 캐러셀 인터랙션
  const onCarouselClick = (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const carousel = e.target.closest('.carousel');
    if (!carousel) return;
    const slides = $('.slides', carousel);
    const dots   = $$('.dots .dot', carousel);
    if (!slides) return;

    const count = dots.length || (slides.children?.length ?? 0);
    if (count <= 1) return;

    let i = parseInt(carousel.dataset.slide || '0', 10);
    const action = btn.dataset.action;
    if (action === 'prev') i = Math.max(0, i - 1);
    else if (action === 'next') i = Math.min(count - 1, i + 1);
    else if (action === 'dot') i = parseInt(btn.dataset.i || '0', 10);
    else return;

    carousel.dataset.slide = String(i);
    slides.style.transform = `translateX(-${i * 100}%)`;
    dots.forEach((d, di) => di === i ? d.setAttribute('aria-current','true') : d.removeAttribute('aria-current'));

    updateArrows(carousel);
  };

  // ── 페이지 이동
  const onPageClick = (e) => {
    const b = e.target.closest('button[data-page]');
    if (!b) return;
    const p = parseInt(b.dataset.page, 10);
    if (!Number.isFinite(p)) return;
    fetchPage(p);
  };
  const onPrevPages = () => {
    const [startW] = pageWindowRange(currentPage, totalPages);
    const prevStart = Math.max(1, startW - PAGE_WINDOW);
    fetchPage(prevStart);
  };
  const onNextPages = () => {
    const [, endW] = pageWindowRange(currentPage, totalPages);
    const nextStart = endW + 1;
    if (nextStart <= totalPages) fetchPage(nextStart);
  };

  // ── 커스텀 정렬 드롭다운
  const closeDropdown = () => {
    sortDd?.classList.remove('open');
    sortTrigger?.setAttribute('aria-expanded','false');
  };
  const openDropdown = () => {
    sortDd?.classList.add('open');
    sortTrigger?.setAttribute('aria-expanded','true');
  };
  const toggleDropdown = () => {
    sortDd?.classList.contains('open') ? closeDropdown() : openDropdown();
  };
  const setSort = (value, labelText) => {
    sortParam = (value === 'tour' || value === '관광지수') ? 'tour' : 'review';
    sortLabelEl.textContent = labelText || (sortParam === 'tour' ? '관광 지수' : '인기도 지수');
    $$('.sort-menu [role="option"]', sortDd).forEach(el => {
      el.setAttribute('aria-selected', el.dataset.value === sortParam ? 'true' : 'false');
    });
    closeDropdown();
    fetchPage(1);
  };

  // 이벤트 바인딩
  if (grid)          grid.addEventListener('click', onCarouselClick);
  sortTrigger?.addEventListener('click', toggleDropdown);
  document.addEventListener('click', (e) => {
    if (!sortDd) return;
    if (sortDd.contains(e.target)) return;
    closeDropdown();
  });
  sortMenu?.addEventListener('click', (e) => {
    const li = e.target.closest('[role="option"]');
    if (!li) return;
    setSort(li.dataset.value, li.textContent.trim());
  });
  sortTrigger?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleDropdown(); }
    if (e.key === 'Escape') closeDropdown();
  });
  if (pageList)      pageList.addEventListener('click', onPageClick);
  if (prevPagesBtn)  prevPagesBtn.addEventListener('click', onPrevPages);
  if (nextPagesBtn)  nextPagesBtn.addEventListener('click', onNextPages);

  // 초기 로드
  fetchPage(1);
})();