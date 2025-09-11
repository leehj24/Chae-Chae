/* static/js/home.js — 홈(랜딩) 그리드 + 캐러셀 + 업로더 슬라이드 */

(() => {
  if (window.homePageInitialized) return;
  window.homePageInitialized = true;

  const $  = (sel, root = document) => root.querySelector(sel);
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
  const sortTrigger  = sortDropdown?.querySelector('.sort-trigger');
  const sortMenu     = sortDropdown?.querySelector('.sort-menu');
  
  const orderDropdown = $('#orderDropdown');
  const orderTrigger  = orderDropdown?.querySelector('.sort-trigger');
  const orderMenu     = orderDropdown?.querySelector('.sort-menu');

  const ratingModal = $('#ratingModal');
  const modalPlaceTitle = $('#modalPlaceTitle');
  const ratingFormStars = ratingModal?.querySelector('.stars');
  const submitRatingBtn = $('#submitRatingBtn');
  let currentRatingTarget = null;

  // 후기 모달 요소 가져오기
  const reviewModal = $('#reviewModal');
  const reviewModalPlaceTitle = $('#reviewModalPlaceTitle');
  const reviewIframe = $('#reviewIframe');

  // ─────────────────────────────────────────────
  // 캐러셀 유틸 (홈 카드 전용)
  // ─────────────────────────────────────────────

  window.handleHomeCardImgError = function(e){
    const img = e.target;
    const slide = img.closest('.carousel-slide') || img;
    const carousel = img.closest('.carousel');
    const slidesEl = carousel?.querySelector('.slides');
    if (!carousel || !slidesEl) return;
    slide.remove();
    const leftImgSlides = slidesEl.querySelectorAll('.carousel-slide[data-type="img"]').length;
    if (leftImgSlides === 0 && !slidesEl.querySelector('.noimage')) {
      const noimgSlide = document.createElement('div');
      noimgSlide.className = 'carousel-slide';
      noimgSlide.innerHTML = `<div class="noimage">이미지 없음</div>`;
      slidesEl.insertBefore(noimgSlide, slidesEl.firstChild);
    }
    const cur = parseInt(carousel.dataset.slide || '0', 10);
    const total = slidesEl.querySelectorAll('.carousel-slide').length;
    updateCarouselState(carousel, Math.min(cur, Math.max(0, total - 1)));
  };

  function updateCarouselState(carousel, newIndex){
    const slidesEl = carousel.querySelector('.slides');
    const slides = slidesEl ? Array.from(slidesEl.querySelectorAll('.carousel-slide')) : [];
    const n = slides.length;
    if (!slidesEl || n === 0) return;
    const i = Math.max(0, Math.min(newIndex, n - 1));
    carousel.dataset.slide = String(i);
    slidesEl.style.transform = `translateX(-${i * 100}%)`;
    const prevBtn = carousel.querySelector('.cbtn.prev');
    const nextBtn = carousel.querySelector('.cbtn.next');
    if (prevBtn) prevBtn.hidden = (i <= 0);
    if (nextBtn) nextBtn.hidden = (i >= n - 1);
  }

  function setupCarousel(container, images, title, addr1){
    const list = (images || []).filter(Boolean).slice(0, 4);
    const hasImages = list.length > 0;
    const canUploadMore = list.length < 4;
    const slideParts = [];

    if (!hasImages){
      slideParts.push(`<div class="carousel-slide"><div class="noimage">이미지 없음</div></div>`);
    }

    for (const src of list){
      const proxied = src.startsWith('/uploads') ? src : `/img-proxy?u=${encodeURIComponent(src)}`;
      slideParts.push(`<div class="carousel-slide" data-type="img"><img src="${proxied}" alt="${title} 이미지" loading="lazy" referrerpolicy="no-referrer" onerror="handleHomeCardImgError(event)"></div>`);
    }

    if (canUploadMore){
      slideParts.push(`
        <div class="carousel-slide uploader-slide" data-type="uploader">
          <label class="image-uploader uploader-label" tabindex="0" aria-label="사진 올리기 또는 촬영">
            <div class="up-ic">📷</div><div class="up-title">사진 올리기 / 촬영</div><div class="up-hint">최대 1장 · 8MB</div>
            <input type="file" class="uploader-input" accept="image/*" capture="environment" />
          </label>
        </div>`);
    }

    const showNav = slideParts.length > 1;
    const nav = showNav ? `<button class="cbtn prev" type="button" aria-label="이전">‹</button><button class="cbtn next" type="button" aria-label="다음">›</button>` : '';
    
    container.dataset.slide = '0';
    container.dataset.title = title;
    container.dataset.addr1 = addr1;
    container.innerHTML = `${nav}<div class="slides">${slideParts.join('')}</div>`;

    const fileInput = container.querySelector('.uploader-input');
    updateCarouselState(container, 0);

    async function doUpload(file){
      if (!file) return;
      
      const card = container.closest('.place-card');
      if (card.classList.contains('upload-locked')) {
        alert('이 장소에는 이미 사진을 올리셨어요. 새로고침 후 다시 시도해주세요.');
        return;
      }

      try{
        const upSlide = container.querySelector('.uploader-slide');
        if (upSlide) upSlide.innerHTML = '<div class="image-uploader">업로드 중…</div>';

        const fd = new FormData();
        fd.append('file', file);
        fd.append('title', title);
        fd.append('addr1', addr1);

        const res = await fetch('/api/upload-image', { method:'POST', body: fd });
        const json = await res.json();
        if (!json.ok) throw new Error(json.error || '업로드 실패');

        card.classList.add('upload-locked');
        setupCarousel(container, json.images, title, addr1);
        
        const newImageIndex = json.images.length - (json.images.length < 4 ? 1 : 0);
        updateCarouselState(container, Math.max(0, newImageIndex - 1));

      } catch(err) {
        alert(err.message || '업로드 중 오류가 발생했습니다.');
        setupCarousel(container, images, title, addr1);
      }
    }

    if (fileInput) fileInput.addEventListener('change', (e)=> doUpload(e.target.files?.[0]));
    
    if (!container.dataset.listenerAttached) {
      container.dataset.listenerAttached = 'true';
      container.addEventListener('click', (e)=>{
        const prev = e.target.closest('.cbtn.prev');
        const next = e.target.closest('.cbtn.next');
        if (!prev && !next) return;
        
        const slidesEl = container.querySelector('.slides');
        if (!slidesEl) return;
        
        const slides = Array.from(slidesEl.querySelectorAll('.carousel-slide'));
        const cur = parseInt(container.dataset.slide || '0', 10);
        const lastIdx = slides.length - 1;
        
        if (prev){ 
          updateCarouselState(container, cur - 1);
        }
        if (next){ 
          const nextIdx = Math.min(cur + 1, lastIdx);
          updateCarouselState(container, nextIdx); 
        }
      });
    }
  }

  // 별점 UI 업데이트 함수
  function updateStarRatingUI(starDisplayElement, avgRating, totalRatings) {
    const avg = avgRating || 0;
    const total = totalRatings || 0;
    const inner = starDisplayElement.querySelector('.stars-inner');
    const avgEl = starDisplayElement.querySelector('.rating-avg');
    const countEl = starDisplayElement.querySelector('.rating-count');

    if (inner) inner.style.width = `${(avg / 5) * 100}%`;
    if (avgEl) avgEl.textContent = avg.toFixed(1);
    if (countEl) countEl.textContent = total;
  }

  // 장소 상세 정보(별점, 카카오URL 등) 가져오기
  async function fetchPlaceDetails(cardElement, forceRefresh = false) {
      if (cardElement.dataset.detailsLoaded && !forceRefresh) {
          return JSON.parse(cardElement.dataset.detailsLoaded);
      }
      
      const title = cardElement.dataset.title;
      const addr1 = cardElement.dataset.addr1;
      const mapx = cardElement.dataset.mapx;
      const mapy = cardElement.dataset.mapy;

      const params = new URLSearchParams({ title, addr1, mapx, mapy });
      try {
          const res = await fetch(`/api/place-details?${params.toString()}`);
          const data = await res.json();
          if (!data.ok) throw new Error('정보 로드 실패');
          
          cardElement.dataset.detailsLoaded = JSON.stringify(data);
          const starDisplay = cardElement.querySelector('.star-rating-display');
          if (starDisplay) {
              updateStarRatingUI(starDisplay, data.avg_rating, data.total_ratings);
          }
          return data;
      } catch (e) {
          console.error('장소 상세 정보 로드 실패:', title, e);
          const starDisplay = cardElement.querySelector('.star-rating-display');
          if (starDisplay) updateStarRatingUI(starDisplay, 0, 0);
          return { kakao_url: null, avg_rating: 0, total_ratings: 0, my_rating: 0 };
      }
  }


  function cardHTML(item){
    const { rank, title, addr1, cat1, cat3, review_score, tour_score, mapx, mapy } = item;
    
    let score = state.sort === 'review' ? review_score : tour_score;
    if (score !== null && typeof score !== 'undefined') {
      score *= 100;
    }

    const scoreBadgeHTML = (score !== null && typeof score !== 'undefined')
      ? `<div class="badge score-badge">${score.toFixed(2)}</div>`
      : '';

    return `
      <article class="place-card" data-title="${title}" data-addr1="${addr1}" data-mapx="${mapx}" data-mapy="${mapy}">
        <div class="rank">#${rank}</div>
        ${scoreBadgeHTML}
        <div class="carousel" aria-label="${title} 이미지 프레임"><div class="slides"></div></div>
        <div class="meta">
          <h3 class="title">${title}</h3>
          <div class="addr">${addr1 || ''}</div>
          <div class="tags">
            ${cat1 ? `<span class="chip">${cat1}</span>` : ''}
            ${cat3 ? `<span class="chip">${cat3}</span>` : ''}
          </div>
          <div class="card-actions">
            <div class="star-rating-display" role="button" tabindex="0" aria-label="별점주기">
              <div class="stars-outer">
                <div class="stars-inner"></div>
              </div>
              <span class="rating-info">
                <span class="rating-avg">…</span> (<span class="rating-count">…</span>)
              </span>
            </div>
            <button type="button" class="review-btn">후기 보기</button>
          </div>
        </div>
      </article>
    `;
  }

  function renderGrid(items){
    if (!grid) return;
    if (!items || items.length === 0){
      grid.innerHTML = `<div class="empty">검색 결과가 없습니다.</div>`;
      return;
    }
    grid.innerHTML = items.map(cardHTML).join('');
    const cards = $$('.place-card', grid);
    items.forEach((item, idx) => {
      const card = cards[idx];
      const frame = card.querySelector('.carousel');
      setupCarousel(frame, item.images || [], item.title || '', item.addr1 || '');
      
      fetchPlaceDetails(card);
    });
  }

  function renderPagination(page, totalPages){
    if (!pageList) return;
    pageList.innerHTML = '';
    const blockSize = 10;
    const currentBlock = Math.floor((page - 1) / blockSize);
    const start = currentBlock * blockSize + 1;
    const end = Math.min(start + blockSize - 1, totalPages);
    prevPagesBtn.disabled = start <= 1;
    nextPagesBtn.disabled = end >= totalPages;
    for (let p = start; p <= end; p++){
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.textContent = String(p);
      if (p === state.page) btn.setAttribute('aria-current', 'page');
      btn.addEventListener('click', ()=>{ if (state.page !== p) { state.page = p; load(); } });
      li.appendChild(btn);
      pageList.appendChild(li);
    }
    prevPagesBtn.onclick = ()=>{ state.page = Math.max(1, start - blockSize); load(); };
    nextPagesBtn.onclick = ()=>{ state.page = Math.min(totalPages, end + 1); load(); };
  }

  const state = {
    page: 1, per_page: 40,
    sort: sortDropdown?.dataset.current || 'review',
    order: orderDropdown?.dataset.current || 'desc',
    sido: 'all', cat1: 'all', cat3: 'all', q: ''
  };

  async function fetchFilterOptions(){
    try{
      const res = await fetch('/api/filter-options');
      const json = await res.json();
      if (!json.ok) return;
      (json.options.sidos || []).forEach(s => { const o = document.createElement('option'); o.value = s; o.textContent = s; sidoSel?.appendChild(o); });
      (json.options.cat1s || []).forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c; cat1Sel?.appendChild(o); });
      (json.options.cat3s || []).forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c; cat3Sel?.appendChild(o); });
    }catch(e){ console.warn('filter-options fetch failed', e); }
  }

  async function load(){
    try{
      const params = new URLSearchParams({
        page: String(state.page),
        per_page: String(state.per_page),
        sort: state.sort,
        order: state.order,
      });
      if (state.sido && state.sido !== 'all') params.set('sido', state.sido);
      if (state.cat1 && state.cat1 !== 'all') params.set('cat1', state.cat1);
      if (state.cat3 && state.cat3 !== 'all') params.set('cat3', state.cat3);
      if (state.q) params.set('q', state.q);
      grid.innerHTML = `<div class="empty">불러오는 중…</div>`;
      const res = await fetch(`${API}?${params.toString()}`);
      const json = await res.json();
      if (!json.ok){
        grid.innerHTML = `<div class="error">목록을 불러오지 못했어요.</div>`;
        return;
      }
      renderGrid(json.items || []);
      renderPagination(json.page, json.total_pages);
    }catch(e){
      grid.innerHTML = `<div class="error">네트워크 오류가 발생했어요.</div>`;
    }
  }

  function setupDropdown(dropdown, trigger, menu, stateKey) {
    if (!dropdown || !trigger || !menu) return;
    trigger.addEventListener('click', () => {
      dropdown.classList.toggle('open');
      trigger.setAttribute('aria-expanded', String(dropdown.classList.contains('open')));
    });
    menu.querySelectorAll('[role="option"]').forEach(opt => {
      opt.addEventListener('click', () => {
        const val = opt.dataset.value;
        if (state[stateKey] === val) {
          dropdown.classList.remove('open');
          trigger.setAttribute('aria-expanded', 'false');
          return;
        }
        menu.querySelectorAll('[role="option"]').forEach(o => o.setAttribute('aria-selected', 'false'));
        opt.setAttribute('aria-selected', 'true');
        dropdown.dataset.current = val;
        trigger.querySelector('.label').textContent = opt.textContent.trim();
        state[stateKey] = val;
        state.page = 1;
        dropdown.classList.remove('open');
        trigger.setAttribute('aria-expanded', 'false');
        load();
      });
    });
    document.addEventListener('click', (e) => {
      if (!dropdown.contains(e.target)) {
        dropdown.classList.remove('open');
        trigger.setAttribute('aria-expanded', 'false');
      }
    });
  }
  
  if (ratingModal) {
      const setStarsInModal = (rating) => {
        const stars = ratingFormStars.querySelectorAll('.star');
        stars.forEach(star => {
          star.classList.toggle('selected', star.dataset.value <= rating);
        });
        ratingFormStars.dataset.rating = rating;
        submitRatingBtn.disabled = rating == 0;
      };

      ratingFormStars.addEventListener('mouseover', e => {
        if (e.target.classList.contains('star')) {
          const rating = e.target.dataset.value;
          const stars = ratingFormStars.querySelectorAll('.star');
          stars.forEach(star => {
            star.classList.toggle('hover', star.dataset.value <= rating);
          });
        }
      });
      ratingFormStars.addEventListener('mouseout', () => {
        const currentRating = ratingFormStars.dataset.rating;
        setStarsInModal(currentRating);
      });
      
      ratingFormStars.addEventListener('click', e => {
        if (e.target.classList.contains('star')) {
          setStarsInModal(e.target.dataset.value);
        }
      });

      const openRatingModal = (targetCard) => {
        currentRatingTarget = targetCard;
        modalPlaceTitle.textContent = targetCard.dataset.title;
        fetchPlaceDetails(targetCard).then(data => setStarsInModal(data.my_rating || 0));
        ratingModal.classList.add('visible');
      };

      const closeRatingModal = () => {
        ratingModal.classList.remove('visible');
        currentRatingTarget = null;
        setStarsInModal(0);
      };

      const openReviewModal = (title, url) => {
        reviewModalPlaceTitle.textContent = title;
        reviewIframe.src = url;
        reviewModal.classList.add('visible');
      }

      const closeReviewModal = () => {
        reviewModal.classList.remove('visible');
        reviewIframe.src = 'about:blank';
      }

      ratingModal.querySelectorAll('.modal-close-btn').forEach(btn => btn.addEventListener('click', closeRatingModal));

      reviewModal?.querySelectorAll('.modal-close-btn').forEach(btn => btn.addEventListener('click', closeReviewModal));

      submitRatingBtn.addEventListener('click', async () => {
        if (!currentRatingTarget) return;
        const { title, addr1, mapx, mapy } = currentRatingTarget.dataset;
        const rating = ratingFormStars.dataset.rating;

        try {
          const res = await fetch('/api/submit-review', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ title, addr1, rating, mapx, mapy }),
          });
          const json = await res.json();
          if (!json.ok) throw new Error(json.error || '저장 실패');
          
          fetchPlaceDetails(currentRatingTarget, true);
          closeRatingModal();
        } catch (err) {
          alert(err.message || '저장 중 오류 발생');
        }
      });

      grid.addEventListener('click', async (e) => {
        const card = e.target.closest('.place-card');
        if (!card) return;

        if (e.target.closest('.star-rating-display')) {
          openRatingModal(card);
        }
        if (e.target.closest('.review-btn')) {
          const details = await fetchPlaceDetails(card);
          if (details.kakao_url) {
            openReviewModal(card.dataset.title, details.kakao_url);
          } else {
            if (confirm('이 장소에 대한 후기 정보가 아직 없습니다. 별점을 남기시겠어요?')) {
              openRatingModal(card);
            }
          }
        }
      });
  }

  setupDropdown(sortDropdown, sortTrigger, sortMenu, 'sort');
  setupDropdown(orderDropdown, orderTrigger, orderMenu, 'order');
  
  sidoSel?.addEventListener('change', ()=>{ state.sido = sidoSel.value; state.page = 1; load(); });
  cat1Sel?.addEventListener('change', ()=>{ state.cat1 = cat1Sel.value; state.page = 1; load(); });
  cat3Sel?.addEventListener('change', ()=>{ state.cat3 = cat3Sel.value; state.page = 1; load(); });

  let searchTimer = null;
  searchInput?.addEventListener('input', ()=>{
    clearTimeout(searchTimer);
    searchTimer = setTimeout(()=>{
      state.q = (searchInput.value || '').trim();
      state.page = 1; load();
    }, 250);
  });

  fetchFilterOptions().finally(load);
})();