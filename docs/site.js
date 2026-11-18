const chapters = [...document.querySelectorAll(".chapter")];
if (chapters.length) {
  const links = [...document.querySelectorAll(".toc a")];
  let activeChapter;
  let scheduled = false;
  function markCurrent() {
    const position = Math.min(window.innerHeight * .3, 200);
    const current = [...chapters].reverse().find(section => section.getBoundingClientRect().top <= position) ?? chapters[0];
    scheduled = false;
    if (activeChapter === current) return;
    activeChapter = current;
    links.forEach(link => {
      if (link.hash === "#" + current.id) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }
  markCurrent();
  window.addEventListener("scroll", () => {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(markCurrent);
    }
  }, {passive:true});
}

// 正文先显示随 HTML 到达的预览图，接近视口时再换成清晰图。
const galleries = document.querySelectorAll('[data-gallery]');
const compactScreen = window.matchMedia('(max-width: 700px)');
async function upgradeGallery(grid) {
  const photos = Array.from(grid.querySelectorAll('img'));
  if (compactScreen.matches && location.protocol !== 'file:') {
    try {
      const response = await fetch(grid.dataset.gallery);
      if (!response.ok) throw new Error('图组未加载');
      const pack = await response.json();
      if (pack.length !== photos.length) throw new Error('图组不完整');
      await Promise.all(photos.map(async (photo, index) => {
        const image = new Image();
        image.src = 'data:image/webp;base64,' + pack[index];
        await image.decode();
        photo.src = image.src;
      }));
      return;
    } catch {
      // 请求中断时仍显示预览图，并尝试常规的响应式图片。
    }
  }
  await Promise.all(photos.map(async photo => {
    const candidate = new Image();
    candidate.sizes = photo.sizes;
    candidate.srcset = photo.dataset.srcset;
    try {
      await candidate.decode();
      photo.srcset = candidate.srcset;
    } catch {
      // 高清图不可用时保留已经显示的预览。
    }
  }));
}
if ('IntersectionObserver' in window) {
  const galleryObserver = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      galleryObserver.unobserve(entry.target);
      upgradeGallery(entry.target);
    }
  }, {rootMargin: '280px 0px'});
  galleries.forEach(grid => galleryObserver.observe(grid));
} else {
  galleries.forEach(upgradeGallery);
}

const player = document.querySelector('#player');
const quality = document.querySelector('#film-quality');
if (player && quality) {
  const direct = document.querySelector('#film-direct');
  const status = document.querySelector('#film-status');
  let pending = null;
  let hasStarted = false;
  function showProblem(message) {
    direct.hidden = false;
    status.textContent = message;
  }
  quality.addEventListener('change', () => {
    const next = {
      url: new URL(quality.value, document.baseURI).href,
      time: pending?.time ?? (player.ended ? 0 : player.currentTime),
      play: pending?.play ?? (!player.paused || !hasStarted || player.ended)
    };
    pending = next;
    status.textContent = '';
    direct.href = next.url;
    direct.hidden = quality.selectedIndex === 0;
    player.preload = next.play ? 'auto' : 'metadata';
    player.src = next.url;
    player.load();
    // 在选择画质的用户操作内请求播放，保留移动浏览器的声音播放授权。
    if (next.play) {
      player.play().catch(error => {
        if (player.src !== next.url || error.name === 'AbortError') return;
        showProblem(error.name === 'NotAllowedError' ? '请轻点视频上的播放键。' : '视频未能播放，可直接打开视频。');
      });
    }
  });
  player.addEventListener('loadedmetadata', () => {
    if (!pending || pending.url !== player.currentSrc) return;
    if (pending.time > 0) player.currentTime = Math.min(pending.time, player.duration);
    pending = null;
  });
  player.addEventListener('error', () => {
    pending = null;
    showProblem('视频未能加载，可直接打开视频。');
  });
  player.addEventListener('playing', () => {
    hasStarted = true;
    status.textContent = '';
  });
  if ('IntersectionObserver' in window && !navigator.connection?.saveData) {
    const videoObserver = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      videoObserver.disconnect();
      if (player.networkState === HTMLMediaElement.NETWORK_EMPTY || player.readyState === 0) {
        player.preload = 'metadata';
      }
    }, {rootMargin: '80px 0px'});
    videoObserver.observe(player);
  }
}
