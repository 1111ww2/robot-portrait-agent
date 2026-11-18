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

const pocketIndex = document.querySelector(".pocket-toc");
if (pocketIndex) {
  pocketIndex.addEventListener("click", event => {
    const link = event.target.closest("a[href^='#']");
    if (link) {
      pocketIndex.open = false;
      const target = document.querySelector(link.hash);
      target.setAttribute("tabindex", "-1");
      target.focus({preventScroll:true});
    }
  });
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
  let pendingTime = 0;
  let resumePlayback = false;
  quality.addEventListener('change', () => {
    if (player.readyState >= 1) {
      pendingTime = player.currentTime;
      resumePlayback = !player.paused && !player.ended;
    }
    player.src = quality.value;
    player.preload = 'metadata';
    player.load();
  });
  player.addEventListener('loadedmetadata', () => {
    if (pendingTime > 0) player.currentTime = Math.min(pendingTime, player.duration);
    if (resumePlayback) player.play().catch(() => {});
    pendingTime = 0;
    resumePlayback = false;
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
