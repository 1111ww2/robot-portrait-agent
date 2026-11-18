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
