// Play the acronym heartbeat once when the summary first enters view.
const acronymHeading = document.getElementById('tldr-heading');
if (acronymHeading) {
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        acronymHeading.classList.add('acronym-revealed');
        observer.disconnect();
      }
    }, { threshold: 0.8 });
    observer.observe(acronymHeading);
  } else {
    acronymHeading.classList.add('acronym-revealed');
  }
}

// Use normalized source coordinates to magnify matching details in every panel.
document.querySelectorAll('.tldr-comparison, .synchronized-comparison').forEach((comparison) => {
  const panels = [...comparison.querySelectorAll('.comparison-cell img')].map((img) => {
    const frame = document.createElement('div');
    frame.className = 'zoom-frame';
    img.before(frame);
    frame.append(img);
    const sizeFrame = () => { if (img.naturalHeight) frame.style.setProperty('--image-ratio', img.naturalWidth / img.naturalHeight); };
    img.addEventListener('load', sizeFrame);
    sizeFrame();
    const lens = document.createElement('div');
    lens.className = 'zoom-lens';
    lens.setAttribute('aria-hidden', 'true');
    lens.style.backgroundImage = `url("${img.getAttribute('src')}")`;
    frame.append(lens);
    return { frame, lens };
  });
  const input = panels[0]?.frame;
  if (input) {
    let point = { x: 0.5, y: 0.5 };
    let touchOpen = false;
    const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
    function show(x, y) {
      point = { x: clamp(x, 0, 1), y: clamp(y, 0, 1) };
      comparison.classList.add('zoom-active');
      for (const { frame, lens } of panels) {
        const w = frame.clientWidth, h = frame.clientHeight;
        const lw = lens.offsetWidth, lh = lens.offsetHeight;
        lens.style.left = `${clamp(point.x * w - lw / 2, 0, w - lw)}px`;
        lens.style.top = `${clamp(point.y * h - lh / 2, 0, h - lh)}px`;
        lens.style.backgroundSize = `${w * 3}px ${h * 3}px`;
        lens.style.backgroundPosition = `${-clamp(point.x * w * 3 - lw / 2, 0, w * 3 - lw)}px ${-clamp(point.y * h * 3 - lh / 2, 0, h * 3 - lh)}px`;
      }
    }
    function hide() { comparison.classList.remove('zoom-active'); touchOpen = false; }
    function atPointer(event) {
      const rect = event.currentTarget.getBoundingClientRect();
      show((event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height);
    }
    const controls = comparison.matches('.synchronized-comparison') ? panels.map(panel => panel.frame) : [input];
    controls.forEach((input) => {
    input.classList.add('zoom-input');
    input.tabIndex = 0;
    input.setAttribute('role', 'group');
    input.setAttribute('aria-label', 'Synchronized image zoom. Use arrow keys to explore; Escape to close.');
    input.addEventListener('pointermove', (event) => { if (event.pointerType !== 'touch') atPointer(event); });
    input.addEventListener('pointerleave', (event) => { if (event.pointerType !== 'touch') hide(); });
    input.addEventListener('pointerdown', (event) => {
      if (event.pointerType === 'touch') {
        if (touchOpen) hide();
        else { touchOpen = true; atPointer(event); }
      }
    });
    input.addEventListener('focus', () => { if (input.matches(':focus-visible')) show(0.5, 0.5); });
    input.addEventListener('blur', hide);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') hide();
    }, true);
    input.addEventListener('keydown', (event) => {
      const steps = { ArrowLeft: [-0.04, 0], ArrowRight: [0.04, 0], ArrowUp: [0, -0.04], ArrowDown: [0, 0.04] };
      if (event.key === 'Escape') { event.preventDefault(); hide(); }
      else if (steps[event.key]) {
        event.preventDefault();
        const [dx, dy] = steps[event.key]; show(point.x + dx, point.y + dy);
      }
    });
    });
    window.addEventListener('resize', () => { if (comparison.classList.contains('zoom-active')) show(point.x, point.y); });
  }
});

// Non-modal reference popovers: hover links, or use the explicit info buttons.
(() => {
  let active = null;
  let timer;
  function hide(restoreFocus = false) {
    clearTimeout(timer);
    if (!active) return;
    const entry = active;
    active = null;
    entry.popup.hidden = true;
    entry.button.setAttribute('aria-expanded', 'false');
    if (restoreFocus) entry.button.focus();
  }
  function scheduleHide(entry) {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (active === entry && !entry.pinned && !entry.popup.contains(document.activeElement)) hide();
    }, 180);
  }
  document.querySelectorAll('.paper-link').forEach((link) => {
    const popup = document.getElementById(link.getAttribute('aria-describedby'));
    if (!popup) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'reference-info';
    button.textContent = 'i';
    button.setAttribute('aria-label', `Reference details for ${link.textContent.trim()}`);
    button.setAttribute('aria-controls', popup.id);
    button.setAttribute('aria-expanded', 'false');
    button.setAttribute('aria-haspopup', 'dialog');
    link.after(button);
    popup.setAttribute('role', 'dialog');
    popup.setAttribute('aria-label', `Reference: ${link.textContent.trim()}`);
    popup.tabIndex = -1;
    const close = document.createElement('button');
    close.type = 'button'; close.className = 'reference-close'; close.textContent = '×';
    close.setAttribute('aria-label', 'Close reference');
    popup.prepend(close);
    const entry = { link, popup, button, pinned: false };
    function show(pinned = false) {
      hide(); active = entry; entry.pinned = pinned;
      popup.hidden = false; button.setAttribute('aria-expanded', 'true');
      const r = (pinned ? button : link).getBoundingClientRect();
      const w = popup.offsetWidth, h = popup.offsetHeight;
      popup.style.left = `${Math.max(12, Math.min(r.left, window.innerWidth - w - 12))}px`;
      const top = r.top - h - 10 >= 12 ? r.top - h - 10 : Math.min(r.bottom + 10, window.innerHeight - h - 12);
      popup.style.top = `${Math.max(12, top)}px`;
      if (pinned) popup.focus({ preventScroll: true });
    }
    button.addEventListener('click', () => {
      if (active === entry && entry.pinned) hide(); else show(true);
    });
    close.addEventListener('click', () => hide(true));
    link.addEventListener('mouseenter', () => { if (!active?.pinned) show(); });
    link.addEventListener('mouseleave', () => scheduleHide(entry));
    link.addEventListener('focus', () => { if (link.matches(':focus-visible')) show(); });
    link.addEventListener('click', () => hide());
    popup.addEventListener('mouseenter', () => { if (active === entry) clearTimeout(timer); });
    popup.addEventListener('mouseleave', () => scheduleHide(entry));
  });
  function outside(target) {
    return active && !active.popup.contains(target) && !active.link.contains(target) && !active.button.contains(target);
  }
  document.addEventListener('pointerdown', (event) => { if (outside(event.target)) hide(); }, true);
  document.addEventListener('focusin', (event) => { if (outside(event.target)) hide(); });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && active) { event.preventDefault(); hide(true); }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) hide(); });
  window.addEventListener('blur', () => hide());
  window.addEventListener('scroll', () => hide(), { passive: true });
  window.addEventListener('resize', () => hide());
})();

// Manual carousel: keep corresponding methods together for each scene.
const gallery = document.querySelector('.results-carousel');
if (gallery) {
  const slides = [...gallery.querySelectorAll('.carousel-slide')];
  const filter = document.getElementById('comparison-dataset');
  const status = document.getElementById('comparison-status');
  const positions = new Map();
  const datasetSlides = () => slides.filter(slide => slide.dataset.dataset === filter.value).slice(0, 5);
  let selected = datasetSlides(), index = 0;
  function render() {
    slides.forEach(slide => { slide.hidden = true; slide.classList.remove('zoom-active'); });
    selected[index].hidden = false;
    status.textContent = `${index + 1} / ${selected.length}`;
    document.getElementById('comparison-prev').disabled = selected.length < 2;
    document.getElementById('comparison-next').disabled = selected.length < 2;
  }
  function move(delta) { index = (index + delta + selected.length) % selected.length; positions.set(filter.value, index); render(); }
  document.getElementById('comparison-prev').addEventListener('click', () => move(-1));
  document.getElementById('comparison-next').addEventListener('click', () => move(1));
  filter.addEventListener('change', () => {
    selected = datasetSlides();
    index = positions.get(filter.value) || 0; render();
  });
  render();
}

// Reveal each major section once, without changing scrolling or anchor navigation.
(() => {
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (motion.matches || !('IntersectionObserver' in window)) return;
  const sections = [...document.querySelectorAll('#overview, #abstract, #method, #visual-results, #results, #bibtex')];
  const reveal = (section) => {
    section.classList.remove('section-reveal-pending');
    observer.unobserve(section);
  };
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => { if (entry.isIntersecting) reveal(entry.target); });
  }, { threshold: 0, rootMargin: '0px 0px -40px 0px' });
  sections.forEach((section) => {
    // Keep sections already reached (including direct hash links) visible.
    if (section.getBoundingClientRect().top < window.innerHeight - 40) return;
    section.classList.add('section-reveal', 'section-reveal-pending');
    observer.observe(section);
    section.addEventListener('focusin', () => reveal(section));
  });
  motion.addEventListener('change', (event) => {
    if (event.matches) {
      sections.forEach(reveal);
      observer.disconnect();
    }
  });
})();

// Copy the citation; select it for manual copying if clipboard access is unavailable.
const copyBibtex = document.getElementById('copy-bibtex');
copyBibtex?.addEventListener('click', async () => {
  const code = document.querySelector('.bibtex code');
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(code.textContent);
    status.textContent = 'BibTeX copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(code);
    const selection = window.getSelection();
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
