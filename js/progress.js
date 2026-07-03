// ============================================================
// Progress Tracking — UI updates for sidebar + progress bar
// ============================================================

import { getActiveProfile, getOverallProgress, getOverallScore } from './profile.js';

export function updateProgressBar() {
  const { percentage } = getOverallProgress();
  const fill = document.getElementById('progress-bar-fill');
  const text = document.getElementById('progress-bar-text');
  if (fill) fill.style.width = percentage + '%';
  if (text) text.textContent = percentage + '%';
}

export function updateSidebarChecks() {
  const profile = getActiveProfile();
  const completed = profile ? (profile.sections_completed || []) : [];
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    const section = item.dataset.section;
    if (completed.includes(section)) {
      item.classList.add('completed');
    } else {
      item.classList.remove('completed');
    }
  });
}

export function updateProfileWidget() {
  const profile = getActiveProfile();
  const nameEl = document.getElementById('profile-name');
  const scoreEl = document.getElementById('profile-score');

  if (profile) {
    nameEl.textContent = profile.name;
    const score = getOverallScore();
    if (score !== null) {
      scoreEl.textContent = score + '%';
    } else {
      scoreEl.textContent = '';
    }
  } else {
    nameEl.textContent = 'Sin perfil';
    scoreEl.textContent = '';
  }
}

export function updateAllUI() {
  updateProgressBar();
  updateSidebarChecks();
  updateProfileWidget();
}
