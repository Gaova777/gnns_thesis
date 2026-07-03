// ============================================================
// Main Application — Routing, navigation, profile modal
// ============================================================

import {
  getAllProfiles, getActiveProfile, createProfile, deleteProfile,
  switchProfile, exportProfile, importProfile, setActiveProfileId
} from './profile.js';
import { updateAllUI } from './progress.js';
import { renderQuiz } from './quiz.js';

// Section loaders
import { getSection as s01 } from './sections/s01-graphs.js';
import { getSection as s02 } from './sections/s02-gnn-intro.js';
import { getSection as s03 } from './sections/s03-architectures.js';
import { getSection as s04 } from './sections/s04-elliptic.js';
import { getSection as s05 } from './sections/s05-imbalance.js';
import { getSection as s06 } from './sections/s06-balancing.js';
import { getSection as s07 } from './sections/s07-explainability.js';
import { getSection as s08 } from './sections/s08-stability.js';
import { getSection as s09 } from './sections/s09-methodology.js';
import { getSection as s10 } from './sections/s10-results.js';
import { getSection as s11 } from './sections/s11-anteproyecto.js';
import { getSection as s12 } from './sections/s12-analysis.js';
import { getSection as s13 } from './sections/s13-discussion.js';

const SECTIONS = [
  'welcome',
  's01-graphs', 's02-gnn-intro', 's03-architectures', 's04-elliptic',
  's05-imbalance', 's06-balancing', 's07-explainability', 's08-stability',
  's09-methodology', 's10-results',
  's11-anteproyecto', 's12-analysis', 's13-discussion'
];

const SECTION_LOADERS = {
  's01-graphs': s01,
  's02-gnn-intro': s02,
  's03-architectures': s03,
  's04-elliptic': s04,
  's05-imbalance': s05,
  's06-balancing': s06,
  's07-explainability': s07,
  's08-stability': s08,
  's09-methodology': s09,
  's10-results': s10,
  's11-anteproyecto': s11,
  's12-analysis': s12,
  's13-discussion': s13
};

let currentSection = 'welcome';

// ---- Navigation ----
function navigateTo(sectionId) {
  currentSection = sectionId;

  // Update active nav
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.section === sectionId);
  });

  // Update profile current section
  const profile = getActiveProfile();
  if (profile) {
    profile.current_section = sectionId;
    import('./profile.js').then(m => m.updateProfile(profile));
  }

  // Close mobile sidebar
  document.getElementById('sidebar')?.classList.remove('open');
  document.getElementById('hamburger')?.classList.remove('active');
  document.getElementById('sidebar-overlay')?.classList.remove('active');

  renderSection(sectionId);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function renderSection(sectionId) {
  const content = document.getElementById('content-inner');

  if (sectionId === 'welcome') {
    content.innerHTML = getWelcomeHTML();
    const startBtn = document.getElementById('start-btn');
    if (startBtn) {
      startBtn.addEventListener('click', () => navigateTo('s01-graphs'));
    }
    return;
  }

  const loader = SECTION_LOADERS[sectionId];
  if (!loader) {
    content.innerHTML = '<p>Seccion no encontrada.</p>';
    return;
  }

  const { html, quiz } = loader();

  // Build nav buttons
  const idx = SECTIONS.indexOf(sectionId);
  const prevSection = idx > 0 ? SECTIONS[idx - 1] : null;
  const nextSection = idx < SECTIONS.length - 1 ? SECTIONS[idx + 1] : null;

  let fullHTML = `<div class="section-content">${html}`;
  fullHTML += `<div id="quiz-mount"></div>`;
  fullHTML += `<div class="section-nav">`;
  if (prevSection) {
    fullHTML += `<button class="btn btn-secondary" data-nav="${prevSection}">&#8592; Anterior</button>`;
  } else {
    fullHTML += `<span></span>`;
  }
  if (nextSection) {
    fullHTML += `<button class="btn btn-primary" data-nav="${nextSection}">Siguiente &#8594;</button>`;
  } else {
    fullHTML += `<span></span>`;
  }
  fullHTML += `</div></div>`;

  content.innerHTML = fullHTML;

  // Mount quiz
  if (quiz && quiz.length > 0) {
    const quizMount = document.getElementById('quiz-mount');
    if (quizMount) {
      renderQuiz(sectionId, quiz, quizMount, () => {
        if (nextSection) navigateTo(nextSection);
      });
    }
  }

  // Bind nav buttons
  content.querySelectorAll('[data-nav]').forEach(btn => {
    btn.addEventListener('click', () => navigateTo(btn.dataset.nav));
  });

  // Trigger stagger animations
  requestAnimationFrame(() => {
    content.querySelectorAll('.imbalance-ball').forEach((ball, i) => {
      ball.style.animationDelay = (i * 0.02) + 's';
    });
  });
}

function getWelcomeHTML() {
  const profile = getActiveProfile();
  return `
    <div class="welcome-hero">
      <h1>Estabilidad de XAI en GNNs para Deteccion de Lavado de Dinero</h1>
      <p>Una guia completa e interactiva para entender esta tesis de maestria, desde grafos hasta los resultados finales.</p>
      ${!profile ? `
        <div class="info-box important">
          <p class="info-title">Primero, crea tu perfil</p>
          <p>Hace clic en el icono de engranaje arriba a la derecha para crear un perfil. Asi se guarda tu progreso.</p>
        </div>
      ` : `
        <p style="color:var(--c-success); font-weight:600;">Bienvenido/a de vuelta, ${profile.name}!</p>
      `}
      <div class="welcome-stats">
        <div class="welcome-stat">
          <div class="welcome-stat-num">10</div>
          <div class="welcome-stat-label">Secciones</div>
        </div>
        <div class="welcome-stat">
          <div class="welcome-stat-num">45</div>
          <div class="welcome-stat-label">Preguntas de quiz</div>
        </div>
        <div class="welcome-stat">
          <div class="welcome-stat-num">4</div>
          <div class="welcome-stat-label">Arquitecturas GNN</div>
        </div>
        <div class="welcome-stat">
          <div class="welcome-stat-num">3</div>
          <div class="welcome-stat-label">Metodos XAI</div>
        </div>
      </div>
      <h2 style="color:var(--text-heading); margin-bottom:16px;">Que vas a aprender?</h2>
      <div style="text-align:left; max-width:600px; margin:0 auto;">
        <ol style="line-height:2;">
          <li><strong>Grafos</strong> — que son y por que modelan transacciones de Bitcoin</li>
          <li><strong>Graph Neural Networks</strong> — como aprenden sobre grafos</li>
          <li><strong>4 Arquitecturas</strong> — GCN, GraphSAGE, GAT, TAGCN</li>
          <li><strong>Elliptic Dataset</strong> — datos reales de Bitcoin para detectar fraude</li>
          <li><strong>Desbalance de clases</strong> — el problema central del fraude</li>
          <li><strong>Tecnicas de balance</strong> — class weighting, focal loss</li>
          <li><strong>Explicabilidad (XAI)</strong> — por que las decisiones, no solo que decide</li>
          <li><strong>Estabilidad</strong> — la pregunta de investigacion central</li>
          <li><strong>Metodologia</strong> — como se diseñaron los experimentos</li>
          <li><strong>Resultados</strong> — que encontramos y por que importa</li>
        </ol>
      </div>
      <button class="btn btn-primary" id="start-btn" style="margin-top:32px; font-size:1.05rem; padding:14px 40px;">
        Empezar el recorrido &#8594;
      </button>
    </div>
  `;
}

// ---- Profile Modal ----
function renderProfileModal() {
  const body = document.getElementById('profile-modal-body');
  const profiles = getAllProfiles();
  const active = getActiveProfile();

  let html = '';

  // Create new profile
  html += `<div class="profile-section-divider">Crear nuevo perfil</div>`;
  html += `<div class="profile-form">
    <input type="text" id="new-profile-name" placeholder="Tu nombre..." maxlength="50">
    <button class="btn btn-primary btn-sm btn-block" id="create-profile-btn">Crear perfil</button>
  </div>`;

  // Existing profiles
  if (profiles.length > 0) {
    html += `<div class="profile-section-divider">Perfiles existentes</div>`;
    html += `<div class="profile-list">`;
    profiles.forEach(p => {
      const isActive = active && active.id === p.id;
      const completedCount = p.sections_completed?.length || 0;
      html += `<div class="profile-list-item ${isActive ? 'active' : ''}" data-pid="${p.id}">
        <div>
          <div class="profile-item-name">${p.name} ${isActive ? '(activo)' : ''}</div>
          <div class="profile-item-date">${completedCount}/14 secciones | Creado: ${new Date(p.created).toLocaleDateString('es')}</div>
        </div>
        <div class="profile-item-actions">
          ${!isActive ? `<button data-switch="${p.id}" title="Activar">&#9654;</button>` : ''}
          <button data-export="${p.id}" title="Exportar">&#11015;</button>
          <button data-delete="${p.id}" title="Eliminar">&#10006;</button>
        </div>
      </div>`;
    });
    html += `</div>`;
  }

  // Import
  html += `<div class="profile-section-divider">Importar perfil</div>`;
  html += `<div style="text-align:center">
    <input type="file" id="import-file" accept=".json" style="display:none">
    <button class="btn btn-secondary btn-sm" id="import-profile-btn">Importar desde JSON</button>
  </div>`;

  body.innerHTML = html;

  // Bind events
  document.getElementById('create-profile-btn')?.addEventListener('click', () => {
    const input = document.getElementById('new-profile-name');
    const name = input?.value?.trim();
    if (name) {
      createProfile(name);
      updateAllUI();
      renderProfileModal();
      renderSection(currentSection);
    }
  });

  document.getElementById('new-profile-name')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') document.getElementById('create-profile-btn')?.click();
  });

  body.querySelectorAll('[data-switch]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      switchProfile(btn.dataset.switch);
      updateAllUI();
      renderProfileModal();
      renderSection(currentSection);
    });
  });

  body.querySelectorAll('[data-export]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const p = profiles.find(pr => pr.id === btn.dataset.export);
      if (p) exportProfile(p);
    });
  });

  body.querySelectorAll('[data-delete]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (confirm('Seguro que queres eliminar este perfil?')) {
        deleteProfile(btn.dataset.delete);
        updateAllUI();
        renderProfileModal();
        renderSection(currentSection);
      }
    });
  });

  document.getElementById('import-profile-btn')?.addEventListener('click', () => {
    document.getElementById('import-file')?.click();
  });

  document.getElementById('import-file')?.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      try {
        await importProfile(file);
        updateAllUI();
        renderProfileModal();
        renderSection(currentSection);
      } catch (err) {
        alert(err.message);
      }
    }
  });
}

// ---- Theme Toggle ----
function initTheme() {
  const saved = localStorage.getItem('xai_theme');
  if (saved) {
    document.documentElement.setAttribute('data-theme', saved);
  }
  document.getElementById('theme-toggle')?.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('xai_theme', next);
  });
}

// ---- Init ----
function init() {
  initTheme();
  updateAllUI();

  // Sidebar navigation
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => navigateTo(item.dataset.section));
  });

  // Profile modal
  document.getElementById('profile-btn')?.addEventListener('click', () => {
    document.getElementById('profile-modal')?.classList.remove('hidden');
    renderProfileModal();
  });
  document.querySelector('.modal-close')?.addEventListener('click', () => {
    document.getElementById('profile-modal')?.classList.add('hidden');
  });
  document.querySelector('.modal-backdrop')?.addEventListener('click', () => {
    document.getElementById('profile-modal')?.classList.add('hidden');
  });

  // Hamburger
  document.getElementById('hamburger')?.addEventListener('click', () => {
    const sidebar = document.getElementById('sidebar');
    const hamburger = document.getElementById('hamburger');
    const overlay = document.getElementById('sidebar-overlay');
    sidebar?.classList.toggle('open');
    hamburger?.classList.toggle('active');
    overlay?.classList.toggle('active');
  });
  document.getElementById('sidebar-overlay')?.addEventListener('click', () => {
    document.getElementById('sidebar')?.classList.remove('open');
    document.getElementById('hamburger')?.classList.remove('active');
    document.getElementById('sidebar-overlay')?.classList.remove('active');
  });

  // Load current section from profile
  const profile = getActiveProfile();
  if (profile && profile.current_section && SECTIONS.includes(profile.current_section)) {
    navigateTo(profile.current_section);
  } else {
    navigateTo('welcome');
  }
}

document.addEventListener('DOMContentLoaded', init);
