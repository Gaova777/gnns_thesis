// ============================================================
// Profile Management — localStorage + JSON export/import
// ============================================================

const STORAGE_KEY = 'xai_learning_profiles';
const ACTIVE_KEY = 'xai_learning_active_profile';

function generateId() {
  return 'p_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 7);
}

export function getAllProfiles() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveAllProfiles(profiles) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(profiles));
}

export function getActiveProfileId() {
  return localStorage.getItem(ACTIVE_KEY);
}

export function setActiveProfileId(id) {
  localStorage.setItem(ACTIVE_KEY, id);
}

export function getActiveProfile() {
  const profiles = getAllProfiles();
  const activeId = getActiveProfileId();
  return profiles.find(p => p.id === activeId) || null;
}

export function createProfile(name) {
  const profiles = getAllProfiles();
  const profile = {
    id: generateId(),
    name: name.trim(),
    created: new Date().toISOString(),
    sections_completed: [],
    quiz_scores: {},
    current_section: 'welcome'
  };
  profiles.push(profile);
  saveAllProfiles(profiles);
  setActiveProfileId(profile.id);
  return profile;
}

export function updateProfile(profile) {
  const profiles = getAllProfiles();
  const idx = profiles.findIndex(p => p.id === profile.id);
  if (idx >= 0) {
    profiles[idx] = profile;
    saveAllProfiles(profiles);
  }
}

export function deleteProfile(id) {
  let profiles = getAllProfiles();
  profiles = profiles.filter(p => p.id !== id);
  saveAllProfiles(profiles);
  if (getActiveProfileId() === id) {
    if (profiles.length > 0) {
      setActiveProfileId(profiles[0].id);
    } else {
      localStorage.removeItem(ACTIVE_KEY);
    }
  }
}

export function switchProfile(id) {
  setActiveProfileId(id);
}

export function exportProfile(profile) {
  const blob = new Blob([JSON.stringify(profile, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `xai_profile_${profile.name.replace(/\s+/g, '_')}_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function importProfile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        if (!data.name || !data.id) {
          reject(new Error('Archivo de perfil inválido'));
          return;
        }
        const profiles = getAllProfiles();
        // Check if profile with same id exists
        const existing = profiles.findIndex(p => p.id === data.id);
        if (existing >= 0) {
          // Merge: keep the one with more progress
          const oldCompleted = profiles[existing].sections_completed?.length || 0;
          const newCompleted = data.sections_completed?.length || 0;
          if (newCompleted > oldCompleted) {
            profiles[existing] = data;
          }
        } else {
          profiles.push(data);
        }
        saveAllProfiles(profiles);
        setActiveProfileId(data.id);
        resolve(data);
      } catch (err) {
        reject(new Error('Error al leer el archivo JSON'));
      }
    };
    reader.onerror = () => reject(new Error('Error al leer el archivo'));
    reader.readAsText(file);
  });
}

export function markSectionCompleted(sectionId) {
  const profile = getActiveProfile();
  if (!profile) return;
  if (!profile.sections_completed.includes(sectionId)) {
    profile.sections_completed.push(sectionId);
  }
  profile.current_section = sectionId;
  updateProfile(profile);
}

export function saveQuizScore(sectionId, score, total) {
  const profile = getActiveProfile();
  if (!profile) return;
  if (!profile.quiz_scores) profile.quiz_scores = {};
  profile.quiz_scores[sectionId] = { score, total, percentage: Math.round((score / total) * 100), date: new Date().toISOString() };
  updateProfile(profile);
}

export function getOverallProgress() {
  const profile = getActiveProfile();
  if (!profile) return { completed: 0, total: 14, percentage: 0 };
  const totalSections = 14; // welcome + 13 sections (s01-s13)
  const completed = profile.sections_completed?.length || 0;
  return { completed, total: totalSections, percentage: Math.round((completed / totalSections) * 100) };
}

export function getOverallScore() {
  const profile = getActiveProfile();
  if (!profile || !profile.quiz_scores) return null;
  const scores = Object.values(profile.quiz_scores);
  if (scores.length === 0) return null;
  const avg = scores.reduce((sum, s) => sum + s.percentage, 0) / scores.length;
  return Math.round(avg);
}
