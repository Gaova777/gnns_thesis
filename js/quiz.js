// ============================================================
// Quiz Engine — Render, score, save
// ============================================================

import { saveQuizScore, markSectionCompleted, getActiveProfile } from './profile.js';
import { updateAllUI } from './progress.js';

const PASS_THRESHOLD = 60;

/**
 * Render a quiz inside a container element.
 * @param {string} sectionId - e.g. 's01-graphs'
 * @param {Array} questions - [{q, options: [string], correct: number, explanation: string}]
 * @param {HTMLElement} container - DOM element to render into
 * @param {Function} onComplete - callback when quiz is submitted
 */
export function renderQuiz(sectionId, questions, container, onComplete) {
  const profile = getActiveProfile();
  const previousScore = profile?.quiz_scores?.[sectionId];

  let userAnswers = new Array(questions.length).fill(-1);
  let submitted = false;

  function render() {
    const letters = ['A', 'B', 'C', 'D', 'E'];
    let html = `<div class="quiz-container" id="quiz-${sectionId}">`;
    html += `<div class="quiz-header">
      <span class="quiz-title">Quiz de la Seccion</span>`;
    if (previousScore) {
      const cls = previousScore.percentage >= PASS_THRESHOLD ? 'passed' : 'failed';
      html += `<span class="quiz-score-badge ${cls}">Anterior: ${previousScore.percentage}%</span>`;
    }
    html += `</div>`;

    questions.forEach((q, qi) => {
      html += `<div class="quiz-question" data-qi="${qi}">`;
      html += `<p class="quiz-q-text"><span class="quiz-q-num">${qi + 1}.</span> ${q.q}</p>`;
      html += `<div class="quiz-options">`;
      q.options.forEach((opt, oi) => {
        const selected = userAnswers[qi] === oi ? 'selected' : '';
        const disabled = submitted ? 'disabled' : '';
        let extraClass = '';
        if (submitted) {
          if (oi === q.correct) extraClass = 'correct';
          else if (userAnswers[qi] === oi && oi !== q.correct) extraClass = 'incorrect';
        }
        html += `<div class="quiz-option ${selected} ${extraClass} ${disabled}" data-qi="${qi}" data-oi="${oi}">
          <span class="option-letter">${letters[oi]}</span>
          <span>${opt}</span>
        </div>`;
      });
      html += `</div>`;

      if (submitted) {
        const isCorrect = userAnswers[qi] === q.correct;
        const fbClass = isCorrect ? 'correct-fb' : 'incorrect-fb';
        html += `<div class="quiz-feedback show ${fbClass}">
          ${isCorrect ? '&#10003; Correcto!' : '&#10007; Incorrecto.'} ${q.explanation}
        </div>`;
      }
      html += `</div>`;
    });

    // Actions
    html += `<div class="quiz-actions">`;
    if (!submitted) {
      const allAnswered = userAnswers.every(a => a >= 0);
      html += `<button class="btn btn-primary" id="quiz-submit" ${allAnswered ? '' : 'disabled'}>Verificar respuestas</button>`;
    } else {
      const score = userAnswers.reduce((s, a, i) => s + (a === questions[i].correct ? 1 : 0), 0);
      const pct = Math.round((score / questions.length) * 100);
      const passed = pct >= PASS_THRESHOLD;
      html += `<div style="text-align:center; width:100%;">`;
      html += `<p style="font-size:1.1rem; font-weight:700; margin-bottom:8px; color:${passed ? 'var(--c-success)' : 'var(--c-error)'}">
        ${score}/${questions.length} — ${pct}% ${passed ? '&#10003; Aprobado!' : '&#10007; No aprobado (minimo 60%)'}</p>`;
      if (!passed) {
        html += `<button class="btn btn-secondary" id="quiz-retry">Reintentar</button>`;
      } else {
        html += `<button class="btn btn-success" id="quiz-continue">Continuar &#8594;</button>`;
      }
      html += `</div>`;
    }
    html += `</div></div>`;

    container.innerHTML = html;

    // Bind events
    if (!submitted) {
      container.querySelectorAll('.quiz-option:not(.disabled)').forEach(el => {
        el.addEventListener('click', () => {
          const qi = parseInt(el.dataset.qi);
          const oi = parseInt(el.dataset.oi);
          userAnswers[qi] = oi;
          render();
        });
      });
      const submitBtn = container.querySelector('#quiz-submit');
      if (submitBtn) {
        submitBtn.addEventListener('click', () => {
          submitted = true;
          const score = userAnswers.reduce((s, a, i) => s + (a === questions[i].correct ? 1 : 0), 0);
          const pct = Math.round((score / questions.length) * 100);
          saveQuizScore(sectionId, score, questions.length);
          if (pct >= PASS_THRESHOLD) {
            markSectionCompleted(sectionId);
          }
          updateAllUI();
          render();
          // Scroll to results
          container.querySelector('.quiz-actions')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
      }
    } else {
      const retryBtn = container.querySelector('#quiz-retry');
      if (retryBtn) {
        retryBtn.addEventListener('click', () => {
          userAnswers = new Array(questions.length).fill(-1);
          submitted = false;
          render();
        });
      }
      const continueBtn = container.querySelector('#quiz-continue');
      if (continueBtn) {
        continueBtn.addEventListener('click', () => {
          if (onComplete) onComplete();
        });
      }
    }
  }

  render();
}
