import { exportData, rebalanceRemaining } from '../utils.js';
import { updateAllConstraintsDropdowns } from './constraints.js';

export function rebalanceDoEDOM() {
    const inputs = Array.from(document.querySelectorAll('.doe-o-importance'));
    if (inputs.length === 0) return;
    const share = (100 / inputs.length).toFixed(1);
    inputs.forEach(input => input.value = share);
}

export function attachDeleteEvents() {
    document.querySelectorAll('.delete-row').forEach(btn => {
        btn.onclick = () => {
            const tr         = btn.closest('tr');
            const nameInput  = tr.querySelector('.doe-o-name, .doe-f-name');
            const isDoeRow   = tr.querySelector('.doe-o-name, .doe-f-name, .doe-o-type, .doe-f-type');

            if (nameInput && !isDoeRow) {
                const name = nameInput.value;
                if (window.objectiveConfigs[name]) {
                    delete window.objectiveConfigs[name];
                    // We import rebalanceObjectiveConfigs dynamically if needed, or get it from window
                    if (window.rebalanceObjectiveConfigs) window.rebalanceObjectiveConfigs();
                }
            }
            tr.remove();

            if (window.currentModule === 'doe') {
                rebalanceDoEDOM();
                updateAllConstraintsDropdowns('doe');
            } else {
                if (window.renderSetup) window.renderSetup();
                updateAllConstraintsDropdowns('bo');
            }
        };
    });
}

export async function runDoE() {
    window.currentDoEFeatures = Array.from(document.querySelectorAll('#doe-features-list tr')).map(tr => ({
        name:  tr.querySelector('.doe-f-name').value,
        type:  tr.querySelector('.doe-f-type').value,
        range: tr.querySelector('.doe-f-range').value
    }));

    window.currentDoEObjectives = Array.from(document.querySelectorAll('#doe-objectives-list tr')).map(tr => ({
        name: tr.querySelector('.doe-o-name').value
    })).filter(o => o.name.trim() !== "");

    const numericalConstraints = Array.from(document.querySelectorAll('#doe-numerical-constraints-list tr.constraint-row'))
        .map(tr => tr.querySelector('.constraint-equation').value.trim())
        .filter(eq => eq !== "");

    const disallowedCombinations = Array.from(document.querySelectorAll('#doe-disallowed-constraints-list tr.disallowed-row'))
        .map(tr => {
            const conditions = Array.from(tr.querySelectorAll('.condition-item')).map(item => ({
                feature: item.querySelector('.condition-feature').value,
                value: item.querySelector('.condition-value').value
            }));
            return conditions;
        })
        .filter(conds => conds.length > 0);

    const tweaks = {
        model:    document.getElementById('doe-model-select').value,
        max_runs: document.getElementById('doe-max-runs').value,
        constraints: {
            numerical: numericalConstraints,
            disallowed: disallowedCombinations
        }
    };

    document.getElementById('loading-text').textContent = "Generating designs...";
    document.getElementById('loading-overlay').classList.remove('hidden');

    window.currentAbortController = new AbortController();
    try {
        const res    = await fetch('/doe', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ features: window.currentDoEFeatures, tweaks }),
            signal: window.currentAbortController.signal
        });
        const result = await res.json();
        document.getElementById('loading-overlay').classList.add('hidden');
        if (result.error) return alert(result.error);

        window.suggestionsData = result.suggestions;
        renderDoEResults(result.metrics);
    } catch (err) {
        document.getElementById('loading-overlay').classList.add('hidden');
        if (err.name === 'AbortError') return console.log('DoE calculation aborted by user.');
        alert("DoE Error: " + err.message);
    } finally {
        window.currentAbortController = null;
    }
}

export function renderDoEResults(metrics) {
    const section   = document.getElementById('doe-results-section');
    const dashboard = document.getElementById('doe-quality-dashboard');
    const head      = document.getElementById('doe-results-head');
    const body      = document.getElementById('doe-results-body');

    const warningBanner = document.getElementById('doe-warning-banner');
    if (warningBanner) {
        if (metrics && (metrics.runs <= 1 || metrics.resolution === 'Error' || metrics.orthogonality === 0 || metrics.efficiency === 0)) {
            warningBanner.innerHTML = `
                <div style="background: #fff1f2; border: 1px solid #fecdd3; color: #9f1239; padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem; font-size: 0.9rem; line-height: 1.5; display: flex; gap: 0.75rem; align-items: flex-start; text-align: left;">
                    <span style="font-size: 1.25rem;">⚠️</span>
                    <div>
                        <strong>Low design quality or calculation errors detected:</strong>
                        <p style="margin: 0.25rem 0 0 0;">It is suggested to increase the "Max number of experiments" or modify/remove constraints.</p>
                    </div>
                </div>
            `;
            warningBanner.classList.remove('hidden');
        } else {
            warningBanner.classList.add('hidden');
            warningBanner.innerHTML = '';
        }
    }

    if (metrics && typeof metrics === 'object') {
        const getFlag = (val, type) => {
            if (type === 'ortho') {
                if (val >= 90) return { text: 'Good',      class: 'flag-good' };
                if (val >= 70) return { text: 'Limited',   class: 'flag-limited' };
                return           { text: 'Poor',       class: 'flag-poor' };
            }
            if (type === 'eff') {
                if (val >= 80) return { text: 'Good',      class: 'flag-good' };
                if (val >= 50) return { text: 'Limited',   class: 'flag-limited' };
                return           { text: 'Poor',       class: 'flag-poor' };
            }
            if (type === 'res') {
                if (val.includes('V'))  return { text: 'Good',      class: 'flag-good' };
                if (val.includes('IV')) return { text: 'Limited',   class: 'flag-limited' };
                return                         { text: 'Screening', class: 'flag-poor' };
            }
            if (type === 'curve') {
                if (val === 'Excellent') return { text: 'Good',    class: 'flag-good' };
                if (val === 'Partial')   return { text: 'Limited', class: 'flag-limited' };
                return                          { text: 'None',    class: 'flag-poor' };
            }
            return { text: 'N/A', class: 'flag-limited' };
        };

        const orthoFlag = getFlag(metrics.orthogonality, 'ortho');
        const effFlag   = getFlag(metrics.efficiency,    'eff');
        const resFlag   = getFlag(metrics.resolution,    'res');
        const curveFlag = getFlag(metrics.curvature,     'curve');

        dashboard.innerHTML = `
            <div class="quality-card">
                <span class="metric-label">Orthogonality</span>
                <span class="metric-value">${metrics.orthogonality}%</span>
                <span class="quality-flag ${orthoFlag.class}">${orthoFlag.text}</span>
            </div>
            <div class="quality-card">
                <span class="metric-label">D-Efficiency</span>
                <span class="metric-value">${metrics.efficiency}%</span>
                <span class="quality-flag ${effFlag.class}">${effFlag.text}</span>
            </div>
            <div class="quality-card">
                <span class="metric-label">Resolution</span>
                <span class="metric-value">${metrics.resolution}</span>
                <span class="quality-flag ${resFlag.class}">${resFlag.text}</span>
            </div>
            <div class="quality-card">
                <span class="metric-label">Curvature</span>
                <span class="metric-value">${metrics.curvature}</span>
                <span class="quality-flag ${curveFlag.class}">${curveFlag.text}</span>
            </div>
        `;
        dashboard.classList.remove('hidden');
    }

    const feats = Object.keys(window.suggestionsData[0]);
    const objHeaders = window.currentDoEObjectives.length > 0
        ? window.currentDoEObjectives.map(o => `<th>${o.name}</th>`).join('')
        : '<th>Target(s)</th>';

    head.innerHTML = '<tr>' + feats.map(f => `<th>${f}</th>`).join('') + objHeaders + '</tr>';

    body.innerHTML = window.suggestionsData.map(row => {
        const featCells = feats.map(f =>
            `<td contenteditable="true" class="feat-input-cell editable-cell" style="background:#f8fafc; border: 1px dashed #94a3b8;">${row[f]}</td>`
        ).join('');
        const objCells = window.currentDoEObjectives.length > 0
            ? window.currentDoEObjectives.map(() =>
                `<td contenteditable="true" class="obj-input-cell editable-cell" style="background:#f0fdf4; border: 1px dashed #22c55e;"></td>`
              ).join('')
            : `<td contenteditable="true" class="obj-input-cell editable-cell" style="background:#f0fdf4; border: 1px dashed #22c55e;"></td>`;

        return '<tr>' + featCells + objCells + '</tr>';
    }).join('');

    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth' });

    const checkDoECommitReadiness = () => {
        const commitBtn = document.getElementById('doe-commit-suggestions-btn');
        const objCells  = body.querySelectorAll('.obj-input-cell');
        let allFilled   = true;
        objCells.forEach(td => {
            const val = td.textContent.trim();
            if (val === '' || isNaN(parseFloat(val))) allFilled = false;
        });
        if (commitBtn) {
            commitBtn.disabled       = !allFilled;
            commitBtn.style.opacity  = allFilled ? '1' : '0.5';
            commitBtn.style.cursor   = allFilled ? 'pointer' : 'not-allowed';
        }
    };

    body.querySelectorAll('.obj-input-cell, .feat-input-cell').forEach(cell => {
        cell.addEventListener('input', checkDoECommitReadiness);
        cell.addEventListener('blur',  checkDoECommitReadiness);
    });
    checkDoECommitReadiness();
}

// Bind to window for inline calls
window.rebalanceDoEDOM = rebalanceDoEDOM;
window.attachDeleteEvents = attachDeleteEvents;
