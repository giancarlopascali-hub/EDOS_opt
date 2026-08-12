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
                    if (window.rebalanceObjectiveConfigs) window.rebalanceObjectiveConfigs();
                }
            }
            tr.remove();

            if (window.currentModule === 'doe') {
                rebalanceDoEDOM();
                updateAllConstraintsDropdowns('doe');
                // Refresh the min-N hint if D-Optimal is active
                if (document.getElementById('doe-model-select')?.value === 'doptimal') {
                    updateDoptMinNHint();
                }
            } else {
                if (window.renderSetup) window.renderSetup();
                updateAllConstraintsDropdowns('bo');
            }
        };
    });
}

// ---------------------------------------------------------------------------
// Toggle helpers – Max / Exact experiment count
// ---------------------------------------------------------------------------

/**
 * Set the Max/Exact switch state and block/unblock controls.
 */
function setExactRunsMode(isExact, isBlocked = false) {
    const hiddenCheckbox  = document.getElementById('doe-exact-runs-toggle');
    const switchMaxBtn    = document.getElementById('doe-switch-max-btn');
    const switchExactBtn  = document.getElementById('doe-switch-exact-btn');
    const switchContainer = document.getElementById('doe-runs-switch');
    const runsLabel       = document.getElementById('doe-runs-label');

    if (hiddenCheckbox) hiddenCheckbox.checked = isExact;

    if (switchMaxBtn && switchExactBtn && switchContainer) {
        if (isExact) {
            switchMaxBtn.classList.remove('active');
            switchExactBtn.classList.add('active');
            if (runsLabel) runsLabel.textContent = 'Exact Experiments';
        } else {
            switchExactBtn.classList.remove('active');
            switchMaxBtn.classList.add('active');
            if (runsLabel) runsLabel.textContent = 'Max Experiments';
        }

        if (isBlocked) {
            switchMaxBtn.disabled   = true;
            switchExactBtn.disabled = true;
            switchContainer.classList.add('disabled');
            switchContainer.title   = "LHS and D-Optimal require an exact run count";
        } else {
            switchMaxBtn.disabled   = false;
            switchExactBtn.disabled = false;
            switchContainer.classList.remove('disabled');
            switchContainer.title   = "Switch between Max and Exact experiment count";
        }
    }
}

/**
 * Sync the Max/Exact toggle state to the selected model.
 * LHS and D-Optimal are inherently exact-N: force to Exact and block controls.
 * All other (structured) models: reset to Max and re-enable controls.
 */
function syncToggleToModel(modelValue) {
    const isExactOnly = (modelValue === 'lhs' || modelValue === 'doptimal');
    setExactRunsMode(isExactOnly, isExactOnly);
}

// ---------------------------------------------------------------------------
// D-Optimal helpers
// ---------------------------------------------------------------------------

/**
 * Compute and display the minimum-N hint beneath the run-count input.
 * Only shown when D-Optimal is the selected model.
 */
function updateDoptMinNHint() {
    const isDopt  = document.getElementById('doe-model-select')?.value === 'doptimal';
    const hintEl  = document.getElementById('doe-dopt-min-n-hint');
    if (!hintEl) return;
    if (!isDopt) { hintEl.style.display = 'none'; return; }

    const k     = document.querySelectorAll('#doe-features-list tr').length;
    const order = document.getElementById('doe-dopt-order-select')?.value ?? 'quadratic';
    let minN;
    if      (order === 'linear')       minN = k + 1;
    else if (order === 'interaction')  minN = k + Math.floor(k * (k - 1) / 2) + 1;
    else                               minN = Math.floor((k + 1) * (k + 2) / 2);  // quadratic

    hintEl.textContent   = `Min recommended: ${minN} runs for ${k} factor${k !== 1 ? 's' : ''}`;
    hintEl.style.display = 'block';
}

// ---------------------------------------------------------------------------
// Wire up listeners (called once on module init)
// ---------------------------------------------------------------------------

function initDoEListeners() {
    const switchMaxBtn   = document.getElementById('doe-switch-max-btn');
    const switchExactBtn = document.getElementById('doe-switch-exact-btn');

    if (switchMaxBtn) {
        switchMaxBtn.addEventListener('click', () => {
            if (!switchMaxBtn.disabled) {
                setExactRunsMode(false, false);
            }
        });
    }

    if (switchExactBtn) {
        switchExactBtn.addEventListener('click', () => {
            if (!switchExactBtn.disabled) {
                setExactRunsMode(true, false);
            }
        });
    }

    // Model selector: sync toggle state + show/hide D-Optimal panel
    const modelSelect = document.getElementById('doe-model-select');
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            const isDopt = e.target.value === 'doptimal';
            syncToggleToModel(e.target.value);

            const orderItem = document.getElementById('doe-dopt-order-item');
            if (orderItem) orderItem.style.display = isDopt ? '' : 'none';

            updateDoptMinNHint();
        });
    }

    // D-Optimal model order: refresh hint on change
    const orderSelect = document.getElementById('doe-dopt-order-select');
    if (orderSelect) {
        orderSelect.addEventListener('change', updateDoptMinNHint);
    }
}

// Run listener setup as soon as this module is imported
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDoEListeners);
} else {
    initDoEListeners();
}

// ---------------------------------------------------------------------------
// Main DoE execution
// ---------------------------------------------------------------------------

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

    const exactToggle = document.getElementById('doe-exact-runs-toggle');
    const useExact    = exactToggle ? exactToggle.checked : false;

    const tweaks = {
        model:            document.getElementById('doe-model-select').value,
        max_runs:         document.getElementById('doe-max-runs').value,
        use_exact_runs:   useExact,
        dopt_model_order: document.getElementById('doe-dopt-order-select')?.value ?? 'quadratic',
        constraints: {
            numerical:  numericalConstraints,
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

// ---------------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------------

export function renderDoEResults(metrics) {
    const section   = document.getElementById('doe-results-section');
    const dashboard = document.getElementById('doe-quality-dashboard');
    const head      = document.getElementById('doe-results-head');
    const body      = document.getElementById('doe-results-body');

    const warningBanner = document.getElementById('doe-warning-banner');
    if (warningBanner) {
        warningBanner.innerHTML = '';
        warningBanner.classList.add('hidden');

        // Low-quality / error warning (red)
        if (metrics && (metrics.runs <= 1 || metrics.resolution === 'Error' ||
                        metrics.orthogonality === 0 || metrics.efficiency === 0)) {
            warningBanner.innerHTML += `
                <div style="background:#fff1f2; border:1px solid #fecdd3; color:#9f1239;
                            padding:1rem; border-radius:12px; margin-bottom:1rem;
                            font-size:0.9rem; line-height:1.5; display:flex; gap:0.75rem;
                            align-items:flex-start; text-align:left;">
                    <span style="font-size:1.25rem;">⚠️</span>
                    <div>
                        <strong>Low design quality or calculation errors detected:</strong>
                        <p style="margin:0.25rem 0 0 0;">It is suggested to increase the
                        "Max / Exact number of experiments" or modify/remove constraints.</p>
                    </div>
                </div>`;
            warningBanner.classList.remove('hidden');
        }

        // Oversampling warning (amber) – only for structured models in Exact mode
        if (metrics && metrics.oversampling_warning) {
            warningBanner.innerHTML += `
                <div style="background:#fffbeb; border:1px solid #fde68a; color:#92400e;
                            padding:1rem; border-radius:12px; margin-bottom:1rem;
                            font-size:0.9rem; line-height:1.5; display:flex; gap:0.75rem;
                            align-items:flex-start; text-align:left;">
                    <span style="font-size:1.25rem;">⚠️</span>
                    <div>
                        <strong>Oversampling detected:</strong>
                        <p style="margin:0.25rem 0 0 0;">The requested number of experiments
                        exceeds the natural size of this design. Extra rows have been added
                        with small jitter, which may reduce orthogonality and D-efficiency.
                        Consider switching to <em>LHS</em> or <em>D-Optimal</em> for a
                        fully flexible run count.</p>
                    </div>
                </div>`;
            warningBanner.classList.remove('hidden');
        }
    }

    if (metrics && typeof metrics === 'object') {
        const getFlag = (val, type) => {
            if (type === 'ortho') {
                if (val === 'N/A') return { text: 'N/A', class: 'flag-limited' };
                if (val >= 90) return { text: 'Good',    class: 'flag-good' };
                if (val >= 70) return { text: 'Limited', class: 'flag-limited' };
                return           { text: 'Poor',    class: 'flag-poor' };
            }
            if (type === 'eff') {
                if (val === 'N/A') return { text: 'N/A', class: 'flag-limited' };
                if (val >= 80) return { text: 'Good',    class: 'flag-good' };
                if (val >= 50) return { text: 'Limited', class: 'flag-limited' };
                return           { text: 'Poor',    class: 'flag-poor' };
            }
            if (type === 'res') {
                if (val === 'D-Optimal')       return { text: 'Model-opt.', class: 'flag-good' };
                if (val.includes('V'))         return { text: 'Good',       class: 'flag-good' };
                if (val.includes('IV'))        return { text: 'Limited',    class: 'flag-limited' };
                return                                { text: 'Screening',  class: 'flag-poor' };
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
window.rebalanceDoEDOM  = rebalanceDoEDOM;
window.attachDeleteEvents = attachDeleteEvents;
