import { exportData, detectParetoIndices } from '../utils.js';
import { updateAllConstraintsDropdowns } from './constraints.js';

/**
 * Wires a "select all / deselect all" header checkbox to a set of row checkboxes.
 * @param {string} headerId   - id of the <th> header checkbox
 * @param {string} rowSelector - CSS selector for the row checkboxes
 * @param {function} syncFn   - called with the column name after each row toggle
 */
function _wireSelectAll(headerId, rowSelector, syncFn) {
    const header = document.getElementById(headerId);
    if (!header) return;

    /** Recomputes header state from the current row states */
    const refreshHeader = () => {
        const rows    = Array.from(document.querySelectorAll(rowSelector));
        const total   = rows.length;
        const checked = rows.filter(cb => cb.checked).length;
        header.checked       = checked === total;
        header.indeterminate = checked > 0 && checked < total;
    };

    // Header click → bulk toggle all rows
    header.addEventListener('change', () => {
        const rows = Array.from(document.querySelectorAll(rowSelector));
        rows.forEach(cb => {
            if (cb.checked !== header.checked) {
                cb.checked = header.checked;
                syncFn(cb.dataset.col);
            }
        });
        header.indeterminate = false;
    });

    // Individual row click → update header state
    document.querySelectorAll(rowSelector).forEach(cb => {
        cb.addEventListener('change', refreshHeader);
    });

    // Set initial header state
    refreshHeader();
}

export function renderBOSetup() {
    const featuresList  = document.getElementById('bo-features-list');
    const objectivesList = document.getElementById('bo-objectives-list');

    let featuresHtml = '';
    window.currentColumns.filter(col => window.columnRoles[col] === 'feature').forEach(col => {
        const cfg = window.boFeatureConfigs[col] || { type: 'continuous', range: '', included: true };
        featuresHtml += `<tr>
            <td>${col}</td>
            <td><select class="bo-f-type" data-col="${col}" onchange="window.syncBOFeatureConfig('${col}', 'type')">
                <option value="continuous" ${cfg.type === 'continuous' ? 'selected' : ''}>continuous</option>
                <option value="regular"    ${cfg.type === 'regular'    ? 'selected' : ''}>regular</option>
                <option value="discrete"   ${cfg.type === 'discrete'   ? 'selected' : ''}>discrete</option>
                <option value="categorical"${cfg.type === 'categorical'? 'selected' : ''}>categorical</option>
            </select></td>
            <td><input type="text" class="bo-f-range" data-col="${col}" value="${cfg.range}" placeholder="[min, max] or [min, max]steps" onchange="window.syncBOFeatureConfig('${col}', 'range')"></td>
            <td><input type="checkbox" class="bo-f-include" data-col="${col}" ${cfg.included ? 'checked' : ''} onchange="window.syncBOFeatureConfig('${col}', 'include')"></td>
        </tr>`;
    });
    featuresList.innerHTML = featuresHtml;

    let objectivesHtml = '';
    const selectedObjs  = window.currentColumns.filter(col => window.columnRoles[col] === 'objective');
    selectedObjs.forEach(col => {
        const cfg       = window.objectiveConfigs[col] || { type: 'maximize', target: '', importance: (100/selectedObjs.length).toFixed(1), included: true };
        const isDisabled = !cfg.included;
        objectivesHtml += `<tr class="${isDisabled ? 'row-disabled' : ''}">
            <td>${col}</td>
            <td><select class="bo-o-type" data-col="${col}" onchange="window.syncObjectiveConfig('${col}', 'bo'); window.toggleRowTarget(this);" ${isDisabled ? 'disabled' : ''}>
                <option value="maximize" ${cfg.type === 'maximize' ? 'selected' : ''}>maximize</option>
                <option value="minimize" ${cfg.type === 'minimize' ? 'selected' : ''}>minimize</option>
                <option value="target"   ${cfg.type === 'target'   ? 'selected' : ''}>target</option>
            </select></td>
            <td><input type="text" class="bo-o-target ${cfg.type === 'target' ? '' : 'hidden'}" data-col="${col}" value="${cfg.target}" placeholder="Target value" onchange="window.syncObjectiveConfig('${col}', 'bo', 'target')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="number" class="bo-o-importance" data-col="${col}" value="${cfg.importance}" onchange="window.syncObjectiveConfig('${col}', 'bo', 'importance')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="checkbox" class="bo-o-include" data-col="${col}" ${cfg.included ? 'checked' : ''} onchange="window.syncObjectiveConfig('${col}', 'bo', 'include')"></td>
        </tr>`;
    });
    objectivesList.innerHTML = objectivesHtml;
    updateAcqSelectForMultiObj();
    updateAllConstraintsDropdowns('bo');

    // --- Select-all / deselect-all for BO Features ---
    _wireSelectAll(
        'bo-select-all-features',
        '#bo-features-list .bo-f-include',
        col => window.syncBOFeatureConfig(col, 'include')
    );

    // --- Select-all / deselect-all for BO Objectives ---
    _wireSelectAll(
        'bo-select-all-objectives',
        '#bo-objectives-list .bo-o-include',
        col => window.syncObjectiveConfig(col, 'bo', 'include')
    );
}


export function updateAcqSelectForMultiObj() {
    const includedObjs = Array.from(document.querySelectorAll('#bo-objectives-list .bo-o-include:checked'));
    const activeObjs   = includedObjs.length;
    const acqSelect    = document.getElementById('acq-type-select');
    const acqItem      = acqSelect ? acqSelect.closest('.tweak-item') : null;
    if (!acqSelect) return;

    const isMultiObj = activeObjs > 1;
    acqSelect.disabled       = isMultiObj;
    acqSelect.style.opacity  = isMultiObj ? '0.45' : '1';
    acqSelect.style.cursor   = isMultiObj ? 'not-allowed' : '';

    if (acqItem) {
        let note = acqItem.querySelector('.mo-note');
        if (isMultiObj) {
            const weights  = includedObjs.map(cb => parseFloat(cb.closest('tr').querySelector('.bo-o-importance').value) || 0);
            const allEqual = weights.every(v => Math.abs(v - weights[0]) < 0.1);

            if (!note) {
                note = document.createElement('small');
                note.className   = 'mo-note';
                note.style.cssText = 'color: var(--text-secondary); font-size: 0.75rem; margin-top: 3px; display: block; border-left: 2px solid var(--accent-color); padding-left: 6px;';
                acqItem.appendChild(note);
            }
            note.innerHTML = allEqual
                ? `<span style="color:var(--accent-color)">Discovery Mode:</span> Exploring full Pareto front via NEHVI.`
                : `<span style="color:var(--accent-color)">Priority Mode:</span> Focus on weighted sum (Scalarized EI).`;
        } else {
            if (note) note.remove();
        }
    }
}

export async function runBO() {
    const perfInfo = document.getElementById('bo-performance-info');
    if (perfInfo) perfInfo.innerHTML = '';
    const estRes = document.getElementById('bo-estimate-results');
    if (estRes)   estRes.innerHTML = '';
    const boResContainer = document.getElementById('bo-results-container');
    if (boResContainer) boResContainer.classList.add('hidden');
    // Clear any stale exhaustion banner from a previous run
    const staleExhBanner = document.getElementById('bo-exhaustion-banner');
    if (staleExhBanner) staleExhBanner.remove();

    const features = Array.from(document.querySelectorAll('#bo-features-list tr'))
        .filter(tr => tr.querySelector('.bo-f-include').checked)
        .map(tr => ({
            name:  tr.querySelector('.bo-f-type').dataset.col,
            type:  tr.querySelector('.bo-f-type').value,
            range: tr.querySelector('.bo-f-range').value
        }));

    const objectives = Array.from(document.querySelectorAll('#bo-objectives-list tr'))
        .filter(tr => tr.querySelector('.bo-o-include').checked)
        .map(tr => ({
            name:       tr.querySelector('.bo-o-type').dataset.col,
            type:       tr.querySelector('.bo-o-type').value,
            target:     tr.querySelector('.bo-o-target').value,
            importance: tr.querySelector('.bo-o-importance').value
        }));

    const strategy = document.querySelector('input[name="bo-strategy"]:checked').value;
    const numericalConstraints = Array.from(document.querySelectorAll('#bo-numerical-constraints-list tr.constraint-row'))
        .map(tr => tr.querySelector('.constraint-equation').value.trim())
        .filter(eq => eq !== "");

    const disallowedCombinations = Array.from(document.querySelectorAll('#bo-disallowed-constraints-list tr.disallowed-row'))
        .map(tr => {
            const conditions = Array.from(tr.querySelectorAll('.condition-item')).map(item => ({
                feature: item.querySelector('.condition-feature').value,
                value: item.querySelector('.condition-value').value
            }));
            return conditions;
        })
        .filter(conds => conds.length > 0);

    const tweaks = {
        batch_size:            document.getElementById('batch-size').value,
        acq_type:              document.getElementById('acq-type-select').value,
        kernel:                document.getElementById('kernel-select').value,
        exploration:           document.getElementById('style-slider').value,
        noiseless:             document.getElementById('bo-noiseless').checked,
        avoid_reval:           document.getElementById('bo-avoid-reval').checked,
        optimization_strategy: strategy,
        constraints: {
            numerical: numericalConstraints,
            disallowed: disallowedCombinations
        }
    };

    document.getElementById('loading-text').textContent = "Optimizing...";
    document.getElementById('loading-overlay').classList.remove('hidden');

    window.currentAbortController = new AbortController();
    try {
        const res    = await fetch('/optimize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ data: window.currentData, columns: window.currentColumns, features, objectives, tweaks }),
            signal: window.currentAbortController.signal
        });
        const result = await res.json();
        document.getElementById('loading-overlay').classList.add('hidden');
        if (result.error) return alert("Optimization Error: " + result.error);

        // Handle exhaustion: all optimal conditions have already been tested
        if (result.exhausted) {
            renderBOExhaustedBanner(result);
            return;
        }

        window.suggestionsData = result.suggestions;

        if (result.performance) renderBOPerformanceTable(result.performance);
        renderBOEstimator();

        const boResContainer = document.getElementById('bo-results-container');
        if (boResContainer) boResContainer.classList.remove('hidden');
    } catch (err) {
        document.getElementById('loading-overlay').classList.add('hidden');
        if (err.name === 'AbortError') return console.log('BO calculation aborted by user.');
        return alert("A critical error occurred during optimization: " + err.message);
    } finally {
        window.currentAbortController = null;
    }

    if (!window.objectiveConfigs) window.objectiveConfigs = {};
    objectives.forEach(obj => { window.objectiveConfigs[obj.name] = { ...obj, included: true }; });

    const section    = document.getElementById('results-section');
    const head       = document.getElementById('results-table-head');
    const body       = document.getElementById('results-table-body');
    const commitBtn  = document.getElementById('commit-suggestions-btn');

    const featCols    = window.currentColumns.filter(c => window.columnRoles[c] === 'feature');
    const objCols     = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');
    const allObjCols  = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');
    const optimizedCols = objectives.map(o => o.name);

    head.innerHTML = '<tr>' +
        featCols.map(c => `<th>${c}</th>`).join('') +
        objCols.map(c => `<th>${c} (Result)</th>`).join('') +
        '</tr>';

    const checkCommitReadiness = () => {
        const objCellsEl  = body.querySelectorAll('.obj-input-cell');
        const featCellsEl = body.querySelectorAll('.feat-input-cell');
        let allFilled = true;

        objCellsEl.forEach(td => {
            const val = td.textContent.trim();
            if (val === '' || isNaN(parseFloat(val))) allFilled = false;
        });
        featCellsEl.forEach(td => {
            if (td.textContent.trim() === '') allFilled = false;
        });

        commitBtn.disabled      = !allFilled;
        commitBtn.style.opacity = allFilled ? '1' : '0.5';
        commitBtn.style.cursor  = allFilled ? 'pointer' : 'not-allowed';
    };

    body.innerHTML = window.suggestionsData.map(row => {
        const fCells = featCols.map(c =>
            `<td contenteditable="true" class="feat-input-cell editable-cell" style="background:#f8fafc; border: 1px dashed #94a3b8;">${row[c]}</td>`
        ).join('');
        const oCells = allObjCols.map(c => {
            const isOptimized = optimizedCols.includes(c);
            const content     = isOptimized ? '' : 'N/D';
            const style       = isOptimized
                ? 'background:#f0fdf4; border: 1px dashed #22c55e;'
                : 'background:#f1f5f9; border: 1px dashed #94a3b8; color: #64748b; font-style: italic;';
            return `<td contenteditable="true" class="obj-input-cell editable-cell" style="${style}">${content}</td>`;
        }).join('');
        return '<tr>' + fCells + oCells + '</tr>';
    }).join('');

    body.querySelectorAll('.obj-input-cell, .feat-input-cell').forEach(cell => {
        cell.addEventListener('input', checkCommitReadiness);
    });
    checkCommitReadiness();

    section.classList.remove('hidden');
    renderTrendPlot();
    section.scrollIntoView({ behavior: 'smooth' });
}

export function renderBOExhaustedBanner(result) {
    const boResContainer = document.getElementById('bo-results-container');
    if (boResContainer) boResContainer.classList.remove('hidden');

    const head    = document.getElementById('results-table-head');
    const body    = document.getElementById('results-table-body');
    const section = document.getElementById('results-section');
    if (head) head.innerHTML = '';
    if (body) body.innerHTML = '';

    const container = boResContainer || section;
    let banner = document.getElementById('bo-exhaustion-banner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'bo-exhaustion-banner';
        if (container) container.prepend(banner);
    }

    banner.innerHTML = `
        <div style="
            background: linear-gradient(135deg,#fef3c7,#fffbeb);
            border-left: 4px solid #f59e0b;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
        ">
            <span style="font-size:1.5rem">⚠️</span>
            <div>
                <b style="color:#92400e">All optimal conditions have already been tested.</b><br>
                <span style="color:#78350f;font-size:0.9rem">
                    BO could not find new distinct combinations at the current
                    significant-figure precision. To explore further, try:<br>
                    &nbsp;• Increase the <b>Exploration</b> slider<br>
                    &nbsp;• Switch the <b>Kernel</b> or <b>Acquisition Function</b><br>
                    &nbsp;• Expand the feature <b>ranges</b><br>
                    &nbsp;• Uncheck <b>Avoid Re-evaluation</b> to allow revisiting candidates
                </span>
            </div>
        </div>`;

    if (result && result.performance) renderBOPerformanceTable(result.performance);
    if (section) {
        section.classList.remove('hidden');
        section.scrollIntoView({ behavior: 'smooth' });
    }
}

export function renderBOPerformanceTable(performance) {
    const container = document.getElementById('bo-performance-info');
    if (!container) return;

    const getFittingSuggestion = (status) => {
        switch(status) {
            case 'Good':    return 'Reliable model for interpolation.';
            case 'Limited': return 'Moderate predictive power; use with caution.';
            case 'Poor':    return 'Low predictive power; the BO model may be uncertain in some regions.';
            default:        return '';
        }
    };

    let html = `
        <table class="setup-table" style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <thead>
                <tr style="text-align: left; border-bottom: 2px solid var(--border-color);">
                    <th style="padding: 10px;">Objective</th>
                    <th style="padding: 10px;">Predictive Power (R²)</th>
                    <th style="padding: 10px;">Status</th>
                    <th style="padding: 10px;">Suggestion</th>
                </tr>
            </thead>
            <tbody>`;

    for (const obj in performance) {
        const r          = performance[obj];
        const statusClass = r.status.toLowerCase();
        let nameDisplay   = `<b>${obj}</b>`;
        if (obj === 'global_success') nameDisplay = `<span style="color: var(--accent-color);">✨ <b>Global Success Index</b></span>`;

        html += `
            <tr style="border-bottom: 1px solid var(--border-color);">
                <td style="padding: 10px;">${nameDisplay}</td>
                <td style="padding: 10px;">${r.r2 ?? 'N/A'}</td>
                <td style="padding: 10px;"><span class="badge ${statusClass}">${r.status}</span></td>
                <td style="padding: 10px; font-size: 0.9rem; color: var(--text-secondary);">${getFittingSuggestion(r.status)}</td>
            </tr>`;
    }
    html += `</tbody></table>`;
    container.innerHTML = html;
}

export function renderBOEstimator() {
    const estDiv = document.getElementById('bo-estimator-inputs');
    if (!estDiv) return;

    const features = Array.from(document.querySelectorAll('#bo-features-list tr'))
        .filter(tr => tr.querySelector('.bo-f-include').checked)
        .map(tr => ({
            name:  tr.querySelector('.bo-f-type').dataset.col,
            type:  tr.querySelector('.bo-f-type').value,
            range: tr.querySelector('.bo-f-range').value
        }));

    estDiv.innerHTML = features.map(f => {
        if (f.type === 'categorical') {
            const choices = f.range.split(',').map(c => c.trim());
            const options = choices.map(v => `<option value="${v}">${v}</option>`).join('');
            return `<div class="tweak-item">
                <label>${f.name}</label>
                <select class="bo-est-input" data-col="${f.name}">${options}</select>
            </div>`;
        } else {
            return `<div class="tweak-item">
                <label>${f.name}</label>
                <input type="number" class="bo-est-input" data-col="${f.name}" step="any">
            </div>`;
        }
    }).join('');
}

export async function runBOEstimate() {
    const inputs = {};
    document.querySelectorAll('.bo-est-input').forEach(el => inputs[el.dataset.col] = el.value);

    const features = Array.from(document.querySelectorAll('#bo-features-list tr'))
        .filter(tr => tr.querySelector('.bo-f-include').checked)
        .map(tr => ({
            name:  tr.querySelector('.bo-f-type').dataset.col,
            type:  tr.querySelector('.bo-f-type').value,
            range: tr.querySelector('.bo-f-range').value
        }));

    const objectives = Array.from(document.querySelectorAll('#bo-objectives-list tr'))
        .filter(tr => tr.querySelector('.bo-o-include').checked)
        .map(tr => ({
            name:       tr.querySelector('.bo-o-type').dataset.col,
            type:       tr.querySelector('.bo-o-type').value,
            target:     tr.querySelector('.bo-o-target').value,
            importance: tr.querySelector('.bo-o-importance').value
        }));

    const strategy = document.querySelector('input[name="bo-strategy"]:checked').value;
    const tweaks   = {
        kernel:                document.getElementById('kernel-select').value,
        optimization_strategy: strategy
    };

    const resultsDiv = document.getElementById('bo-estimate-results');
    resultsDiv.innerHTML = '<div style="color: var(--text-secondary); animation: pulse 1.5s infinite;"><i>Calculating BO predictions...</i></div>';

    try {
        const res    = await fetch('/estimate_bo', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ data: window.currentData, columns: window.currentColumns, features, objectives, tweaks, inputs })
        });
        const result = await res.json();
        if (result.error) {
            resultsDiv.innerHTML = `<span style="color:red">Error: ${result.error}</span>`;
            return;
        }

        let html = '<div style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid var(--border-color);">';
        html += `<b>Estimated Global Success Score: <span style="font-size: 1.2rem; color: var(--accent-color);">${result.global_success}%</span></b></div>`;
        html += '<b>Predictions:</b><br>';
        for (const obj in result.predictions) {
            const val  = result.predictions[obj];
            const err  = result.std_errors ? result.std_errors[obj] : null;
            const succ = result.success_scores[obj];
            const errBadge = (err !== null && err !== undefined) ? `<span style="font-size: 0.85rem; color: var(--text-secondary); margin-left: 4px;">(&plusmn; ${err})</span>` : '';
            html += `<div style="display: flex; justify-content: space-between; align-items: center; margin: 4px 0;">
                        <span><b>${obj}:</b> ${val}${errBadge}</span>
                        <span style="color: ${succ > 80 ? '#22c55e' : (succ > 50 ? '#f59e0b' : '#ef4444')}">Success: ${succ}%</span>
                     </div>`;
        }
        resultsDiv.innerHTML = html;
    } catch(err) {
        resultsDiv.innerHTML = `<span style="color:red">Request failed: ${err.message}</span>`;
        console.error(err);
    }
}

export function initBOVisualizationDropdown() {
    const visSelect = document.getElementById('bo-vis-type-select');
    if (!visSelect || visSelect.dataset.initialized) return;
    visSelect.dataset.initialized = 'true';

    visSelect.addEventListener('change', () => {
        renderBOVisualization();
    });
}

let currentSurfaceRequestId = 0;
let currentSurfaceAbortController = null;

export function renderBOVisualization() {
    if (!window.currentData || window.currentData.length === 0) {
        try {
            Plotly.purge('trend-plot');
            const contourEl = document.getElementById('bo-contour-plot');
            if (contourEl) { Plotly.purge(contourEl); contourEl.innerHTML = ''; }
            const surfEl = document.getElementById('bo-surface-plot');
            if (surfEl) { Plotly.purge(surfEl); surfEl.innerHTML = ''; }
        } catch(e) {}
        const visSection = document.getElementById('bo-visualization-section') || document.getElementById('bo-visuals');
        if (visSection) visSection.classList.add('hidden');
        const paretoSection = document.getElementById('pareto-plot-section');
        if (paretoSection) paretoSection.classList.add('hidden');
        return;
    }

    const visSection = document.getElementById('bo-visualization-section') || document.getElementById('bo-visuals');
    if (visSection) visSection.classList.remove('hidden');

    initBOVisualizationDropdown();

    const viewType = document.getElementById('bo-vis-type-select')?.value || 'trend';
    const trendPlotEl = document.getElementById('trend-plot');
    const contourPlotEl = document.getElementById('bo-contour-plot');
    const surfacePlotEl = document.getElementById('bo-surface-plot');
    const controlsEl = document.getElementById('bo-vis-controls');
    const titleEl = document.getElementById('bo-vis-title');

    if (viewType === 'trend') {
        if (titleEl) titleEl.textContent = 'Objectives Trend';
        if (trendPlotEl) trendPlotEl.style.display = 'block';
        if (contourPlotEl) { contourPlotEl.style.display = 'none'; try { Plotly.purge(contourPlotEl); } catch(e) {} contourPlotEl.innerHTML = ''; }
        if (surfacePlotEl) { surfacePlotEl.style.display = 'none'; try { Plotly.purge(surfacePlotEl); } catch(e) {} surfacePlotEl.innerHTML = ''; }
        if (controlsEl) { controlsEl.style.display = 'none'; controlsEl.innerHTML = ''; }
        renderTrendPlotOnly();
    } else if (viewType === 'contour') {
        if (titleEl) titleEl.textContent = 'Contour Plot (Response Surface)';
        if (trendPlotEl) { trendPlotEl.style.display = 'none'; try { Plotly.purge(trendPlotEl); } catch(e) {} }
        if (surfacePlotEl) { surfacePlotEl.style.display = 'none'; try { Plotly.purge(surfacePlotEl); } catch(e) {} surfacePlotEl.innerHTML = ''; }
        if (contourPlotEl) contourPlotEl.style.display = 'block';
        const isValid = renderContourControls();
        if (isValid) {
            updateContourPlot();
        }
    } else if (viewType === 'surface') {
        if (titleEl) titleEl.textContent = '3D Surface Plot (Response Surface)';
        if (trendPlotEl) { trendPlotEl.style.display = 'none'; try { Plotly.purge(trendPlotEl); } catch(e) {} }
        if (contourPlotEl) { contourPlotEl.style.display = 'none'; try { Plotly.purge(contourPlotEl); } catch(e) {} contourPlotEl.innerHTML = ''; }
        if (surfacePlotEl) surfacePlotEl.style.display = 'block';
        const isValid = renderSurfaceControls();
        if (isValid) {
            updateSurfacePlot();
        }
    }

    renderParetoPlot();
}

function renderTrendPlotOnly() {
    const objCols = window.currentColumns.filter(c => window.columnRoles[c] === 'objective' && (window.objectiveConfigs[c] ? window.objectiveConfigs[c].included : true));

    if (objCols.length === 0) {
        try { Plotly.purge('trend-plot'); } catch(e) {}
        const trendEl = document.getElementById('trend-plot');
        if (trendEl) trendEl.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-secondary);"><i>No included objectives to display trend.</i></div>';
        return;
    }

    const traces = objCols.map(col => {
        const colIdx = window.currentColumns.indexOf(col);
        const xDist  = [];
        const yDist  = [];

        window.currentData.forEach((row, i) => {
            const rawVal = row[colIdx];
            const parsed = parseFloat(rawVal);
            if (rawVal !== '' && !isNaN(parsed)) {
                xDist.push(i + 1);
                yDist.push(parsed);
            }
        });

        return { x: xDist, y: yDist, name: col, mode: 'lines+markers', line: { width: 3 }, marker: { size: 8 } };
    }).filter(t => t.x.length > 0);

    if (traces.length === 0) return;

    Plotly.newPlot('trend-plot', traces, {
        title:    objCols.length === 1 ? `Trend: ${objCols[0]}` : 'Objectives Trend',
        autosize: true,
        height:   400,
        margin:   { l: 60, r: 40, t: 80, b: 60 },
        legend:   { orientation: 'h', y: -0.2 }
    }, { responsive: true });

    setTimeout(() => { Plotly.Plots.resize('trend-plot'); }, 100);
}

export function renderTrendPlot() {
    renderBOVisualization();
}

// Helper: get feature type ('categorical' | 'continuous' | 'discrete' | 'regular')
// Reads from authoritative in-memory boFeatureConfigs first; falls back to DOM.
function getBOFeatureType(featName) {
    // Authoritative source: in-memory config (immune to DOM rebuild timing)
    if (window.boFeatureConfigs && window.boFeatureConfigs[featName]) {
        return window.boFeatureConfigs[featName].type || 'continuous';
    }
    // Fallback: read from DOM table row if config not yet populated
    const row = Array.from(document.querySelectorAll('#bo-features-list tr'))
        .find(tr => tr.querySelector('.bo-f-type')?.dataset.col === featName);
    return row ? row.querySelector('.bo-f-type').value : 'continuous';
}

// Helper: get unique string values of a categorical feature from current data
function getBOFeatureCategoricalValues(featName) {
    const colIdx = window.currentColumns.indexOf(featName);
    if (colIdx < 0) return [];
    const seen = new Set();
    window.currentData.forEach(r => { if (r[colIdx] != null && r[colIdx] !== '') seen.add(String(r[colIdx])); });
    return Array.from(seen).sort();
}

// Helper: build a fixed-value input for a feature (dropdown for categorical, number for others)
function buildFixedInput(f, cssClass) {
    const colIdx = window.currentColumns.indexOf(f);
    const fType  = getBOFeatureType(f);
    const vals   = window.currentData.map(r => r[colIdx]).filter(v => v != null && v !== '');

    if (fType === 'categorical') {
        const options = getBOFeatureCategoricalValues(f);
        const defaultVal = options[0] || '';
        const optHtml = options.map(v => `<option value="${v}">${v}</option>`).join('');
        return `<div class="tweak-item"><label>${f}: <span style="font-size:0.75rem;color:var(--text-secondary);">(categorical)</span></label>
                    <select class="${cssClass}" data-feat="${f}" style="width:100%;">${optHtml}</select></div>`;
    } else {
        const numVals   = vals.map(v => parseFloat(v)).filter(v => !isNaN(v));
        const defaultVal = numVals.length > 0 ? (numVals.reduce((a,b) => a+b, 0) / numVals.length).toFixed(3) : '';
        return `<div class="tweak-item"><label>${f}:</label>
                    <input type="number" class="${cssClass}" data-feat="${f}" value="${defaultVal}" step="any" style="width:100%;"></div>`;
    }
}

function renderContourControls() {
    const controls = document.getElementById('bo-vis-controls');
    if (!controls) return false;

    const allFeatCols = window.currentColumns.filter(c => window.columnRoles[c] === 'feature' && (window.boFeatureConfigs[c] ? window.boFeatureConfigs[c].included : true));
    // Only numeric features can be on the X/Y axes
    const numFeatCols = allFeatCols.filter(f => getBOFeatureType(f) !== 'categorical');
    const catFeatCols = allFeatCols.filter(f => getBOFeatureType(f) === 'categorical');
    const objCols     = window.currentColumns.filter(c => window.columnRoles[c] === 'objective' && (window.objectiveConfigs[c] ? window.objectiveConfigs[c].included : true));

    if (numFeatCols.length < 2 || objCols.length === 0) {
        controls.innerHTML = `<p class="text-secondary" style="grid-column: 1 / -1; margin: 0.5rem 0;"><i>Contour plot requires at least 2 included <b>numerical</b> features and 1 objective.${
            catFeatCols.length > 0 ? ' Categorical features can only be set as fixed values.' : ''}</i></p>`;
        controls.style.display = 'grid';
        const contourPlotEl = document.getElementById('bo-contour-plot');
        if (contourPlotEl) {
            try { Plotly.purge(contourPlotEl); } catch(e) {}
            contourPlotEl.innerHTML = '';
        }
        return false;
    }

    const currentX   = document.getElementById('bo-contour-x')?.value;
    const currentY   = document.getElementById('bo-contour-y')?.value;
    const currentObj = document.getElementById('bo-contour-obj')?.value;

    const xSelected   = numFeatCols.includes(currentX) ? currentX : numFeatCols[0];
    const ySelected   = numFeatCols.includes(currentY) && currentY !== xSelected ? currentY : (numFeatCols.find(f => f !== xSelected) || numFeatCols[0]);
    const objSelected = objCols.includes(currentObj)   ? currentObj : objCols[0];

    let html = `
        <div class="tweak-item">
            <label>X-Axis Feature (numerical):</label>
            <select id="bo-contour-x" style="width: 100%;">
                ${numFeatCols.map(f => `<option value="${f}" ${f === xSelected ? 'selected' : ''}>${f}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Y-Axis Feature (numerical):</label>
            <select id="bo-contour-y" style="width: 100%;">
                ${numFeatCols.map(f => `<option value="${f}" ${f === ySelected ? 'selected' : ''}>${f}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Objective (Color &amp; Contours):</label>
            <select id="bo-contour-obj" style="width: 100%;">
                ${objCols.map(o => `<option value="${o}" ${o === objSelected ? 'selected' : ''}>${o}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Grid Resolution:</label>
            <select id="bo-contour-res" style="width: 100%;">
                <option value="15">15 &#215; 15 (Fast)</option>
                <option value="25" selected>25 &#215; 25 (Standard)</option>
                <option value="40">40 &#215; 40 (High)</option>
            </select>
        </div>
    `;

    // Fixed value inputs for all non-axis features (categorical → dropdown, numerical → number)
    const fixedFeats = allFeatCols.filter(f => f !== xSelected && f !== ySelected);
    if (fixedFeats.length > 0) {
        html += `<div style="grid-column: 1 / -1; margin-top: 0.5rem;"><b style="font-size: 0.85rem; color: var(--text-secondary);">Fixed values for remaining features:</b></div>`;
        fixedFeats.forEach(f => { html += buildFixedInput(f, 'bo-contour-fixed'); });
    }

    html += `<div style="grid-column: 1 / -1; display: flex; justify-content: flex-end; margin-top: 0.5rem;">
                <button id="bo-contour-update-btn" class="btn primary small">Calculate &amp; Update Contour</button>
             </div>`;

    controls.innerHTML = html;
    controls.style.display = 'grid';

    document.getElementById('bo-contour-update-btn').addEventListener('click', updateContourPlot);
    // Re-render when X/Y change (so fixed-value fields update too)
    document.getElementById('bo-contour-x').addEventListener('change', () => { renderContourControls(); updateContourPlot(); });
    document.getElementById('bo-contour-y').addEventListener('change', () => { renderContourControls(); updateContourPlot(); });
    document.getElementById('bo-contour-obj').addEventListener('change', updateContourPlot);
    return true;
}

function renderSurfaceControls() {
    const controls = document.getElementById('bo-vis-controls');
    if (!controls) return false;

    const allFeatCols = window.currentColumns.filter(c => window.columnRoles[c] === 'feature' && (window.boFeatureConfigs[c] ? window.boFeatureConfigs[c].included : true));
    // Only numeric features can be on X/Y axes
    const numFeatCols = allFeatCols.filter(f => getBOFeatureType(f) !== 'categorical');
    const catFeatCols = allFeatCols.filter(f => getBOFeatureType(f) === 'categorical');
    const objCols     = window.currentColumns.filter(c => window.columnRoles[c] === 'objective' && (window.objectiveConfigs[c] ? window.objectiveConfigs[c].included : true));

    if (numFeatCols.length < 2 || objCols.length === 0) {
        controls.innerHTML = `<p class="text-secondary" style="grid-column: 1 / -1; margin: 0.5rem 0;"><i>3D surface plot requires at least 2 included <b>numerical</b> features and 1 objective.${
            catFeatCols.length > 0 ? ' Categorical features can only be set as fixed values.' : ''}</i></p>`;
        controls.style.display = 'grid';
        const surfacePlotEl = document.getElementById('bo-surface-plot');
        if (surfacePlotEl) {
            try { Plotly.purge(surfacePlotEl); } catch(e) {}
            surfacePlotEl.innerHTML = '';
        }
        return false;
    }

    const currentX     = document.getElementById('bo-surf-x')?.value;
    const currentY     = document.getElementById('bo-surf-y')?.value;
    const currentZ     = document.getElementById('bo-surf-z')?.value;
    const currentColor = document.getElementById('bo-surf-color')?.value;
    const currentRes   = document.getElementById('bo-surf-res')?.value || '15';

    const xSelected     = numFeatCols.includes(currentX) ? currentX : numFeatCols[0];
    const ySelected     = numFeatCols.includes(currentY) && currentY !== xSelected ? currentY : (numFeatCols.find(f => f !== xSelected) || numFeatCols[0]);
    const zSelected     = (objCols.includes(currentZ)) ? currentZ : objCols[0];
    const colorSelected = objCols.includes(currentColor) ? currentColor : objCols[0];

    let zOptionsHtml = `<optgroup label="Objectives">`;
    objCols.forEach(o => {
        zOptionsHtml += `<option value="${o}" ${o === zSelected ? 'selected' : ''}>${o} (Objective)</option>`;
    });
    zOptionsHtml += `</optgroup>`;

    let html = `
        <div class="tweak-item">
            <label>X-Axis Feature (numerical):</label>
            <select id="bo-surf-x" style="width: 100%;">
                ${numFeatCols.map(f => `<option value="${f}" ${f === xSelected ? 'selected' : ''}>${f}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Y-Axis Feature (numerical):</label>
            <select id="bo-surf-y" style="width: 100%;">
                ${numFeatCols.map(f => `<option value="${f}" ${f === ySelected ? 'selected' : ''}>${f}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Z-Axis / Height (Objective):</label>
            <select id="bo-surf-z" style="width: 100%;">
                ${zOptionsHtml}
            </select>
        </div>
        <div class="tweak-item">
            <label>Surface Color (Objective):</label>
            <select id="bo-surf-color" style="width: 100%;">
                ${objCols.map(o => `<option value="${o}" ${o === colorSelected ? 'selected' : ''}>${o}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Grid Resolution:</label>
            <select id="bo-surf-res" style="width: 100%;">
                <option value="15" ${currentRes === '15' ? 'selected' : ''}>15 &#215; 15 (Smooth 3D)</option>
                <option value="20" ${currentRes === '20' ? 'selected' : ''}>20 &#215; 20 (Detailed)</option>
                <option value="30" ${currentRes === '30' ? 'selected' : ''}>30 &#215; 30 (Ultra)</option>
            </select>
        </div>
    `;

    // Fixed value inputs for ALL non-axis features: dropdown for categorical, number for numeric
    const fixedFeats = allFeatCols.filter(f => f !== xSelected && f !== ySelected);
    if (fixedFeats.length > 0) {
        html += `<div style="grid-column: 1 / -1; margin-top: 0.5rem;"><b style="font-size: 0.85rem; color: var(--text-secondary);">Fixed values for remaining features:</b></div>`;
        fixedFeats.forEach(f => { html += buildFixedInput(f, 'bo-surf-fixed'); });
    }

    html += `<div style="grid-column: 1 / -1; display: flex; justify-content: flex-end; margin-top: 0.5rem;">
                <button id="bo-surf-update-btn" class="btn primary small">Calculate &amp; Update 3D Surface</button>
             </div>`;

    controls.innerHTML = html;
    controls.style.display = 'grid';

    document.getElementById('bo-surf-update-btn').addEventListener('click', updateSurfacePlot);
    // Re-render full controls on X/Y change so fixed-value fields update dynamically
    document.getElementById('bo-surf-x').addEventListener('change', () => { renderSurfaceControls(); updateSurfacePlot(); });
    document.getElementById('bo-surf-y').addEventListener('change', () => { renderSurfaceControls(); updateSurfacePlot(); });
    document.getElementById('bo-surf-z').addEventListener('change', updateSurfacePlot);
    document.getElementById('bo-surf-color').addEventListener('change', updateSurfacePlot);
    return true;
}

async function updateContourPlot() {
    const xFeat = document.getElementById('bo-contour-x')?.value;
    const yFeat = document.getElementById('bo-contour-y')?.value;
    const colorObj = document.getElementById('bo-contour-obj')?.value;
    const res = parseInt(document.getElementById('bo-contour-res')?.value || '25', 10);
    const plotContainer = document.getElementById('bo-contour-plot');

    if (!xFeat || !yFeat || !colorObj || !plotContainer) return;
    if (xFeat === yFeat) {
        try { Plotly.purge(plotContainer); } catch(e) {}
        plotContainer.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#ef4444;font-style:italic;">Please select two distinct features for X and Y axes.</div>';
        return;
    }

    const fixedValues = {};
    document.querySelectorAll('.bo-contour-fixed').forEach(el => {
        // select = categorical (keep as string); input = numerical (parse as float)
        fixedValues[el.dataset.feat] = el.tagName === 'SELECT' ? el.value : parseFloat(el.value);
    });

    const features = window.currentColumns
        .filter(c => window.columnRoles[c] === 'feature' && (window.boFeatureConfigs[c] ? window.boFeatureConfigs[c].included : true))
        .map(c => {
            const cfg = window.boFeatureConfigs[c] || {};
            return {
                name: c,
                type: cfg.type || 'continuous',
                range: cfg.range || ''
            };
        });

    const objectives = window.currentColumns
        .filter(c => window.columnRoles[c] === 'objective' && (window.objectiveConfigs[c] ? window.objectiveConfigs[c].included : true))
        .map(c => {
            const cfg = window.objectiveConfigs[c] || {};
            return {
                name: c,
                type: cfg.type || 'maximize',
                target: cfg.target || '',
                importance: cfg.importance || 0
            };
        });

    if (features.length < 2 || objectives.length === 0) {
        try { Plotly.purge(plotContainer); } catch(e) {}
        plotContainer.innerHTML = '';
        return;
    }

    // Ensure colorObj is valid
    const validObjNames = objectives.map(o => o.name);
    const effectiveColorObj = validObjNames.includes(colorObj) ? colorObj : validObjNames[0];

    const tweaks = {
        kernel: document.getElementById('kernel-select')?.value || 'matern52'
    };

    // Cancel any previous active request
    if (currentSurfaceAbortController) {
        try { currentSurfaceAbortController.abort(); } catch(e) {}
    }
    currentSurfaceAbortController = new AbortController();
    const thisRequestId = ++currentSurfaceRequestId;

    // Silent background computation — no spinner injected into DOM
    try {
        const response = await fetch('/bo_surface', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: currentSurfaceAbortController.signal,
            body: JSON.stringify({
                data: window.currentData,
                columns: window.currentColumns,
                features,
                objectives,
                tweaks,
                x_feat: xFeat,
                y_feat: yFeat,
                z_axis: effectiveColorObj,
                color_obj: effectiveColorObj,
                fixed_values: fixedValues,
                resolution: res
            })
        });

        if (thisRequestId !== currentSurfaceRequestId) return;

        const result = await response.json();
        if (result.error) {
            try { Plotly.purge(plotContainer); } catch(e) {}
            plotContainer.innerHTML = `<div style="color:red;padding:1rem;">Contour calculation error: ${result.error}</div>`;
            return;
        }

        const contourTrace = {
            x: result.x,
            y: result.y,
            z: result.z,
            type: 'contour',
            colorscale: 'Viridis',
            contours: {
                coloring: 'heatmap',
                showlabels: true,
                labelfont: { family: 'Inter, sans-serif', size: 11, color: '#ffffff' }
            },
            line: { smoothing: 0.85, width: 1.5 },
            colorbar: { title: result.color_obj, len: 0.9 },
            hovertemplate: `<b>${result.x_feat}:</b> %{x}<br><b>${result.y_feat}:</b> %{y}<br><b>${result.color_obj}:</b> %{z:.3f}<extra></extra>`
        };

        const inSliceExps = result.experiments.filter(e => e.in_slice);
        const offSliceExps = result.experiments.filter(e => !e.in_slice);

        const traces = [contourTrace];

        if (inSliceExps.length > 0) {
            traces.push({
                x: inSliceExps.map(e => e.x),
                y: inSliceExps.map(e => e.y),
                mode: 'markers',
                type: 'scatter',
                name: 'In-Slice Experiments (Solid)',
                marker: {
                    size: 9,
                    color: '#ef4444',
                    line: { width: 1.5, color: '#ffffff' }
                },
                customdata: inSliceExps.map(e => [e.exp_id, e.color, e.conditions]),
                hovertemplate: `<b>Experiment #%{customdata[0]}</b> [In-Slice]<br>${result.x_feat}: %{x}<br>${result.y_feat}: %{y}<br>${result.color_obj}: %{customdata[1]}<br>%{customdata[2]}<extra></extra>`
            });
        }

        if (offSliceExps.length > 0) {
            traces.push({
                x: offSliceExps.map(e => e.x),
                y: offSliceExps.map(e => e.y),
                mode: 'markers',
                type: 'scatter',
                name: 'Other Experiments (Off-Slice)',
                marker: {
                    size: 8,
                    color: 'rgba(239, 68, 68, 0.15)',
                    line: { width: 1.5, color: '#ef4444' }
                },
                customdata: offSliceExps.map(e => [e.exp_id, e.color, e.conditions]),
                hovertemplate: `<b>Experiment #%{customdata[0]}</b> [Off-Slice]<br>${result.x_feat}: %{x}<br>${result.y_feat}: %{y}<br>${result.color_obj}: %{customdata[1]}<br>%{customdata[2]}<extra></extra>`
            });
        }

        try { Plotly.purge(plotContainer); } catch(e) {}
        Plotly.newPlot(plotContainer, traces, {
            title: `Contour Plot: ${result.color_obj} vs (${result.x_feat}, ${result.y_feat})`,
            xaxis: { title: result.x_feat, automargin: true },
            yaxis: { title: result.y_feat, automargin: true },
            margin: { l: 70, r: 50, t: 70, b: 60 },
            height: 500,
            legend: { orientation: 'h', y: -0.15 }
        }, { responsive: true });

    } catch (err) {
        if (err.name === 'AbortError') return;
        if (thisRequestId !== currentSurfaceRequestId) return;
        try { Plotly.purge(plotContainer); } catch(e) {}
        plotContainer.innerHTML = `<div style="color:red;padding:1rem;">Failed to calculate contour: ${err.message}</div>`;
        console.error(err);
    }
}

async function updateSurfacePlot() {
    const xFeat = document.getElementById('bo-surf-x')?.value;
    const yFeat = document.getElementById('bo-surf-y')?.value;
    const zAxis = document.getElementById('bo-surf-z')?.value;
    const colorObj = document.getElementById('bo-surf-color')?.value;
    const res = parseInt(document.getElementById('bo-surf-res')?.value || '15', 10);
    const plotContainer = document.getElementById('bo-surface-plot');

    if (!xFeat || !yFeat || !colorObj || !plotContainer) return;
    if (xFeat === yFeat) {
        try { Plotly.purge(plotContainer); } catch(e) {}
        plotContainer.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#ef4444;font-style:italic;">Please select two distinct features for X and Y axes.</div>';
        return;
    }

    const fixedValues = {};
    document.querySelectorAll('.bo-surf-fixed').forEach(el => {
        // select = categorical (keep as string); input = numerical (parse as float)
        fixedValues[el.dataset.feat] = el.tagName === 'SELECT' ? el.value : parseFloat(el.value);
    });

    const features = window.currentColumns
        .filter(c => window.columnRoles[c] === 'feature' && (window.boFeatureConfigs[c] ? window.boFeatureConfigs[c].included : true))
        .map(c => {
            const cfg = window.boFeatureConfigs[c] || {};
            return {
                name: c,
                type: cfg.type || 'continuous',
                range: cfg.range || ''
            };
        });

    const objectives = window.currentColumns
        .filter(c => window.columnRoles[c] === 'objective' && (window.objectiveConfigs[c] ? window.objectiveConfigs[c].included : true))
        .map(c => {
            const cfg = window.objectiveConfigs[c] || {};
            return {
                name: c,
                type: cfg.type || 'maximize',
                target: cfg.target || '',
                importance: cfg.importance || 0
            };
        });

    if (features.length < 2 || objectives.length === 0) {
        try { Plotly.purge(plotContainer); } catch(e) {}
        plotContainer.innerHTML = '';
        return;
    }

    // Ensure colorObj and zAxis are valid
    const validObjNames = objectives.map(o => o.name);
    const effectiveColorObj = validObjNames.includes(colorObj) ? colorObj : validObjNames[0];
    const effectiveZAxis = validObjNames.includes(zAxis) ? zAxis : effectiveColorObj;

    const tweaks = {
        kernel: document.getElementById('kernel-select')?.value || 'matern52'
    };

    // Cancel any previous active request
    if (currentSurfaceAbortController) {
        try { currentSurfaceAbortController.abort(); } catch(e) {}
    }
    currentSurfaceAbortController = new AbortController();
    const thisRequestId = ++currentSurfaceRequestId;

    // Silent background computation — no spinner injected into DOM
    try {
        const response = await fetch('/bo_surface', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: currentSurfaceAbortController.signal,
            body: JSON.stringify({
                data: window.currentData,
                columns: window.currentColumns,
                features,
                objectives,
                tweaks,
                x_feat: xFeat,
                y_feat: yFeat,
                z_axis: effectiveZAxis,
                color_obj: effectiveColorObj,
                fixed_values: fixedValues,
                resolution: res
            })
        });

        if (thisRequestId !== currentSurfaceRequestId) return;

        const result = await response.json();
        if (result.error) {
            try { Plotly.purge(plotContainer); } catch(e) {}
            plotContainer.innerHTML = `<div style="color:red;padding:1rem;">Surface calculation error: ${result.error}</div>`;
            return;
        }

        const surfaceTrace = {
            x: result.x,
            y: result.y,
            z: result.z,
            surfacecolor: result.color,
            type: 'surface',
            colorscale: 'Viridis',
            contours: {
                z: { show: true, usecolormap: true, highlightcolor: "#42f462", project: { z: false } }
            },
            colorbar: { title: result.color_obj, len: 0.8 },
            hovertemplate: `<b>${result.x_feat}:</b> %{x}<br><b>${result.y_feat}:</b> %{y}<br><b>${result.z_axis}:</b> %{z:.3f}<br><b>${result.color_obj}:</b> %{surfacecolor:.3f}<extra></extra>`
        };

        const inSliceExps = result.experiments.filter(e => e.in_slice);
        const offSliceExps = result.experiments.filter(e => !e.in_slice);

        const traces = [surfaceTrace];

        if (inSliceExps.length > 0) {
            traces.push({
                x: inSliceExps.map(e => e.x),
                y: inSliceExps.map(e => e.y),
                z: inSliceExps.map(e => e.z),
                mode: 'markers',
                type: 'scatter3d',
                name: 'In-Slice Experiments (Solid)',
                marker: {
                    size: 6,
                    color: '#ef4444',
                    line: { width: 1, color: '#ffffff' }
                },
                customdata: inSliceExps.map(e => [e.exp_id, e.color, e.conditions]),
                hovertemplate: `<b>Experiment #%{customdata[0]}</b> [In-Slice]<br>${result.x_feat}: %{x}<br>${result.y_feat}: %{y}<br>${result.z_axis}: %{z}<br>${result.color_obj}: %{customdata[1]}<br>%{customdata[2]}<extra></extra>`
            });
        }

        if (offSliceExps.length > 0) {
            traces.push({
                x: offSliceExps.map(e => e.x),
                y: offSliceExps.map(e => e.y),
                z: offSliceExps.map(e => e.z),
                mode: 'markers',
                type: 'scatter3d',
                name: 'Other Experiments (Off-Slice)',
                marker: {
                    size: 5,
                    color: 'rgba(239, 68, 68, 0.15)',
                    line: { width: 1.5, color: '#ef4444' }
                },
                customdata: offSliceExps.map(e => [e.exp_id, e.color, e.conditions]),
                hovertemplate: `<b>Experiment #%{customdata[0]}</b> [Off-Slice]<br>${result.x_feat}: %{x}<br>${result.y_feat}: %{y}<br>${result.z_axis}: %{z}<br>${result.color_obj}: %{customdata[1]}<br>%{customdata[2]}<extra></extra>`
            });
        }

        try { Plotly.purge(plotContainer); } catch(e) {}
        Plotly.newPlot(plotContainer, traces, {
            title: `3D Surface: ${result.z_axis} vs (${result.x_feat}, ${result.y_feat}) [Color: ${result.color_obj}]`,
            scene: {
                xaxis: { title: result.x_feat },
                yaxis: { title: result.y_feat },
                zaxis: { title: result.z_axis }
            },
            margin: { l: 0, r: 0, t: 60, b: 0 },
            height: 550,
            legend: { orientation: 'h', y: -0.1 }
        }, { responsive: true });


    } catch (err) {
        if (err.name === 'AbortError') return;
        if (thisRequestId !== currentSurfaceRequestId) return;
        try { Plotly.purge(plotContainer); } catch(e) {}
        plotContainer.innerHTML = `<div style="color:red;padding:1rem;">Failed to calculate surface: ${err.message}</div>`;
        console.error(err);
    }
}

export function renderParetoPlot() {
    const objCols   = window.currentColumns.filter(c => window.columnRoles[c] === 'objective' && (window.objectiveConfigs[c] ? window.objectiveConfigs[c].included : true));
    const featCols  = window.currentColumns.filter(c => window.columnRoles[c] === 'feature');
    const section   = document.getElementById('pareto-plot-section');
    const plotEl    = document.getElementById('pareto-plot');
    const tableCont = document.getElementById('pareto-table-container');
    const exportBtn = document.getElementById('pareto-export-btn');
    const titleEl   = document.getElementById('pareto-title');

    if (objCols.length < 2) {
        try { Plotly.purge('pareto-plot'); } catch(e) {}
        section.classList.add('hidden');
        return;
    }

    const completeData = window.currentData.filter(row =>
        objCols.every(col => {
            const val = row[window.currentColumns.indexOf(col)];
            return val !== '' && !isNaN(parseFloat(val));
        })
    );

    if (completeData.length === 0) {
        try { Plotly.purge('pareto-plot'); } catch(e) {}
        section.classList.add('hidden');
        return;
    }

    section.classList.remove('hidden');
    const paretoIndices = detectParetoIndices(completeData, objCols);
    const isPareto      = new Array(completeData.length).fill(false);
    paretoIndices.forEach(idx => isPareto[idx] = true);

    const getHoverText = (idx) => {
        let text = `<b>Experiment ${idx + 1}</b><br><br>`;
        objCols.forEach(col  => { text += `${col}: ${completeData[idx][window.currentColumns.indexOf(col)]}<br>`; });
        text += `<br><i>Conditions:</i><br>`;
        featCols.forEach(col => { text += `${col}: ${completeData[idx][window.currentColumns.indexOf(col)]}<br>`; });
        return text;
    };

    const transformPoints = (indices, colsToUse) => {
        return indices.map(idx => {
            const p = {};
            colsToUse.forEach(col => {
                const cfg    = window.objectiveConfigs[col];
                const rawVal = parseFloat(completeData[idx][window.currentColumns.indexOf(col)]) || 0;
                p[col] = (cfg && cfg.type === 'target') ? Math.abs(rawVal - parseFloat(cfg.target || 0)) : rawVal;
            });
            return p;
        });
    };

    const getAxisTitle  = (col) => {
        const cfg = window.objectiveConfigs[col] || { type: 'maximize', target: 0, importance: 100 };
        return cfg.type === 'target'
            ? `Dist. to ${cfg.target || 0} (${col})`
            : `${col} (${cfg.type === 'maximize' ? 'Max' : 'Min'})`;
    };

    const getAxisLayout = (col) => {
        const cfg    = window.objectiveConfigs[col] || { type: 'maximize' };
        const layout = { title: getAxisTitle(col) };
        if (cfg.type === 'minimize' || cfg.type === 'target') layout.autorange = 'reversed';
        return layout;
    };

    if (objCols.length > 3) {
        titleEl.textContent = "Pareto Optimal Conditions";
        plotEl.classList.add('hidden');
        tableCont.classList.remove('hidden');
        exportBtn.classList.remove('hidden');

        const head = document.getElementById('pareto-table-head');
        const body = document.getElementById('pareto-table-body');

        head.innerHTML = '<tr>' + window.currentColumns.map(c => `<th>${c}</th>`).join('') + '</tr>';
        body.innerHTML = paretoIndices.map(idx => {
            const row = completeData[idx];
            return '<tr>' + row.map(v => `<td>${v}</td>`).join('') + '</tr>';
        }).join('');

    } else {
        titleEl.textContent = "Pareto Front";
        plotEl.classList.remove('hidden');
        tableCont.classList.add('hidden');
        exportBtn.classList.add('hidden');

        let traces = [];

        if (objCols.length === 2) {
            const dominatedIndices = completeData.map((_, i) => i).filter(i => !isPareto[i]);
            const optimalIndices   = paretoIndices;

            const dpDominated = transformPoints(dominatedIndices, objCols);
            const dpOptimal   = transformPoints(optimalIndices,   objCols);

            traces = [
                {
                    x: dpDominated.map(p => p[objCols[0]]),
                    y: dpDominated.map(p => p[objCols[1]]),
                    customdata:    dominatedIndices.map(i => getHoverText(i)),
                    hovertemplate: '%{customdata}<extra></extra>',
                    mode: 'markers', name: 'Dominated', type: 'scatter',
                    marker: { size: 8, color: '#94a3b8', opacity: 0.5 }
                },
                {
                    x: dpOptimal.map(p => p[objCols[0]]),
                    y: dpOptimal.map(p => p[objCols[1]]),
                    customdata:    optimalIndices.map(i => getHoverText(i)),
                    hovertemplate: '%{customdata}<extra></extra>',
                    mode: 'markers', name: 'Pareto Optimal', type: 'scatter',
                    marker: { size: 12, color: '#F97316', symbol: 'circle', line: { width: 1, color: '#EA580C' } }
                }
            ];

            Plotly.newPlot('pareto-plot', traces, {
                title:  `Pareto Front: ${objCols[0]} vs ${objCols[1]}`,
                xaxis:  getAxisLayout(objCols[0]),
                yaxis:  getAxisLayout(objCols[1]),
                autosize: true, margin: { l: 60, r: 40, t: 80, b: 60 },
                legend: { orientation: 'h', y: -0.2 }
            }, { responsive: true });

        } else {
            const cols             = objCols.slice(0, 3);
            const dominatedIndices = completeData.map((_, i) => i).filter(i => !isPareto[i]);
            const optimalIndices   = paretoIndices;

            const dpDominated = transformPoints(dominatedIndices, cols);
            const dpOptimal   = transformPoints(optimalIndices,   cols);

            traces = [
                {
                    x: dpDominated.map(p => p[cols[0]]),
                    y: dpDominated.map(p => p[cols[1]]),
                    z: dpDominated.map(p => p[cols[2]]),
                    customdata: dominatedIndices.map(i => getHoverText(i)),
                    hovertemplate: '%{customdata}<extra></extra>',
                    mode: 'markers', name: 'Dominated', type: 'scatter3d',
                    marker: { size: 4, color: '#94a3b8', opacity: 0.3 }
                },
                {
                    x: dpOptimal.map(p => p[cols[0]]),
                    y: dpOptimal.map(p => p[cols[1]]),
                    z: dpOptimal.map(p => p[cols[2]]),
                    customdata: optimalIndices.map(i => getHoverText(i)),
                    hovertemplate: '%{customdata}<extra></extra>',
                    mode: 'markers', name: 'Pareto Optimal', type: 'scatter3d',
                    marker: { size: 7, color: '#F97316', symbol: 'circle' }
                }
            ];

            Plotly.newPlot('pareto-plot', traces, {
                title:  `Pareto Front (3D): ${cols.join(', ')}`,
                scene: {
                    xaxis: getAxisLayout(cols[0]),
                    yaxis: getAxisLayout(cols[1]),
                    zaxis: getAxisLayout(cols[2])
                },
                autosize: true, margin: { l: 0, r: 0, t: 80, b: 0 },
                legend: { orientation: 'h', y: -0.1 }
            }, { responsive: true });
        }

        setTimeout(() => Plotly.Plots.resize('pareto-plot'), 150);
    }
}

// Bind to window for inline calls
window.syncBOFeatureConfig = (col, field) => {
    const tr = document.querySelector(`.bo-f-type[data-col="${col}"]`)?.closest('tr');
    if (!tr || !window.boFeatureConfigs[col]) return;

    if (field === 'type'    || !field) window.boFeatureConfigs[col].type     = tr.querySelector('.bo-f-type').value;
    if (field === 'range'   || !field) window.boFeatureConfigs[col].range    = tr.querySelector('.bo-f-range').value;
    if (field === 'include' || !field) window.boFeatureConfigs[col].included = tr.querySelector('.bo-f-include').checked;
    updateAllConstraintsDropdowns('bo');
};
