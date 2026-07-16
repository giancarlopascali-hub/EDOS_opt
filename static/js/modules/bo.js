import { exportData, detectParetoIndices } from '../utils.js';
import { updateAllConstraintsDropdowns } from './constraints.js';

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
            const succ = result.success_scores[obj];
            html += `<div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span><b>${obj}:</b> ${val}</span>
                        <span style="color: ${succ > 80 ? '#22c55e' : (succ > 50 ? '#f59e0b' : '#ef4444')}">Success: ${succ}%</span>
                     </div>`;
        }
        resultsDiv.innerHTML = html;
    } catch(err) {
        resultsDiv.innerHTML = `<span style="color:red">Request failed: ${err.message}</span>`;
        console.error(err);
    }
}

export function renderTrendPlot() {
    if (!window.currentData || window.currentData.length === 0) {
        try { Plotly.purge('trend-plot'); } catch(e) {}
        const visuals = document.getElementById('bo-visuals');
        if (visuals) visuals.classList.add('hidden');
        return;
    }

    const objCols = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');

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

    if (traces.length === 0) {
        document.getElementById('bo-visuals').classList.add('hidden');
        return;
    }

    document.getElementById('bo-visuals').classList.remove('hidden');

    Plotly.newPlot('trend-plot', traces, {
        title:    objCols.length === 1 ? `Trend: ${objCols[0]}` : 'Objectives Trend',
        autosize: true,
        height:   400,
        margin:   { l: 60, r: 40, t: 80, b: 60 },
        legend:   { orientation: 'h', y: -0.2 }
    }, { responsive: true });

    setTimeout(() => { Plotly.Plots.resize('trend-plot'); }, 100);
    renderParetoPlot();
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
