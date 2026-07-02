export function renderSASetup() {
    const list    = document.getElementById('sa-features-list');
    const objList = document.getElementById('sa-objectives-list');

    let featHtml = '';
    window.currentColumns.filter(c => window.columnRoles[c] === 'feature').forEach(col => {
        const cfg = window.saFeatureConfigs[col] || { type: 'continuous', included: true };
        featHtml += `<tr>
            <td>${col}</td>
            <td><select class="sa-f-type" data-col="${col}" onchange="window.syncSAFeatureConfig('${col}', 'type')">
                <option value="continuous" ${cfg.type === 'continuous' ? 'selected' : ''}>Numerical</option>
                <option value="discrete"   ${cfg.type === 'discrete'   ? 'selected' : ''}>Discrete</option>
                <option value="categorical"${cfg.type === 'categorical'? 'selected' : ''}>Categorical</option>
            </select></td>
            <td><input type="checkbox" class="sa-f-include" data-col="${col}" ${cfg.included ? 'checked' : ''} onchange="window.syncSAFeatureConfig('${col}', 'include')"> Include</td>
        </tr>`;
    });
    list.innerHTML = featHtml;

    let objHtml    = '';
    const selectedObjs = window.currentColumns.filter(col => window.columnRoles[col] === 'objective');
    selectedObjs.forEach(col => {
        const cfg       = window.objectiveConfigs[col] || { type: 'maximize', target: '', importance: (100/selectedObjs.length).toFixed(1), included: true };
        const isDisabled = !cfg.included;
        objHtml += `<tr class="${isDisabled ? 'row-disabled' : ''}">
            <td>${col}</td>
            <td><select class="sa-o-type" data-col="${col}" onchange="window.syncObjectiveConfig('${col}', 'sa', 'type'); window.toggleRowTarget(this)" ${isDisabled ? 'disabled' : ''}>
                <option value="maximize" ${cfg.type === 'maximize' ? 'selected' : ''}>maximize</option>
                <option value="minimize" ${cfg.type === 'minimize' ? 'selected' : ''}>minimize</option>
                <option value="target"   ${cfg.type === 'target'   ? 'selected' : ''}>target</option>
            </select></td>
            <td><input type="text" class="sa-o-target ${cfg.type === 'target' ? '' : 'hidden'}" data-col="${col}" value="${cfg.target || ''}" placeholder="Target" onchange="window.syncObjectiveConfig('${col}', 'sa', 'target')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="number" class="sa-o-importance" data-col="${col}" value="${cfg.importance}" onchange="window.syncObjectiveConfig('${col}', 'sa', 'importance')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="checkbox" class="sa-o-include" data-col="${col}" ${cfg.included ? 'checked' : ''} onchange="window.syncObjectiveConfig('${col}', 'sa', 'include')"> Include</td>
        </tr>`;
    });
    objList.innerHTML = objHtml;
}

export async function runSA() {
    const features = Array.from(document.querySelectorAll('.sa-f-include:checked')).map(cb => {
        const row = cb.closest('tr');
        return {
            name:  cb.dataset.col,
            type:  row.querySelector('.sa-f-type').value
        };
    });

    const objectives = Array.from(document.querySelectorAll('#sa-objectives-list tr'))
        .filter(tr => tr.querySelector('.sa-o-include').checked)
        .map(tr => ({
            name:       tr.querySelector('.sa-o-type').dataset.col,
            type:       tr.querySelector('.sa-o-type').value,
            target:     tr.querySelector('.sa-o-target').value,
            importance: tr.querySelector('.sa-o-importance').value
        }));

    window.saSelectedObjectives = objectives.map(o => o.name);
    window.saSelectedFeatures   = Array.from(document.querySelectorAll('.sa-f-include:checked')).map(cb => cb.dataset.col);

    const tweaks = {
        model: document.getElementById('sa-model-select').value,
        params: {
            n_estimators: document.getElementById('sa-rf-trees').value,
            max_depth:    document.getElementById('sa-rf-depth').value,
            mlp_layers:   document.getElementById('sa-mlp-layers').value,
            mlp_iter:     document.getElementById('sa-mlp-iter').value
        }
    };

    document.getElementById('loading-text').textContent = "Analyzing...";
    document.getElementById('loading-overlay').classList.remove('hidden');
    document.getElementById('sa-estimate-results').innerHTML = '';

    window.currentAbortController = new AbortController();
    try {
        const res    = await fetch('/sa', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ data: window.currentData, columns: window.currentColumns, features, objectives, tweaks }),
            signal: window.currentAbortController.signal
        });
        const result = await res.json();
        document.getElementById('loading-overlay').classList.add('hidden');
        if (result.error) return alert("SA Error: " + result.error);

        window.lastSAResult = result;
        renderSAResults(result);
    } catch(err) {
        document.getElementById('loading-overlay').classList.add('hidden');
        if (err.name === 'AbortError') return console.log('SA calculation aborted by user.');
        alert("A critical error occurred during SA: " + err.message);
        console.error(err);
    } finally {
        window.currentAbortController = null;
    }
}

export function renderSAResults(result) {
    document.getElementById('sa-results-container').classList.remove('hidden');

    const numCorr  = result.numerical_correlation;
    const numFeats = result.num_features;
    const objNames = result.extended_obj_names;

    if (numFeats.length > 0) {
        const zNum = objNames.map(obj => numFeats.map(f => (numCorr[f] || {})[obj] ?? 0));
        Plotly.newPlot('sa-num-corr',
            [{ z: zNum, x: numFeats, y: objNames, type: 'heatmap',
               colorscale: 'RdBu', zmin: -1, zmax: 1,
               text: zNum.map(row => row.map(v => v.toFixed(2))),
               hovertemplate: 'Feature: %{x}<br>Objective: %{y}<br>Correlation: %{z:.3f}<extra></extra>',
               texttemplate: '%{text}', showscale: true }],
            {
                xaxis: { title: 'Numerical Features',  automargin: true },
                yaxis: { title: 'Success Targets',      automargin: true },
                margin: { l: 150, b: 80, t: 30, r: 50 },
                height: 400
            });
    } else {
        document.getElementById('sa-num-corr').innerHTML = '<p class="text-secondary"><i>No numerical features available for correlation.</i></p>';
    }

    const catContainer = document.getElementById('sa-cat-interactions-container');
    const catControls  = document.getElementById('sa-cat-controls');
    catContainer.innerHTML = '';
    catControls.innerHTML  = '';

    const interactions = result.cat_interactions;
    const pairs        = Object.keys(interactions);

    if (pairs.length > 0) {
        catControls.innerHTML = `<div class="tweak-item">
            <label>Select Pair:</label>
            <select id="sa-cat-pair-select" style="width: 100%;">
                ${pairs.map(p => `<option value="${p}">${p}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Select Objective:</label>
            <select id="sa-cat-obj-select" style="width: 100%;">
                ${objNames.map(o => `<option value="${o}">${o}</option>`).join('')}
            </select>
        </div>`;

        const updateInteractionPlot = () => {
            const pair = document.getElementById('sa-cat-pair-select').value;
            const obj  = document.getElementById('sa-cat-obj-select').value;
            const data = (interactions[pair] || {})[obj];
            if (!data) {
                catContainer.innerHTML = '<p class="text-secondary"><i>No data found for this combination.</i></p>';
                return;
            }
            const trace = {
                z: data.z, x: data.x, y: data.y,
                type: 'heatmap', colorscale: 'RdBu', zmin: 0, zmax: 1,
                hovertemplate: `<b>Best Achievement</b><br>${pair.split(' vs ')[1]}: %{x}<br>${pair.split(' vs ')[0]}: %{y}<br>Success Score: %{z:.3f}<extra></extra>`
            };
            Plotly.newPlot(catContainer, [trace], {
                title: `Optimal Outcome: ${obj} (Max Success)`,
                xaxis: { title: pair.split(' vs ')[1], automargin: true },
                yaxis: { title: pair.split(' vs ')[0], automargin: true },
                margin: { l: 150, b: 80, t: 50, r: 50 },
                height: 500
            });
        };

        document.getElementById('sa-cat-pair-select').addEventListener('change', updateInteractionPlot);
        document.getElementById('sa-cat-obj-select').addEventListener('change',  updateInteractionPlot);
        updateInteractionPlot();
    } else {
        catContainer.innerHTML = '<p class="text-secondary"><i>No categorical interactions possible (requires 2+ categorical features).</i></p>';
    }

    const parcoordsContainer = document.getElementById('sa-parcoords-container');
    parcoordsContainer.innerHTML = '';

    const featCols = (window.saSelectedFeatures || window.currentColumns.filter(c => window.columnRoles[c] === 'feature'));
    const objCols  = (window.saSelectedObjectives || window.currentColumns.filter(c => window.columnRoles[c] === 'objective'));

    if (featCols.length > 0 && objCols.length > 0 && window.currentData.length > 0) {
        objCols.forEach((objCol, idx) => {
            const plotCols = [...featCols, objCol];

            const dimensions = plotCols.map(col => {
                const colIdx  = window.currentColumns.indexOf(col);
                const rawVals = window.currentData.map(r => r[colIdx]);
                const numVals = rawVals.map(v => parseFloat(v));
                const isCat   = numVals.some(isNaN);

                if (isCat) {
                    const uniqueVals = [...new Set(rawVals.filter(v => v !== '' && v !== null))];
                    return {
                        label: col,
                        values:   rawVals.map(v => uniqueVals.indexOf(v) >= 0 ? uniqueVals.indexOf(v) : 0),
                        tickvals: uniqueVals.map((_, i) => i),
                        ticktext: uniqueVals
                    };
                } else {
                    return { label: col, values: numVals.map(v => isNaN(v) ? 0 : v) };
                }
            });

            const objIdx    = window.currentColumns.indexOf(objCol);
            const colorVals = window.currentData.map(r => parseFloat(r[objIdx]) || 0);

            const pcDiv = document.createElement('div');
            pcDiv.id    = `sa-parcoords-${idx}`;
            pcDiv.classList.add('plot-container');
            pcDiv.style.cssText = 'width: 100%; height: 420px; margin-bottom: 2rem;';
            parcoordsContainer.appendChild(pcDiv);

            const finiteColorVals = colorVals.filter(v => isFinite(v));
            const cmin = finiteColorVals.length > 0 ? Math.min(...finiteColorVals) : 0;
            const cmax = finiteColorVals.length > 0 ? Math.max(...finiteColorVals) : 1;

            Plotly.newPlot(`sa-parcoords-${idx}`, [{
                type: 'parcoords',
                line: {
                    color: colorVals,
                    colorscale: [
                        [0, '#0d0887'], [0.11, '#46039f'], [0.22, '#7201a8'], [0.33, '#9c179e'],
                        [0.44, '#bd3786'], [0.55, '#d8576b'], [0.66, '#ed7953'], [0.77, '#fb9f3a'],
                        [0.88, '#fdca26'], [1, '#f0f921']
                    ],
                    showscale: true, reversescale: false, opacity: 0.85,
                    cmin, cmax,
                    colorbar: { title: objCol, thickness: 15 }
                },
                unselected: { line: { opacity: 0, color: 'rgba(0,0,0,0)' } },
                dimensions
            }], {
                title: `Parallel Coordinates — Colored by: ${objCol}`,
                margin: { l: 80, r: 80, t: 120, b: 20 }
            }, { responsive: true });
        });
    } else {
        parcoordsContainer.innerHTML = '<p class="text-secondary"><i>Parallel coordinates require at least one feature and one objective with experiments loaded.</i></p>';
    }

    const impactContainer = document.getElementById('sa-impact-container');
    impactContainer.innerHTML = '';

    const globalCatMap = {};
    objNames.forEach(obj => {
        const res = result.results[obj];
        if (res && res.shap_data && res.shap_data.features) {
            res.shap_data.features.forEach((feat, i) => {
                const fVals = res.shap_data.feature_values[i];
                let isCategorical = fVals.some(v => isNaN(parseFloat(v)));
                if (isCategorical) {
                    if (!globalCatMap[feat]) {
                        globalCatMap[feat] = new Set();
                    }
                    fVals.forEach(v => globalCatMap[feat].add(v));
                }
            });
        }
    });
    Object.keys(globalCatMap).forEach(feat => {
        globalCatMap[feat] = Array.from(globalCatMap[feat]).sort();
    });

    objNames.forEach((obj, idx) => {
        const res = result.results[obj];
        if (!res || !res.importance) return;

        const imp           = res.importance;
        const sortedEntries = Object.entries(imp).sort((a, b) => a[1] - b[1]);
        const yFeats        = sortedEntries.map(e => e[0]);
        const xVals         = sortedEntries.map(e => e[1]);

        const container   = document.createElement('div');
        container.classList.add('impact-row');
        container.style.cssText = 'display:flex;flex-wrap:wrap;gap:15px;margin-bottom:2.5rem;padding:1rem;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;';

        const titleDiv        = document.createElement('h3');
        titleDiv.textContent  = `Feature Impact: ${obj}`;
        titleDiv.style.cssText = 'width:100%;margin:0 0 1rem 0;';
        container.appendChild(titleDiv);

        const barId  = `sa-bar-${idx}`;
        const barDiv = document.createElement('div');
        barDiv.id    = barId;
        barDiv.style.cssText = 'flex:1;min-width:350px;height:400px;';
        container.appendChild(barDiv);

        const shapId  = `sa-shap-${idx}`;
        const shapDiv = document.createElement('div');
        shapDiv.id    = shapId;
        shapDiv.style.cssText = 'flex:1.3;min-width:450px;height:400px;';
        container.appendChild(shapDiv);

        impactContainer.appendChild(container);

        const hasShap = res.shap_data && res.shap_data.features && res.shap_data.features.length > 0;

        Plotly.newPlot(barId,
            [{ x: xVals, y: yFeats, type: 'bar', orientation: 'h',
               marker: { color: obj === 'global_success' ? '#FFD700' : 'var(--accent-color)' } }],
            { xaxis: { title: 'Importance (%)', range: [0, 100] },
              yaxis: { automargin: true },
              margin: { l: 120, b: 50, t: 20, r: 20 },
              showlegend: false, height: 400 },
            { responsive: true }
        );

        if (hasShap) {
            const shapTraces = [];
            const sd = res.shap_data;
            sd.features.forEach((feat, i) => {
                const sVals = sd.shap_values[i];
                const fVals = sd.feature_values[i];

                let numericFVals = fVals.map(v => parseFloat(v));
                let isCategorical = numericFVals.some(isNaN);
                let colorArray, cmin, cmax, colorscale, reversescale;

                if (isCategorical) {
                    const uniqueStr = globalCatMap[feat] || [...new Set(fVals)].sort();
                    colorArray   = fVals.map(v => uniqueStr.indexOf(v));
                    colorscale   = 'Portland'; reversescale = false;
                } else {
                    colorArray  = numericFVals;
                    colorscale  = 'RdBu';
                    cmin        = Math.min(...colorArray);
                    cmax        = Math.max(...colorArray);
                    reversescale = true;
                }

                const yBase = yFeats.indexOf(feat);
                shapTraces.push({
                    x: sVals,
                    y: sVals.map(() => yBase + (Math.random() - 0.5) * 0.4),
                    mode: 'markers', type: 'scatter',
                    marker: {
                        size: 7, opacity: 0.6, color: colorArray,
                        colorscale, cmin, cmax, reversescale,
                        line: { width: 0.5, color: '#fff' }
                    },
                    customdata: fVals,
                    hovertemplate: `<b>Value:</b> %{customdata}<br><b>Impact:</b> %{x:.3f}<extra></extra>`
                });
            });

            Plotly.newPlot(shapId, shapTraces, {
                xaxis: { title: 'SHAP Value (Impact)', zeroline: true, zerolinecolor: '#94a3b8' },
                yaxis: { showticklabels: false, range: [-0.5, yFeats.length - 0.5] },
                margin: { l: 20, b: 50, t: 20, r: 20 },
                showlegend: false, height: 400
            }, { responsive: true });
        } else {
            shapDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;">No SHAP data available for this model.</div>';
        }
    });

    const getFittingSuggestion = (status) => {
        switch(status) {
            case 'Good':    return 'Reliable model for interpolation.';
            case 'Limited': return 'Moderate predictive power; use with caution.';
            case 'Poor':    return 'Low predictive power; consider removing irrelevant features or conducting more experiments to improve data coverage.';
            default:        return '';
        }
    };

    let relHtml = `
        <div style="margin-bottom: 10px; font-weight: 600; color: var(--accent-color);">
            Active Model: ${document.getElementById('sa-model-select').options[document.getElementById('sa-model-select').selectedIndex].text}
        </div>
        <div style="margin-bottom: 12px; font-size: 0.82rem; color: var(--text-secondary); background: var(--bg-secondary, #f8fafc); border-left: 3px solid var(--accent-color); padding: 8px 12px; border-radius: 4px;">
            ℹ️ <b>Note:</b> R² is measured on the normalized success space [0–1], not on raw objective values.
            A high R² means the model reliably predicts relative performance, not absolute measurements.
        </div>
        <table class="setup-table" style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <thead>
                <tr style="text-align: left; border-bottom: 2px solid var(--border-color);">
                    <th style="padding: 10px;">Objective</th>
                    <th style="padding: 10px;">Predictive Power (R²) <span title="R² is computed on the normalized success space [0–1]. It reflects how well the model predicts relative performance, not raw measurement values." style="cursor:help; color:var(--text-secondary); font-size:0.85em;">ⓘ</span></th>
                    <th style="padding: 10px;">Status</th>
                    <th style="padding: 10px;">Suggestion</th>
                </tr>
            </thead>
            <tbody>`;

    objNames.forEach(obj => {
        const r          = result.results[obj];
        const statusClass = r.fit_flag.toLowerCase();
        let nameDisplay   = `<b>${obj}</b>`;
        if (obj === 'global_success') nameDisplay = `<span style="color: var(--accent-color);">✨ <b>Global Success Index</b></span>`;
        if (obj === 'pareto_optimal') nameDisplay = `<i>Pareto Optimal Flag</i>`;

        relHtml += `
            <tr style="border-bottom: 1px solid var(--border-color);">
                <td style="padding: 10px;">${nameDisplay}</td>
                <td style="padding: 10px;">${r.reliability ?? 'N/A'}</td>
                <td style="padding: 10px;"><span class="badge ${statusClass}">${r.fit_flag}</span></td>
                <td style="padding: 10px; font-size: 0.9rem; color: var(--text-secondary);">${getFittingSuggestion(r.fit_flag)}</td>
            </tr>`;
    });
    document.getElementById('sa-reliability-info').innerHTML = relHtml;

    document.getElementById('sa-cat-interactions-container').style.minHeight = '500px';

    const estDiv = document.getElementById('sa-estimator-inputs');
    const feats  = window.saSelectedFeatures || window.currentColumns.filter(c => window.columnRoles[c] === 'feature');

    estDiv.innerHTML = feats.map(f => {
        const colIdx     = window.currentColumns.indexOf(f);
        const uniqueVals = [...new Set(window.currentData.map(r => r[colIdx]).filter(v => v !== null && v !== ''))];
        const isNumeric  = uniqueVals.every(v => !isNaN(parseFloat(v)) && isFinite(v));

        if (isNumeric) {
            return `<div class="tweak-item">
                <label>${f}</label>
                <input type="number" class="sa-est-input" data-col="${f}" step="any">
            </div>`;
        } else {
            const options = uniqueVals.map(v => `<option value="${v}">${v}</option>`).join('');
            return `<div class="tweak-item">
                <label>${f}</label>
                <select class="sa-est-input" data-col="${f}">${options}</select>
            </div>`;
        }
    }).join('');
}

export async function runEstimate() {
    const inputs  = {};
    document.querySelectorAll('.sa-est-input').forEach(el => inputs[el.dataset.col] = el.value);

    const featList = (window.saSelectedFeatures || window.currentColumns.filter(c => window.columnRoles[c] === 'feature'))
        .map(c => {
            const cfg = window.saFeatureConfigs[c] || { type: 'continuous' };
            return {
                name: c,
                type: cfg.type
            };
        });

    const objList = Array.from(document.querySelectorAll('#sa-objectives-list tr'))
        .filter(tr => tr.querySelector('.sa-o-include').checked)
        .map(tr => ({
            name:       tr.querySelector('.sa-o-type').dataset.col,
            type:       tr.querySelector('.sa-o-type').value,
            target:     tr.querySelector('.sa-o-target').value,
            importance: tr.querySelector('.sa-o-importance').value
        }));

    document.getElementById('sa-estimate-results').innerHTML = '<div style="color: var(--text-secondary); animation: pulse 1.5s infinite;"><i>Calculating predictions...</i></div>';

    try {
        const res    = await fetch('/estimate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                data: window.currentData, columns: window.currentColumns,
                features: featList, objectives: objList, inputs,
                model: document.getElementById('sa-model-select').value,
                params: {
                    n_estimators: document.getElementById('sa-rf-trees').value,
                    max_depth:    document.getElementById('sa-rf-depth').value,
                    mlp_layers:   document.getElementById('sa-mlp-layers').value,
                    mlp_iter:     document.getElementById('sa-mlp-iter').value
                }
            })
        });
        const result = await res.json();
        if (result.error) {
            document.getElementById('sa-estimate-results').innerHTML = `<span style="color:red">Error: ${result.error}</span>`;
            return;
        }

        let html = '<div style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid var(--border-color);">';
        html += `<b>Global Success Score: <span style="font-size: 1.2rem; color: var(--accent-color);">${result.global_success}%</span></b></div>`;
        html += '<b>Predictions:</b><br>';
        for (const obj in result.predictions) {
            const val  = result.predictions[obj];
            const succ = result.success_scores[obj];
            html += `<div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span><b>${obj}:</b> ${val}</span>
                        <span style="color: ${succ > 80 ? '#22c55e' : (succ > 50 ? '#f59e0b' : '#ef4444')}">Success: ${succ}%</span>
                     </div>`;
        }
        document.getElementById('sa-estimate-results').innerHTML = html;
    } catch(err) {
        document.getElementById('sa-estimate-results').innerHTML = `<span style="color:red">Request failed: ${err.message}</span>`;
        console.error(err);
    }
}

export async function exportSAReport() {
    const overlay = document.getElementById('loading-overlay');
    document.getElementById('loading-text').textContent = "Generating Dynamic Report...";
    overlay.classList.remove('hidden');

    try {
        const timestamp   = new Date().toLocaleString();
        const activeModel = document.getElementById('sa-model-select').options[document.getElementById('sa-model-select').selectedIndex].text;

        const plotsData  = [];
        const capturePlot = (id, title) => {
            const el = document.getElementById(id);
            if (!el || !el.data) return;
            plotsData.push({ id, title, data: el.data, layout: el.layout });
        };

        capturePlot('sa-num-corr', 'Numerical Correlation Matrix');
        document.querySelectorAll('[id^="sa-parcoords-"]').forEach(el => {
            capturePlot(el.id, el.layout?.title?.text || el.layout?.title || 'Parallel Coordinates');
        });
        document.querySelectorAll('[id^="sa-bar-"], [id^="sa-shap-"]').forEach(el => {
            const h3 = el.parentElement ? el.parentElement.querySelector('h3') : null;
            const title = (h3 ? h3.textContent : null) || el.layout?.title?.text || el.layout?.title || 'Feature Impact Group';
            capturePlot(el.id, title);
        });

        const reliabilityHtml  = document.getElementById('sa-reliability-info').innerHTML;
        const catInteractions  = window.lastSAResult ? window.lastSAResult.cat_interactions : {};
        const objNames         = window.lastSAResult ? window.lastSAResult.extended_obj_names : [];

        let reportHtml = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EDOS Analysis Report - ${timestamp}</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"><\/script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; color: #1e293b; background: #f8fafc; max-width: 1200px; margin: 40px auto; line-height: 1.6; padding: 0 20px; }
        .card { background: white; border-radius: 12px; border: 1px solid #e2e8f0; padding: 25px; margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .header { border-bottom: 2px solid #3b82f6; padding-bottom: 20px; margin-bottom: 40px; display: flex; justify-content: space-between; align-items: flex-end; }
        h1 { color: #0f172a; margin: 0; font-size: 2.2rem; }
        .meta { color: #64748b; font-size: 0.95rem; text-align: right; }
        h2 { color: #1e293b; margin-top: 0; border-left: 5px solid #3b82f6; padding-left: 15px; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th { background: #f1f5f9; text-align: left; padding: 12px; border: 1px solid #e2e8f0; font-weight: 600; }
        td { padding: 12px; border: 1px solid #e2e8f0; }
        .badge { padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
        .good { background: #dcfce7; color: #166534; }
        .limited { background: #fef9c3; color: #854d0e; }
        .poor { background: #fee2e2; color: #991b1b; }
        .impact-group { display: flex; flex-wrap: wrap; gap: 20px; }
        .plot-box { flex: 1; min-width: 450px; height: 450px; }
        .full-plot { width: 100%; height: 500px; }
        .controls { display: flex; gap: 15px; margin-bottom: 20px; background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; }
        .tweak-item { flex: 1; display: flex; flex-direction: column; gap: 5px; }
        select { padding: 8px; border-radius: 6px; border: 1px solid #cbd5e1; background: white; font-family: inherit; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>EDOS Analysis Report</h1>
            <p style="margin: 5px 0 0 0; color: #64748b;">Statistical Analysis Model Results</p>
        </div>
        <div class="meta">
            <b>Model:</b> ${activeModel}<br>
            <b>Date:</b> ${timestamp}
        </div>
    </div>

    <div class="card">
        <h2>Model Reliability &amp; Validation</h2>
        ${reliabilityHtml}
    </div>

    <div class="card" id="num-corr-section" style="display:none;">
        <h2>Numerical Correlation Matrix</h2>
        <div id="sa-num-corr" class="full-plot"></div>
    </div>

    <div class="card" id="cat-section" style="display:none;">
        <h2>Categorical Interaction Analysis</h2>
        <div class="controls">
            <div class="tweak-item">
                <label>Select Pair:</label>
                <select id="report-cat-pair">
                    ${Object.keys(catInteractions).map(p => `<option value="${p}">${p}</option>`).join('')}
                </select>
            </div>
            <div class="tweak-item">
                <label>Select Objective:</label>
                <select id="report-cat-obj">
                    ${objNames.map(o => `<option value="${o}">${o}</option>`).join('')}
                </select>
            </div>
        </div>
        <div id="sa-cat-interactions-container" class="full-plot"></div>
    </div>

    <div id="parcoords-section"></div>
    <div id="impact-section"></div>

    <script>
        const plotsData      = ${JSON.stringify(plotsData)};
        const catInteractions = ${JSON.stringify(catInteractions)};

        const corrPlot = plotsData.find(p => p.id === 'sa-num-corr');
        if (corrPlot) {
            document.getElementById('num-corr-section').style.display = 'block';
            Plotly.newPlot('sa-num-corr', corrPlot.data, corrPlot.layout, { responsive: true });
        }

        if (Object.keys(catInteractions).length > 0) {
            document.getElementById('cat-section').style.display = 'block';
            const updateCatPlot = () => {
                const pair = document.getElementById('report-cat-pair').value;
                const obj  = document.getElementById('report-cat-obj').value;
                const data = (catInteractions[pair] || {})[obj];
                if (!data) return;
                const trace = { z: data.z, x: data.x, y: data.y, type: 'heatmap', colorscale: 'RdBu', zmin: 0, zmax: 1 };
                const layout = {
                    title: \`Optimal Outcome: \${obj}\`,
                    xaxis: { title: pair.split(' vs ')[1], automargin: true },
                    yaxis: { title: pair.split(' vs ')[0], automargin: true },
                    margin: { l: 150, b: 80, t: 50, r: 50 }, height: 500
                };
                Plotly.newPlot('sa-cat-interactions-container', [trace], layout, { responsive: true });
            };
            document.getElementById('report-cat-pair').addEventListener('change', updateCatPlot);
            document.getElementById('report-cat-obj').addEventListener('change',  updateCatPlot);
            updateCatPlot();
        }

        const pcSection = document.getElementById('parcoords-section');
        const pcPlots   = plotsData.filter(p => p.id.startsWith('sa-parcoords'));
        if (pcPlots.length > 0) {
            const head = document.createElement('h2');
            head.textContent = 'Parallel Coordinates Analysis';
            pcSection.appendChild(head);
            pcPlots.forEach(p => {
                const div = document.createElement('div');
                div.id = p.id; div.className = 'card full-plot';
                pcSection.appendChild(div);
                Plotly.newPlot(p.id, p.data, p.layout, { responsive: true });
            });
        }

        const impactSection = document.getElementById('impact-section');
        const barPlots      = plotsData.filter(p => p.id.startsWith('sa-bar'));
        if (barPlots.length > 0) {
            const head = document.createElement('h2');
            head.textContent = 'Feature Impact Analysis';
            impactSection.appendChild(head);
            barPlots.forEach((p, idx) => {
                const group   = document.createElement('div');
                group.className = 'card';
                group.innerHTML = '<h3>' + (p.title || p.layout.title?.text || p.layout.title || 'Feature Impact Group') + '</h3><div class="impact-group"></div>';
                const container = group.querySelector('.impact-group');

                const barDiv = document.createElement('div');
                barDiv.id = p.id; barDiv.className = 'plot-box';
                container.appendChild(barDiv);

                const shapPlot = plotsData.find(pd => pd.id === 'sa-shap-' + p.id.split('-')[2]);
                if (shapPlot) {
                    const shapDiv = document.createElement('div');
                    shapDiv.id = shapPlot.id; shapDiv.className = 'plot-box';
                    container.appendChild(shapDiv);
                }

                impactSection.appendChild(group);
                Plotly.newPlot(p.id, p.data, p.layout, { responsive: true });

                if (shapPlot) {
                    const finalLayout = Object.assign({}, shapPlot.layout);
                    finalLayout.yaxis = Object.assign({}, finalLayout.yaxis, { showticklabels: true });
                    Plotly.newPlot(shapPlot.id, shapPlot.data, finalLayout, { responsive: true });
                }
            });
        }
    <\/script>
</body>
</html>`;

        const blob = new Blob([reportHtml], { type: 'text/html' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `EDOS_Report_${new Date().toISOString().slice(0,10)}.html`;
        a.click();

    } catch (err) {
        alert("Dynamic report generation failed: " + err.message);
        console.error(err);
    } finally {
        overlay.classList.add('hidden');
    }
}

window.syncSAFeatureConfig = (col, field) => {
    const tr = document.querySelector(`.sa-f-type[data-col="${col}"]`)?.closest('tr');
    if (!tr || !window.saFeatureConfigs[col]) return;

    if (field === 'type'    || !field) window.saFeatureConfigs[col].type     = tr.querySelector('.sa-f-type').value;
    if (field === 'include' || !field) window.saFeatureConfigs[col].included = tr.querySelector('.sa-f-include').checked;
};

// Bind to window for inline calls
window.syncSAFeatureConfig = window.syncSAFeatureConfig;
