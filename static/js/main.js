import './state.js';
import { safeAddListener, exportData, detectParetoIndices, rebalanceRemaining, rebalanceObjectiveConfigs, updateRangePlaceholder } from './utils.js';
import { runDoE, renderDoEResults, rebalanceDoEDOM, attachDeleteEvents } from './modules/doe.js';
import { renderBOSetup, runBO, runBOEstimate, renderTrendPlot, renderParetoPlot, updateAcqSelectForMultiObj } from './modules/bo.js';
import { renderSASetup, runSA, runEstimate, exportSAReport, abortSA } from './modules/sa.js';
import { updateAllConstraintsDropdowns } from './modules/constraints.js';

// Bind utilities to window for inline calls
window.rebalanceRemaining = rebalanceRemaining;
window.rebalanceObjectiveConfigs = rebalanceObjectiveConfigs;

// DOMContentLoaded Entrypoint
document.addEventListener('DOMContentLoaded', () => {
    // Inject runtime CSS for row-disabled state
    const style = document.createElement('style');
    style.textContent = `
        .row-disabled { opacity: 0.55; background-color: #f1f5f9; }
        .row-disabled input:not([type="checkbox"]), .row-disabled select { 
            background: #e2e8f0; color: #94a3b8; cursor: not-allowed; pointer-events: none; 
        }
        .row-disabled td { color: #94a3b8; }
    `;
    document.head.appendChild(style);

    initModuleSwitcher();
    initFileUpload();
    initTabs();
    initButtons();
    initSettingsListeners();
    initStrategyToggle();

    // Abort button listener
    safeAddListener('abort-calc-btn', 'click', () => {
        if (window.currentAbortController) {
            window.currentAbortController.abort();
            window.currentAbortController = null;
        }
        document.getElementById('loading-overlay').classList.add('hidden');
    });
    safeAddListener('sa-abort-btn', 'click', abortSA);

    // Clear Data button listener
    safeAddListener('clear-data-btn', 'click', () => {
        if (confirm('Are you sure you want to clear all loaded data and reset the module configurations? This cannot be undone.')) {
            abortSA();
            window.clearObjectiveConstraints?.();
            window.currentData      = null;
            window.currentColumns   = [];
            window.columnRoles      = {};
            window.objectiveConfigs = {};
            window.boFeatureConfigs = {};
            window.saFeatureConfigs = {};
            window.suggestionsData  = null;
            window.currentDoEFeatures   = [];
            window.currentDoEObjectives = [];

            const resultsTableBody = document.getElementById('results-table-body');
            if (resultsTableBody) resultsTableBody.innerHTML = '';
            const doeResultsBody = document.getElementById('doe-results-body');
            if (doeResultsBody) doeResultsBody.innerHTML = '';

            try {
                if (document.getElementById('trend-plot'))  Plotly.purge('trend-plot');
                if (document.getElementById('bo-contour-plot')) Plotly.purge('bo-contour-plot');
                if (document.getElementById('bo-surface-plot')) Plotly.purge('bo-surface-plot');
                if (document.getElementById('pareto-plot')) Plotly.purge('pareto-plot');
                if (document.getElementById('sa-num-corr')) Plotly.purge('sa-num-corr');
                document.querySelectorAll('[id^="sa-parcoords-"]').forEach(el => Plotly.purge(el.id));
                document.querySelectorAll('[id^="sa-bar-"]').forEach(el => Plotly.purge(el.id));
                document.querySelectorAll('[id^="sa-shap-"]').forEach(el => Plotly.purge(el.id));
            } catch(e) { console.error("Error purging plots:", e); }

            renderMainTable();
            renderSetup();

            document.getElementById('data-section').classList.add('hidden');
            document.getElementById('upload-section').classList.remove('hidden');

            document.getElementById('doe-results-section').classList.add('hidden');
            document.getElementById('results-section').classList.add('hidden');
            document.getElementById('sa-results-container').classList.add('hidden');
            if (document.getElementById('bo-visualization-section')) document.getElementById('bo-visualization-section').classList.add('hidden');
            if (document.getElementById('bo-visuals')) document.getElementById('bo-visuals').classList.add('hidden');
            if (document.getElementById('pareto-plot-section')) document.getElementById('pareto-plot-section').classList.add('hidden');
        }
    });
});

function initStrategyToggle() {
    const radios = document.querySelectorAll('input[name="bo-strategy"]');
    radios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            const strategy      = e.target.value;
            const banner        = document.getElementById('bo-warning-banner');
            const warningText   = document.getElementById('bo-warning-text');
            const noiselessCb   = document.getElementById('bo-noiseless');
            const avoidRevalCb  = document.getElementById('bo-avoid-reval');
            let reclassified    = [];

            if (strategy === 'exhaustive_grid') {
                if (noiselessCb)  { noiselessCb.disabled  = true; noiselessCb.closest('.tweak-item').style.opacity  = '0.5'; }
                if (avoidRevalCb) { avoidRevalCb.disabled = true; avoidRevalCb.closest('.tweak-item').style.opacity = '0.5'; }

                document.querySelectorAll('.bo-f-type').forEach(select => {
                    if (select.value === 'continuous') {
                        const col        = select.dataset.col;
                        const tr         = select.closest('tr');
                        const rangeInput = tr.querySelector('.bo-f-range');

                        select.value = 'regular';
                        let val = rangeInput.value.trim();
                        if (val.includes('[') && val.includes(']')) {
                            if (!val.match(/\]\d+/)) rangeInput.value = val + "10";
                        } else {
                            const matches = val.match(/[-+]?\d*\.?\d+/g);
                            if (matches && matches.length >= 2) rangeInput.value = `[${matches[0]}, ${matches[1]}]10`;
                        }

                        window.syncBOFeatureConfig(col, 'type');
                        window.syncBOFeatureConfig(col, 'range');
                        reclassified.push(col);
                    }
                });

                if (reclassified.length > 0) {
                    banner.classList.remove('hidden');
                    warningText.innerHTML = `**Exhaustive Grid selected:** ${reclassified.join(', ')} reclassified as **Regular** (10 steps) to define the search space.`;
                }
            } else {
                if (noiselessCb)  { noiselessCb.disabled  = false; noiselessCb.closest('.tweak-item').style.opacity  = '1'; }
                if (avoidRevalCb) { avoidRevalCb.disabled = false; avoidRevalCb.closest('.tweak-item').style.opacity = '1'; }

                document.querySelectorAll('.bo-f-type').forEach(select => {
                    if (select.value === 'regular') {
                        const col        = select.dataset.col;
                        const tr         = select.closest('tr');
                        const rangeInput = tr.querySelector('.bo-f-range');

                        select.value      = 'continuous';
                        rangeInput.value  = rangeInput.value.replace(/\]\d+$/, ']');

                        window.syncBOFeatureConfig(col, 'type');
                        window.syncBOFeatureConfig(col, 'range');
                        reclassified.push(col);
                    }
                });
                banner.classList.add('hidden');
            }
        });
    });
}

function initModuleSwitcher() {
    document.querySelectorAll('.module-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            switchModule(btn.dataset.module);
        });
    });
}

export function switchModule(module) {
    console.log(`Switching to module: ${module}`);
    window.currentModule = module;

    document.querySelectorAll('.module-btn').forEach(b => b.classList.toggle('active', b.dataset.module === module));
    document.querySelectorAll('.module-view').forEach(v => v.classList.toggle('hidden', v.id !== `${module}-module`));

    if (module === 'bo' || module === 'sa') {
        document.getElementById('upload-section').classList.remove('hidden');
        if (window.currentData) {
            document.getElementById('data-section').classList.remove('hidden');
            if (module === 'bo') renderTrendPlot();
        }
    } else {
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('data-section').classList.add('hidden');
    }
    renderSetup();
}
window.switchModule = switchModule;

function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const parent = btn.closest('section, .module-view');
            parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            parent.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.remove('hidden');
        });
    });
}

function initFileUpload() {
    safeAddListener('file-input', 'change', (e) => {
        if (e.target.files.length > 0) {
            handleFiles(e.target.files);
            e.target.value = '';
        }
    });

    const dropZone = document.getElementById('drop-zone');
    if (dropZone) {
        dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            handleFiles(e.dataTransfer.files);
        });
    }
}

// A value is numeric only when its complete trimmed representation is finite.
// parseFloat('10 mg') returns 10, which would incorrectly classify labels that
// begin with a number as continuous features.
function detectColumnIsNumeric(vals) {
    return vals.length > 0 && vals.every(v => {
        const text = String(v).trim();
        return text !== '' && Number.isFinite(Number(text));
    });
}

function computeColumnNumericRange(vals) {
    const numericVals = vals.map(v => Number(String(v).trim()));
    return {
        min: Math.min(...numericVals),
        max: Math.max(...numericVals)
    };
}

async function handleFiles(files) {
    if (files.length === 0) return;
    abortSA();
    window.clearObjectiveConstraints?.();
    const formData = new FormData();
    formData.append('file', files[0]);

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const result   = await response.json();
        if (result.error) throw new Error(result.error);

        window.currentData    = result.data;
        window.currentColumns = result.columns;

        window.columnRoles      = {};
        window.objectiveConfigs = {};
        window.currentColumns.forEach((col, idx) => {
            window.columnRoles[col] = (idx === window.currentColumns.length - 1) ? 'objective' : 'feature';
        });

        const objCols = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');
        objCols.forEach(col => {
            window.objectiveConfigs[col] = {
                name:       col,
                type:       'maximize',
                target:     '',
                importance: (100 / objCols.length).toFixed(1),
                included:   true
            };
        });

        window.boFeatureConfigs = {};
        window.saFeatureConfigs = {};
        window.currentColumns.forEach((col, idx) => {
            if (window.columnRoles[col] === 'feature') {
                const vals  = window.currentData.map(r => r[idx]).filter(v => v != null && v !== '');
                const isNum = detectColumnIsNumeric(vals);
                const { min = 0, max = 1 } = isNum ? computeColumnNumericRange(vals) : {};
                const range = isNum ? `[${min}, ${max}]` : [...new Set(vals)].join(', ');

                window.boFeatureConfigs[col] = { type: isNum ? 'continuous' : 'categorical', range, included: true };
                window.saFeatureConfigs[col] = { type: isNum ? 'continuous' : 'categorical', included: true };
            }
        });

        renderMainTable();
        renderSetup();
        if (window.currentModule === 'bo') renderTrendPlot();

        document.getElementById('data-section').classList.remove('hidden');
    } catch (err) {
        alert('Error loading file: ' + err.message);
    }
}

export function renderMainTable() {
    const head = document.getElementById('main-table-head');
    const body = document.getElementById('main-table-body');
    if (!head || !body) return;

    let headHtml = '<tr>';
    window.currentColumns.forEach(col => {
        const role = window.columnRoles[col];
        headHtml += `
            <th>
                <div class="col-header-name">${col}</div>
                <div class="type-toggle" onclick="window.toggleColumnRole('${col}')">
                    <span class="${role === 'feature'   ? 'active' : ''}">F</span>
                    <span class="${role === 'objective' ? 'active' : ''}">O</span>
                </div>
            </th>`;
    });
    headHtml += '</tr>';
    head.innerHTML = headHtml;

    let bodyHtml = '';
    if (window.currentData) {
        window.currentData.forEach((row, rowIdx) => {
            bodyHtml += '<tr>' +
                row.map((v, colIdx) =>
                    `<td contenteditable="true" onblur="window.updateCellData(${rowIdx}, ${colIdx}, this.textContent)">${v}</td>`
                ).join('') +
                '</tr>';
        });
    }
    body.innerHTML = bodyHtml;
}
window.renderMainTable = renderMainTable;

window.updateCellData = (rowIdx, colIdx, value) => {
    window.currentData[rowIdx][colIdx] = value;
    if (window.currentModule === 'bo') renderTrendPlot();
};

window.toggleColumnRole = (colName) => {
    const oldRole = window.columnRoles[colName];
    const newRole = oldRole === 'feature' ? 'objective' : 'feature';
    window.columnRoles[colName] = newRole;

    if (newRole === 'feature') {
        const idx   = window.currentColumns.indexOf(colName);
        const vals  = window.currentData.map(r => r[idx]).filter(v => v != null && v !== '');
        const isNum = detectColumnIsNumeric(vals);
        const { min = 0, max = 1 } = isNum ? computeColumnNumericRange(vals) : {};
        const range = isNum ? `[${min}, ${max}]` : [...new Set(vals)].join(', ');

        if (!window.boFeatureConfigs[colName]) window.boFeatureConfigs[colName] = { type: isNum ? 'continuous' : 'categorical', range, included: true };
        if (!window.saFeatureConfigs[colName]) window.saFeatureConfigs[colName] = { type: isNum ? 'continuous' : 'categorical', included: true };
    } else {
        delete window.boFeatureConfigs[colName];
        delete window.saFeatureConfigs[colName];
    }

    const activeObjs = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');
    activeObjs.forEach(col => {
        if (!window.objectiveConfigs[col]) {
            window.objectiveConfigs[col] = { name: col, type: 'maximize', target: '', importance: 0, included: true };
        }
    });
    Object.keys(window.objectiveConfigs).forEach(col => {
        if (!activeObjs.includes(col)) delete window.objectiveConfigs[col];
    });

    rebalanceObjectiveConfigs();
    renderMainTable();
    renderSetup();
    if (window.currentModule === 'bo') renderTrendPlot();
};

export function renderSetup() {
    if      (window.currentModule === 'doe') renderDoESetup();
    else if (window.currentModule === 'bo')  renderBOSetup();
    else if (window.currentModule === 'sa')  renderSASetup();
}
window.renderSetup = renderSetup;

function renderDoESetup() {
    // Driven by DOM manually
}

window.updateObjectiveName = (oldName, newName) => {
    if (oldName === newName) return;
    if (window.objectiveConfigs[oldName]) {
        window.objectiveConfigs[newName]      = window.objectiveConfigs[oldName];
        window.objectiveConfigs[newName].name = newName;
        delete window.objectiveConfigs[oldName];
        renderSetup();
    }
};

window.syncObjectiveConfig = (col, sourceModule, field) => {
    if (sourceModule === 'doe') return;

    const rolePrefix = sourceModule === 'bo' ? 'bo-o' : 'sa-o';
    const row        = document.querySelector(`.${rolePrefix}-type[data-col="${col}"]`).closest('tr');

    document.querySelectorAll(`.${rolePrefix}-include`).forEach(cb => {
        const c = cb.dataset.col;
        const r = cb.closest('tr');
        if (window.objectiveConfigs[c]) {
            window.objectiveConfigs[c].type      = r.querySelector(`.${rolePrefix}-type`).value;
            window.objectiveConfigs[c].target    = r.querySelector(`.${rolePrefix}-target`).value;
            if (field === 'importance') window.objectiveConfigs[c].importance = r.querySelector(`.${rolePrefix}-importance`).value;
            window.objectiveConfigs[c].included  = cb.checked;
        }
    });

    if (field === 'importance') {
        const selector = `.${rolePrefix}-importance`;
        rebalanceRemaining(col, row.querySelector(selector).value, selector);
        document.querySelectorAll(selector).forEach(el => {
            if (window.objectiveConfigs[el.dataset.col]) window.objectiveConfigs[el.dataset.col].importance = el.value;
        });
    } else if (field === 'include' || field === 'type') {
        rebalanceObjectiveConfigs();
    }

    renderSetup();
    updateAcqSelectForMultiObj();
    if (window.currentModule === 'bo') renderTrendPlot();
};

function initButtons() {
    safeAddListener('doe-add-feature-btn', 'click', () => {
        const list = document.getElementById('doe-features-list');
        const tr   = document.createElement('tr');
        tr.innerHTML = `
            <td><input type="text" class="doe-f-name" value="Feature_${list.children.length + 1}"></td>
            <td><select class="doe-f-type" onchange="window.updatePlaceholder(this)">
                <option value="continuous">continuous</option>
                <option value="regular">regular</option>
                <option value="discrete">discrete</option>
                <option value="categorical">categorical</option>
            </select></td>
            <td><input type="text" class="doe-f-range" placeholder="[min, max]"></td>
            <td><button class="btn secondary small delete-row">×</button></td>`;
        list.appendChild(tr);
        attachDeleteEvents();
    });

    safeAddListener('doe-add-objective-btn', 'click', () => {
        const list  = document.getElementById('doe-objectives-list');
        const count = list.children.length + 1;
        const tr    = document.createElement('tr');
        tr.innerHTML = `
            <td><input type="text" class="doe-o-name" value="Objective_${count}"></td>
            <td><select class="doe-o-type" onchange="window.toggleRowTarget(this)">
                <option value="maximize">maximize</option>
                <option value="minimize">minimize</option>
                <option value="target">target</option>
            </select></td>
            <td><input type="text" class="doe-o-target hidden" placeholder="Target"></td>
            <td><input type="number" class="doe-o-importance" value="0"></td>
            <td><button class="btn secondary small delete-row">×</button></td>`;
        list.appendChild(tr);
        attachDeleteEvents();
        rebalanceDoEDOM();
    });

    safeAddListener('doe-run-btn',      'click', runDoE);
    safeAddListener('run-btn',          'click', runBO);
    safeAddListener('sa-run-btn',       'click', runSA);
    safeAddListener('sa-estimate-btn',  'click', runEstimate);
    safeAddListener('bo-estimate-btn',  'click', runBOEstimate);

    safeAddListener('doe-add-num-constraint-btn', 'click', () => window.addNumericalConstraintRow('doe'));
    safeAddListener('doe-add-disallowed-constraint-btn', 'click', () => window.addDisallowedCombinationRow('doe'));
    safeAddListener('bo-add-num-constraint-btn', 'click', () => window.addNumericalConstraintRow('bo'));
    safeAddListener('bo-add-disallowed-constraint-btn', 'click', () => window.addDisallowedCombinationRow('bo'));

    const doeFeatList = document.getElementById('doe-features-list');
    if (doeFeatList) {
        doeFeatList.addEventListener('input', () => updateAllConstraintsDropdowns('doe'));
        doeFeatList.addEventListener('change', () => updateAllConstraintsDropdowns('doe'));
    }

    safeAddListener('shutdown-btn', 'click', async () => {
        if (confirm('Are you sure you want to shut down the EDOS server?')) {
            try {
                await fetch('/shutdown', { method: 'POST' });
            } catch (_) {}
            document.body.innerHTML = `
                <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;background:#0f172a;color:#e2e8f0;font-family:sans-serif;gap:20px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18.4 6.1a9 9 0 1 1-12.8 0"/><line x1="12" y1="2" x2="12" y2="12"/></svg>
                    <h1 style="font-size:2rem;margin:0;">Server Stopped</h1>
                    <p style="color:#64748b;max-width:400px;text-align:center;">The EDOS server has been shut down. You can safely close this browser tab.</p>
                </div>`;
        }
    });

    safeAddListener('export-main-btn', 'click', () => exportData(window.currentData, window.currentColumns));

    safeAddListener('commit-suggestions-btn', 'click', (e) => {
        if (e.target.disabled) return;
        const rows    = document.querySelectorAll('#results-table-body tr');
        const featCols = window.currentColumns.filter(c => window.columnRoles[c] === 'feature');
        const objCols  = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');

        let addedCount = 0;
        rows.forEach(tr => {
            const cells  = tr.querySelectorAll('td');
            const newRow = new Array(window.currentColumns.length).fill('');

            featCols.forEach((col, i) => {
                newRow[window.currentColumns.indexOf(col)] = cells[i].textContent.trim();
            });
            objCols.forEach((col, i) => {
                const val = cells[featCols.length + i].textContent.trim();
                newRow[window.currentColumns.indexOf(col)] = (val === 'N/D') ? '' : val;
            });

            window.currentData.push(newRow);
            addedCount++;
        });

        if (addedCount > 0) {
            renderMainTable();
            renderTrendPlot();
            renderParetoPlot();
            alert(`Added ${addedCount} experiments to the main dataset. Plots refreshed.`);
            document.getElementById('results-section').classList.add('hidden');
        }
    });

    safeAddListener('doe-commit-suggestions-btn', 'click', () => {
        if (!window.suggestionsData || window.suggestionsData.length === 0) return;
        const rows = document.querySelectorAll('#doe-results-body tr');
        if (rows.length === 0) return;

        const feats = Object.keys(window.suggestionsData[0]);
        const objs  = window.currentDoEObjectives.map(o => o.name);

        if (!window.currentData || window.currentColumns.length === 0) {
            window.currentColumns = [...feats, ...objs];
            window.currentData    = [];
            feats.forEach(col => { window.columnRoles[col] = 'feature'; });
            objs.forEach(col  => { window.columnRoles[col] = 'objective'; });
        }

        window.currentDoEFeatures.forEach(f => {
            if (window.currentColumns.includes(f.name)) {
                window.boFeatureConfigs[f.name] = { type: f.type, range: f.range, included: true };
                window.saFeatureConfigs[f.name] = { type: f.type, included: true };
            }
        });
        window.currentDoEObjectives.forEach(o => {
            if (window.currentColumns.includes(o.name) && !window.objectiveConfigs[o.name]) {
                window.objectiveConfigs[o.name] = { name: o.name, type: 'maximize', target: '', importance: (100/objs.length).toFixed(1), included: true };
            }
        });

        const colIndices = {};
        let missing = [];
        [...feats, ...objs].forEach(name => {
            const idx = window.currentColumns.indexOf(name);
            if (idx === -1) missing.push(name);
            else colIndices[name] = idx;
        });

        if (missing.length > 0) {
            alert(`Cannot commit: The current main dataset is missing columns: ${missing.join(', ')}`);
            return;
        }

        let addedCount = 0;
        rows.forEach(tr => {
            const cells  = tr.querySelectorAll('td');
            const newRow = new Array(window.currentColumns.length).fill('');

            feats.forEach((name, i) => { newRow[colIndices[name]] = cells[i].textContent.trim(); });
            objs.forEach((name,  i) => {
                const val = cells[feats.length + i].textContent.trim();
                newRow[colIndices[name]] = (val === 'N/D') ? '' : val;
            });

            window.currentData.push(newRow);
            addedCount++;
        });

        if (addedCount > 0) {
            renderMainTable();
            renderSetup();
            alert(`Added ${addedCount} experiments to the main dataset.`);
            document.getElementById('doe-results-section').classList.add('hidden');
            document.getElementById('data-section').classList.remove('hidden');
            switchModule('bo');
        }
    });

    safeAddListener('doe-export-btn', 'click', () => {
        if (!window.suggestionsData || window.suggestionsData.length === 0) return;
        const rows  = document.querySelectorAll('#doe-results-body tr');
        if (rows.length === 0) return;
        const feats = Object.keys(window.suggestionsData[0]);
        const cols  = [...feats, ...window.currentDoEObjectives.map(o => o.name)];
        const exportArr = Array.from(rows).map(tr => Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim()));
        exportData(exportArr, cols);
    });

    safeAddListener('export-results-btn', 'click', () => {
        if (!window.suggestionsData || window.suggestionsData.length === 0) return;
        const rows     = document.querySelectorAll('#results-table-body tr');
        if (rows.length === 0) return;
        const featCols = window.currentColumns.filter(c => window.columnRoles[c] === 'feature');
        const objCols  = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');
        const cols     = [...featCols, ...objCols.map(c => `${c} (Result)`)];
        const exportArr = Array.from(rows).map(tr => Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim()));
        exportData(exportArr, cols);
    });

    safeAddListener('pareto-export-btn', 'click', () => {
        const objCols      = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');
        const paretoIndices = detectParetoIndices(window.currentData, objCols);
        const paretoData   = window.currentData.filter((_, i) => paretoIndices.includes(i));
        exportData(paretoData, window.currentColumns);
    });

    const applyHighlight = (tableId, type) => {
        const table = document.getElementById(tableId);
        if (!table) return;
        const ths = table.querySelectorAll('thead th');
        ths.forEach((th, idx) => {
            let matches = false;
            if (type === 'feature') {
                if (tableId === 'doe-results-table') {
                    const feats = window.suggestionsData && window.suggestionsData.length > 0 ? Object.keys(window.suggestionsData[0]) : [];
                    matches = idx < feats.length;
                } else {
                    matches = !th.textContent.includes('(Result)');
                }
            } else if (type === 'objective') {
                if (tableId === 'doe-results-table') {
                    const feats = window.suggestionsData && window.suggestionsData.length > 0 ? Object.keys(window.suggestionsData[0]) : [];
                    matches = idx >= feats.length;
                } else {
                    matches = th.textContent.includes('(Result)');
                }
            }
            if (matches) {
                th.classList.add(type === 'feature' ? 'highlight-feature-col' : 'highlight-objective-col');
                table.querySelectorAll('tbody tr').forEach(tr => {
                    const td = tr.children[idx];
                    if (td) td.classList.add(type === 'feature' ? 'highlight-feature-col' : 'highlight-objective-col');
                });
            }
        });
    };

    const removeHighlight = (tableId) => {
        const table = document.getElementById(tableId);
        if (table) table.querySelectorAll('.highlight-feature-col, .highlight-objective-col').forEach(el => {
            el.classList.remove('highlight-feature-col', 'highlight-objective-col');
        });
    };

    safeAddListener('doe-export-features-btn', 'mouseenter', () => applyHighlight('doe-results-table', 'feature'));
    safeAddListener('doe-export-features-btn', 'mouseleave', () => removeHighlight('doe-results-table'));
    safeAddListener('export-features-btn',     'mouseenter', () => applyHighlight('results-table', 'feature'));
    safeAddListener('export-features-btn',     'mouseleave', () => removeHighlight('results-table'));
    safeAddListener('doe-import-results-btn',  'mouseenter', () => applyHighlight('doe-results-table', 'objective'));
    safeAddListener('doe-import-results-btn',  'mouseleave', () => removeHighlight('doe-results-table'));
    safeAddListener('import-results-btn',      'mouseenter', () => applyHighlight('results-table', 'objective'));
    safeAddListener('import-results-btn',      'mouseleave', () => removeHighlight('results-table'));

    safeAddListener('doe-export-features-btn', 'click', () => {
        if (!window.suggestionsData || window.suggestionsData.length === 0) return;
        const rows  = document.querySelectorAll('#doe-results-body tr');
        if (rows.length === 0) return;
        const feats = Object.keys(window.suggestionsData[0]);
        const exportArr = Array.from(rows).map(tr =>
            Array.from(tr.querySelectorAll('td')).slice(0, feats.length).map(td => td.textContent.trim())
        );
        exportData(exportArr, feats);
    });

    safeAddListener('export-features-btn', 'click', () => {
        if (!window.suggestionsData || window.suggestionsData.length === 0) return;
        const rows     = document.querySelectorAll('#results-table-body tr');
        if (rows.length === 0) return;
        const featCols = window.currentColumns.filter(c => window.columnRoles[c] === 'feature');
        const exportArr = Array.from(rows).map(tr =>
            Array.from(tr.querySelectorAll('td')).slice(0, featCols.length).map(td => td.textContent.trim())
        );
        exportData(exportArr, featCols);
    });

    safeAddListener('doe-import-results-btn', 'click', () => document.getElementById('doe-import-results-input').click());
    safeAddListener('doe-import-results-input', 'change', (e) =>
        handleResultsImport(e, 'doe-results-table', window.currentDoEObjectives.map(o => o.name), false)
    );

    safeAddListener('import-results-btn',   'click', () => document.getElementById('import-results-input').click());
    safeAddListener('import-results-input', 'change', (e) =>
        handleResultsImport(e, 'results-table', window.currentColumns.filter(c => window.columnRoles[c] === 'objective'), true)
    );
}

function handleResultsImport(e, tableId, expectedObjCols, autoCommit) {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = '';

    const reader = new FileReader();
    reader.onload = (event) => {
        const text    = event.target.result;
        const lines   = text.trim().split('\n');
        if (lines.length < 2) {
            alert('CSV should contain at least a header row and one data row.');
            return;
        }

        const headers  = lines[0].split(',').map(h => h.trim().replace(/^['"]|['"]$/g, ''));
        const dataRows = lines.slice(1).map(l => l.split(',').map(c => c.trim().replace(/^['"]|['"]$/g, '')));

        let colMap        = [];
        let useSequential = false;

        expectedObjCols.forEach(col => {
            const rawCol = col.replace(' (Result)', '');
            colMap.push(headers.findIndex(h => h === rawCol || h === col));
        });

        if (colMap.every(idx => idx === -1)) {
            if (headers.length >= expectedObjCols.length) {
                useSequential = true;
            } else {
                alert(`Could not match headers. Expected objective columns like: ${expectedObjCols.join(', ')}`);
                return;
            }
        }

        const table     = document.getElementById(tableId);
        const tbodyRows = table.querySelectorAll('tbody tr');
        let importedCount = 0;

        tbodyRows.forEach((tr, rIdx) => {
            if (rIdx < dataRows.length) {
                const rowData = dataRows[rIdx];
                const tds     = Array.from(tr.querySelectorAll('td'));
                const offset  = tds.length - expectedObjCols.length;

                expectedObjCols.forEach((col, cIdx) => {
                    const td = tds[offset + cIdx];
                    if (td) {
                        let val;
                        if (useSequential) {
                            const csvOffset = Math.max(0, rowData.length - expectedObjCols.length);
                            val = rowData[csvOffset + cIdx];
                        } else {
                            if (colMap[cIdx] !== -1) val = rowData[colMap[cIdx]];
                        }
                        if (val !== undefined && val !== '') {
                            td.textContent = val;
                            td.dispatchEvent(new Event('input', { bubbles: true }));
                        }
                    }
                });
                importedCount++;
            }
        });

        if (importedCount > 0) {
            alert(`Successfully imported results for ${importedCount} experiments.`);
            if (autoCommit) {
                setTimeout(() => {
                    const commitBtn = document.getElementById('commit-suggestions-btn');
                    if (commitBtn && !commitBtn.disabled) {
                        commitBtn.click();
                    } else if (commitBtn && commitBtn.disabled) {
                        console.warn('Commit button still disabled after import. Check for missing data.');
                    }
                }, 100);
            }
        }
    };
    reader.readAsText(file);
}

function initSettingsListeners() {
    const saModel = document.getElementById('sa-model-select');
    const updateSAParamsVisibility = () => {
        const val = saModel.value;
        document.querySelectorAll('.sa-rf-param').forEach(el  => el.classList.toggle('hidden', val !== 'random_forest'));
        document.querySelectorAll('.sa-mlp-param').forEach(el => el.classList.toggle('hidden', val !== 'mlp'));
    };
    saModel.addEventListener('change', updateSAParamsVisibility);
    updateSAParamsVisibility();

    safeAddListener('sa-export-report-btn', 'click', exportSAReport);

    const styleSlider = document.getElementById('style-slider');
    const styleNumber = document.getElementById('style-number');
    if (styleSlider && styleNumber) {
        styleSlider.addEventListener('input', (e) => styleNumber.value = e.target.value);
        styleNumber.addEventListener('input', (e) => styleSlider.value = e.target.value);
    }

    const attachNorm = (selector) => {
        document.addEventListener('change', (e) => {
            if (e.target.matches(selector)) {
                let inputs = Array.from(document.querySelectorAll(selector));
                const isDoe = selector.includes('doe');
                if (!isDoe) {
                    inputs = inputs.filter(el => {
                        const col = el.dataset.col;
                        return window.objectiveConfigs[col] && window.objectiveConfigs[col].included;
                    });
                }
                rebalanceRemaining(e.target, e.target.value, selector);

                if (!isDoe) {
                    Array.from(document.querySelectorAll(selector)).forEach(el => {
                        const col = el.dataset.col;
                        if (col && window.objectiveConfigs[col]) window.objectiveConfigs[col].importance = el.value;
                    });
                }

                if (!isDoe && window.currentModule === 'bo') renderTrendPlot();
            }
        });
    };
    attachNorm('.doe-o-importance');
    attachNorm('.bo-o-importance');
    attachNorm('.sa-o-importance');
}

window.toggleRowTarget = (select) => {
    const tr          = select.closest('tr');
    const targetInput = tr.querySelector('.bo-o-target, .doe-o-target, .sa-o-target');
    if (select.value === 'target') {
        targetInput.classList.remove('hidden');
    } else {
        targetInput.classList.add('hidden');
    }
};

window.updatePlaceholder = (select) => {
    const tr    = select.closest('tr');
    const input = tr.querySelector('.doe-f-range, .bo-f-range');
    updateRangePlaceholder(select.value, input);
};
