/**
 * EDOS Platform — Frontend Application Logic
 * ===========================================
 * Handles all client-side behavior for three core modules:
 *   - DoE  (Design of Experiments): generate experimental plans
 *   - BO   (Bayesian Optimization): suggest next experiments using a GP surrogate
 *   - SA   (Statistical Analysis): analyze feature–outcome relationships + SHAP
 *
 * Dependencies (loaded via CDN in index.html):
 *   - Plotly.js  — interactive charts
 *   - No external JS frameworks; vanilla JS only
 */

// ===========================================================================
// GLOBAL STATE
// ===========================================================================
// These variables persist for the lifetime of the page session.
// They are the single source of truth for data, column metadata,
// and module configuration at any point in time.

let currentModule  = 'doe';      // Active module: 'doe' | 'bo' | 'sa'
let currentData    = null;       // Full dataset: Array of Arrays (row-major)
let currentColumns = [];         // Column header names: Array of Strings
let columnRoles    = {};         // Map of column name → 'feature' | 'objective'
let objectiveConfigs = {};       // Objective settings: {name, type, target, importance, included}
let boFeatureConfigs = {};       // BO feature settings: {type, range, included}
let saFeatureConfigs = {};       // SA feature settings: {type, included}
let suggestionsData  = null;     // Latest DoE / BO results from the backend
let currentDoEFeatures   = [];   // Snapshot of DoE feature configs at the time of the last run
let currentDoEObjectives = [];   // Snapshot of DoE objective configs at the time of the last run
let currentAbortController = null; // AbortController to cancel in-flight fetch requests


// ===========================================================================
// INITIALIZATION (DOMContentLoaded)
// ===========================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Inject runtime CSS for row-disabled state (used when objectives are excluded)
    const style = document.createElement('style');
    style.textContent = `
        .row-disabled { opacity: 0.55; background-color: #f1f5f9; }
        .row-disabled input:not([type="checkbox"]), .row-disabled select { 
            background: #e2e8f0; color: #94a3b8; cursor: not-allowed; pointer-events: none; 
        }
        .row-disabled td { color: #94a3b8; }
    `;
    document.head.appendChild(style);

    // Wire up all interactive subsystems
    initModuleSwitcher();
    initFileUpload();
    initTabs();
    initButtons();
    initSettingsListeners();
    initStrategyToggle();

    // Abort button: cancels the current fetch and hides the loading overlay
    safeAddListener('abort-calc-btn', 'click', () => {
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
        }
        document.getElementById('loading-overlay').classList.add('hidden');
    });

    // Clear Data button: resets ALL state and hides all result sections
    safeAddListener('clear-data-btn', 'click', () => {
        if (confirm('Are you sure you want to clear all loaded data and reset the module configurations? This cannot be undone.')) {
            // Reset global state
            currentData      = null;
            currentColumns   = [];
            columnRoles      = {};
            objectiveConfigs = {};
            boFeatureConfigs = {};
            saFeatureConfigs = {};
            suggestionsData  = null;
            currentDoEFeatures   = [];
            currentDoEObjectives = [];

            // Clear result table bodies
            const resultsTableBody = document.getElementById('results-table-body');
            if (resultsTableBody) resultsTableBody.innerHTML = '';
            const doeResultsBody = document.getElementById('doe-results-body');
            if (doeResultsBody) doeResultsBody.innerHTML = '';

            // Release Plotly memory for all plot containers
            try {
                if (document.getElementById('trend-plot'))  Plotly.purge('trend-plot');
                if (document.getElementById('pareto-plot')) Plotly.purge('pareto-plot');
                if (document.getElementById('sa-num-corr')) Plotly.purge('sa-num-corr');
                document.querySelectorAll('[id^="sa-parcoords-"]').forEach(el => Plotly.purge(el.id));
                document.querySelectorAll('[id^="sa-bar-"]').forEach(el => Plotly.purge(el.id));
                document.querySelectorAll('[id^="sa-shap-"]').forEach(el => Plotly.purge(el.id));
            } catch(e) { console.error("Error purging plots:", e); }

            renderMainTable();
            renderSetup();

            // Show upload prompt; hide data section
            document.getElementById('data-section').classList.add('hidden');
            document.getElementById('upload-section').classList.remove('hidden');

            // Hide all result panels
            document.getElementById('doe-results-section').classList.add('hidden');
            document.getElementById('results-section').classList.add('hidden');
            document.getElementById('sa-results-container').classList.add('hidden');
            if (document.getElementById('bo-visuals')) document.getElementById('bo-visuals').classList.add('hidden');
        }
    });
});


// ===========================================================================
// BO STRATEGY TOGGLE — Exhaustive Grid vs Gradient-Based
// ===========================================================================

function initStrategyToggle() {
    /**
     * Responds to changes in the BO optimization strategy radio buttons.
     * When "Exhaustive Grid" is selected, continuous features must be reclassified
     * as "regular" (discrete grid) because continuous bounds cannot be exhaustively
     * enumerated.  Switching back to gradient-based restores them to continuous.
     */
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
                // Disable tweaks that don't apply to exhaustive search
                if (noiselessCb)  { noiselessCb.disabled  = true; noiselessCb.closest('.tweak-item').style.opacity  = '0.5'; }
                if (avoidRevalCb) { avoidRevalCb.disabled = true; avoidRevalCb.closest('.tweak-item').style.opacity = '0.5'; }

                // Auto-reclassify any 'continuous' features to 'regular' (10 steps default)
                document.querySelectorAll('.bo-f-type').forEach(select => {
                    if (select.value === 'continuous') {
                        const col        = select.dataset.col;
                        const tr         = select.closest('tr');
                        const rangeInput = tr.querySelector('.bo-f-range');

                        select.value = 'regular';
                        let val = rangeInput.value.trim();
                        // Append step count if not already present
                        if (val.includes('[') && val.includes(']')) {
                            if (!val.match(/\]\d+/)) rangeInput.value = val + "10";
                        } else {
                            const matches = val.match(/[-+]?\d*\.?\d+/g);
                            if (matches && matches.length >= 2) rangeInput.value = `[${matches[0]}, ${matches[1]}]10`;
                        }

                        syncBOFeatureConfig(col, 'type');
                        syncBOFeatureConfig(col, 'range');
                        reclassified.push(col);
                    }
                });

                if (reclassified.length > 0) {
                    banner.classList.remove('hidden');
                    warningText.innerHTML = `**Exhaustive Grid selected:** ${reclassified.join(', ')} reclassified as **Regular** (10 steps) to define the search space.`;
                }
            } else {
                // Re-enable tweaks for gradient-based mode
                if (noiselessCb)  { noiselessCb.disabled  = false; noiselessCb.closest('.tweak-item').style.opacity  = '1'; }
                if (avoidRevalCb) { avoidRevalCb.disabled = false; avoidRevalCb.closest('.tweak-item').style.opacity = '1'; }

                // Revert 'regular' features back to 'continuous' and strip the step count suffix
                document.querySelectorAll('.bo-f-type').forEach(select => {
                    if (select.value === 'regular') {
                        const col        = select.dataset.col;
                        const tr         = select.closest('tr');
                        const rangeInput = tr.querySelector('.bo-f-range');

                        select.value      = 'continuous';
                        rangeInput.value  = rangeInput.value.replace(/\]\d+$/, ']');  // Remove trailing step count

                        syncBOFeatureConfig(col, 'type');
                        syncBOFeatureConfig(col, 'range');
                        reclassified.push(col);
                    }
                });
                banner.classList.add('hidden');
            }
        });
    });
}


// ===========================================================================
// MODULE SWITCHER
// ===========================================================================

function initModuleSwitcher() {
    /** Wire click events to all module selector buttons. */
    document.querySelectorAll('.module-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            switchModule(btn.dataset.module);
        });
    });
}

function switchModule(module) {
    /**
     * Switches the active module view and refreshes the setup panel.
     * BO and SA both require uploaded data, so the upload section is shown
     * for those modules. DoE is data-independent (standalone).
     */
    console.log(`Switching to module: ${module}`);
    currentModule = module;

    // Update tab button active states
    document.querySelectorAll('.module-btn').forEach(b => b.classList.toggle('active', b.dataset.module === module));
    // Show/hide the corresponding module view
    document.querySelectorAll('.module-view').forEach(v => v.classList.toggle('hidden', v.id !== `${module}-module`));

    if (module === 'bo' || module === 'sa') {
        document.getElementById('upload-section').classList.remove('hidden');
        if (currentData) {
            document.getElementById('data-section').classList.remove('hidden');
            if (module === 'bo') renderTrendPlot();   // Refresh trend chart when switching to BO
        }
    } else {
        // DoE module: hide the data section as no data is required
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('data-section').classList.add('hidden');
    }
    renderSetup();
}


// ===========================================================================
// TAB NAVIGATION
// ===========================================================================

function initTabs() {
    /**
     * Generic tab system.
     * Each '.tab-btn' must have a data-tab attribute pointing to the ID of
     * the '.tab-content' panel it should reveal.
     */
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const parent = btn.closest('section, .module-view');
            // Deactivate all sibling tabs
            parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            parent.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            // Activate the clicked tab
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.remove('hidden');
        });
    });
}


// ===========================================================================
// FILE UPLOAD — Drag & Drop + File Input
// ===========================================================================

function initFileUpload() {
    /**
     * Sets up both the file <input> and a drag-and-drop zone.
     * Files are sent to /upload which parses the CSV server-side and returns
     * JSON.  The input is reset after each selection so re-uploading the same
     * file triggers a new change event.
     */
    safeAddListener('file-input', 'change', (e) => {
        if (e.target.files.length > 0) {
            handleFiles(e.target.files);
            e.target.value = '';  // Allow re-selection of the same file
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

async function handleFiles(files) {
    /**
     * Uploads the first selected file to /upload, then:
     *   - Stores the parsed data in currentData / currentColumns.
     *   - Auto-assigns the last column as an objective; all others as features.
     *   - Initializes BO/SA feature config stores from the data ranges.
     *   - Triggers a full re-render of the table and setup panels.
     */
    if (files.length === 0) return;
    const formData = new FormData();
    formData.append('file', files[0]);

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const result   = await response.json();
        if (result.error) throw new Error(result.error);

        currentData    = result.data;
        currentColumns = result.columns;

        // Default role assignment: last column → objective, all others → feature
        columnRoles      = {};
        objectiveConfigs = {};
        currentColumns.forEach((col, idx) => {
            columnRoles[col] = (idx === currentColumns.length - 1) ? 'objective' : 'feature';
        });

        // Build objective configs for all auto-assigned objective columns
        const objCols = currentColumns.filter(c => columnRoles[c] === 'objective');
        objCols.forEach(col => {
            objectiveConfigs[col] = {
                name:       col,
                type:       'maximize',
                target:     '',
                importance: (100 / objCols.length).toFixed(1),
                included:   true
            };
        });

        // Build feature configs by inspecting the data for each feature column
        boFeatureConfigs = {};
        saFeatureConfigs = {};
        currentColumns.forEach((col, idx) => {
            if (columnRoles[col] === 'feature') {
                const vals  = currentData.map(r => r[idx]).filter(v => v != null && v !== '');
                const isNum = vals.every(v => !isNaN(parseFloat(v)));
                const min   = vals.length ? Math.min(...vals.map(v => parseFloat(v))) : 0;
                const max   = vals.length ? Math.max(...vals.map(v => parseFloat(v))) : 1;
                const range = isNum ? `[${min}, ${max}]` : [...new Set(vals)].join(', ');

                boFeatureConfigs[col] = { type: isNum ? 'continuous' : 'categorical', range, included: true };
                saFeatureConfigs[col] = { type: isNum ? 'continuous' : 'categorical', included: true };
            }
        });

        renderMainTable();
        renderSetup();
        if (currentModule === 'bo') renderTrendPlot();

        document.getElementById('data-section').classList.remove('hidden');
    } catch (err) {
        alert('Error loading file: ' + err.message);
    }
}


// ===========================================================================
// MAIN DATA TABLE — Render + Inline Edit
// ===========================================================================

function renderMainTable() {
    /**
     * Re-renders the main dataset table with:
     *   - Sticky headers containing F/O role toggle buttons.
     *   - contenteditable cells for inline data correction.
     */
    const head = document.getElementById('main-table-head');
    const body = document.getElementById('main-table-body');
    if (!head || !body) return;

    // Build header row with role toggle buttons
    let headHtml = '<tr>';
    currentColumns.forEach(col => {
        const role = columnRoles[col];
        headHtml += `
            <th>
                <div class="col-header-name">${col}</div>
                <div class="type-toggle" onclick="toggleColumnRole('${col}')">
                    <span class="${role === 'feature'   ? 'active' : ''}">F</span>
                    <span class="${role === 'objective' ? 'active' : ''}">O</span>
                </div>
            </th>`;
    });
    headHtml += '</tr>';
    head.innerHTML = headHtml;

    // Build data rows with inline-editable cells
    let bodyHtml = '';
    if (currentData) {
        currentData.forEach((row, rowIdx) => {
            bodyHtml += '<tr>' +
                row.map((v, colIdx) =>
                    `<td contenteditable="true" onblur="updateCellData(${rowIdx}, ${colIdx}, this.textContent)">${v}</td>`
                ).join('') +
                '</tr>';
        });
    }
    body.innerHTML = bodyHtml;
}

window.updateCellData = (rowIdx, colIdx, value) => {
    /** Sync an inline-edited cell value back to currentData and refresh the trend plot. */
    currentData[rowIdx][colIdx] = value;
    if (currentModule === 'bo') renderTrendPlot();
};

window.toggleColumnRole = (colName) => {
    /**
     * Toggles a column between 'feature' and 'objective' roles.
     * Also initializes/cleans up associated config entries and rebalances
     * importance weights across remaining objectives.
     */
    const oldRole = columnRoles[colName];
    const newRole = oldRole === 'feature' ? 'objective' : 'feature';
    columnRoles[colName] = newRole;

    if (newRole === 'feature') {
        // Initialize feature config from current data values
        const idx   = currentColumns.indexOf(colName);
        const vals  = currentData.map(r => r[idx]).filter(v => v != null && v !== '');
        const isNum = vals.every(v => !isNaN(parseFloat(v)));
        const min   = vals.length ? Math.min(...vals.map(v => parseFloat(v))) : 0;
        const max   = vals.length ? Math.max(...vals.map(v => parseFloat(v))) : 1;
        const range = isNum ? `[${min}, ${max}]` : [...new Set(vals)].join(', ');

        if (!boFeatureConfigs[colName]) boFeatureConfigs[colName] = { type: isNum ? 'continuous' : 'categorical', range, included: true };
        if (!saFeatureConfigs[colName]) saFeatureConfigs[colName] = { type: isNum ? 'continuous' : 'categorical', included: true };
    } else {
        // Clean up feature configs when column is promoted to objective
        delete boFeatureConfigs[colName];
        delete saFeatureConfigs[colName];
    }

    // Ensure objectiveConfigs is in sync with the current set of objective columns
    const activeObjs = currentColumns.filter(c => columnRoles[c] === 'objective');
    activeObjs.forEach(col => {
        if (!objectiveConfigs[col]) {
            objectiveConfigs[col] = { name: col, type: 'maximize', target: '', importance: 0, included: true };
        }
    });
    Object.keys(objectiveConfigs).forEach(col => {
        if (!activeObjs.includes(col)) delete objectiveConfigs[col];
    });

    rebalanceObjectiveConfigs();
    renderMainTable();
    renderSetup();
    if (currentModule === 'bo') renderTrendPlot();
};


// ===========================================================================
// IMPORTANCE REBALANCING
// ===========================================================================

function rebalanceRemaining(changedIdOrEl, newValue, selector) {
    /**
     * "Waterfall" rebalancing of importance weights after the user edits one input.
     *
     * The 1000-unit total represents 100.0% in 0.1% steps (avoids floating-point
     * rounding issues).
     *
     * Rules:
     *   - If user edits a row that is NOT the last: keep rows ABOVE fixed,
     *     distribute the remaining budget evenly among rows BELOW.
     *   - If user edits the LAST row: keep it fixed and rebalance ABOVE rows.
     */
    const inputs = Array.from(document.querySelectorAll(selector));
    let activeInputs = inputs;

    // For BO/SA, only include the objectives that are checked as 'included'
    if (selector.includes('bo') || selector.includes('sa')) {
        activeInputs = inputs.filter(el => {
            const col = el.dataset.col;
            return objectiveConfigs[col] && objectiveConfigs[col].included;
        });
    }

    if (activeInputs.length <= 1) {
        if (activeInputs.length === 1) activeInputs[0].value = (100).toFixed(1);
        return;
    }

    const changedIdx = activeInputs.findIndex(el => {
        if (changedIdOrEl instanceof HTMLElement) return el === changedIdOrEl;
        return el.dataset.col === changedIdOrEl;
    });
    if (changedIdx === -1) return;

    const total = 1000;  // Represents 100.0% in tenths

    if (changedIdx < activeInputs.length - 1) {
        // Editing a non-last row: cascade downward
        let sumAbove = 0;
        for (let i = 0; i < changedIdx; i++) {
            sumAbove += Math.round(parseFloat(activeInputs[i].value) * 10) || 0;
        }
        let changedVal = Math.round(parseFloat(newValue) * 10) || 0;
        if (sumAbove + changedVal > total) {
            changedVal = Math.max(0, total - sumAbove);
            activeInputs[changedIdx].value = (changedVal / 10).toFixed(1);
        }
        const remainingForBelow = total - sumAbove - changedVal;
        const belowRows = activeInputs.slice(changedIdx + 1);
        const share     = Math.floor(remainingForBelow / belowRows.length);
        let currentTargetSum = 0;

        belowRows.forEach((el, idx) => {
            let val = (idx === belowRows.length - 1) ? remainingForBelow - currentTargetSum : share;
            if (val < 0) val = 0;
            el.value = (val / 10).toFixed(1);
            currentTargetSum += val;
        });
    } else {
        // Editing the last row: rebalance above
        let changedVal = Math.round(parseFloat(newValue) * 10) || 0;
        if (changedVal > total) { changedVal = total; activeInputs[changedIdx].value = (100).toFixed(1); }

        const remainderForAbove = total - changedVal;
        const aboveRows = activeInputs.slice(0, changedIdx);
        const share     = Math.floor(remainderForAbove / aboveRows.length);
        let currentTargetSum = 0;

        aboveRows.forEach((el, idx) => {
            let val = (idx === aboveRows.length - 1) ? remainderForAbove - currentTargetSum : share;
            if (val < 0) val = 0;
            el.value = (val / 10).toFixed(1);
            currentTargetSum += val;
        });
    }
}

function rebalanceObjectiveConfigs() {
    /**
     * Re-distributes importance weights evenly among objectives that are
     * marked as 'included', zeroing out excluded ones.
     * Used after toggling inclusion or adding/removing an objective.
     */
    const objCols      = Object.keys(objectiveConfigs);
    const includedCols = objCols.filter(c => objectiveConfigs[c].included);

    // Zero out excluded objectives
    objCols.forEach(col => {
        if (!objectiveConfigs[col].included) objectiveConfigs[col].importance = (0).toFixed(1);
    });

    if (includedCols.length === 0) return;

    const total = 1000;
    const share = Math.floor(total / includedCols.length);
    let currentSum = 0;

    includedCols.forEach((col, idx) => {
        if (idx === includedCols.length - 1) {
            objectiveConfigs[col].importance = ((total - currentSum) / 10).toFixed(1);
        } else {
            objectiveConfigs[col].importance = (share / 10).toFixed(1);
            currentSum += share;
        }
    });
}


// ===========================================================================
// SETUP PANEL RENDERING — Delegates to the active module
// ===========================================================================

function renderSetup() {
    if      (currentModule === 'doe') renderDoESetup();
    else if (currentModule === 'bo')  renderBOSetup();
    else if (currentModule === 'sa')  renderSASetup();
}

function renderDoESetup() {
    /**
     * DoE setup panel is driven entirely by DOM (user manually adds rows).
     * No programmatic rendering needed; this function is intentionally empty.
     */
}

function rebalanceDoEDOM() {
    /** Evenly redistribute importance weights among all DoE objective rows. */
    const inputs = Array.from(document.querySelectorAll('.doe-o-importance'));
    if (inputs.length === 0) return;
    const share = (100 / inputs.length).toFixed(1);
    inputs.forEach(input => input.value = share);
}

window.updateObjectiveName = (oldName, newName) => {
    /** Rename an objective in objectiveConfigs when the user edits its name field. */
    if (oldName === newName) return;
    if (objectiveConfigs[oldName]) {
        objectiveConfigs[newName]      = objectiveConfigs[oldName];
        objectiveConfigs[newName].name = newName;
        delete objectiveConfigs[oldName];
        renderSetup();
    }
};


// ===========================================================================
// BO SETUP RENDERING
// ===========================================================================

function renderBOSetup() {
    /**
     * Renders the Bayesian Optimization configuration tables:
     *   - Features table: type selector, range input, include checkbox.
     *   - Objectives table: type selector, target input, importance spinner, include checkbox.
     * Both tables read from global config objects and write back via sync* functions.
     */
    const featuresList  = document.getElementById('bo-features-list');
    const objectivesList = document.getElementById('bo-objectives-list');

    // Features
    let featuresHtml = '';
    currentColumns.filter(col => columnRoles[col] === 'feature').forEach(col => {
        const cfg = boFeatureConfigs[col] || { type: 'continuous', range: '', included: true };
        featuresHtml += `<tr>
            <td>${col}</td>
            <td><select class="bo-f-type" data-col="${col}" onchange="syncBOFeatureConfig('${col}', 'type')">
                <option value="continuous" ${cfg.type === 'continuous' ? 'selected' : ''}>continuous</option>
                <option value="regular"    ${cfg.type === 'regular'    ? 'selected' : ''}>regular</option>
                <option value="discrete"   ${cfg.type === 'discrete'   ? 'selected' : ''}>discrete</option>
                <option value="categorical"${cfg.type === 'categorical'? 'selected' : ''}>categorical</option>
            </select></td>
            <td><input type="text" class="bo-f-range" data-col="${col}" value="${cfg.range}" placeholder="[min, max] or [min, max]steps" onchange="syncBOFeatureConfig('${col}', 'range')"></td>
            <td><input type="checkbox" class="bo-f-include" data-col="${col}" ${cfg.included ? 'checked' : ''} onchange="syncBOFeatureConfig('${col}', 'include')"></td>
        </tr>`;
    });
    featuresList.innerHTML = featuresHtml;

    // Objectives
    let objectivesHtml = '';
    const selectedObjs  = currentColumns.filter(col => columnRoles[col] === 'objective');
    selectedObjs.forEach(col => {
        const cfg       = objectiveConfigs[col] || { type: 'maximize', target: '', importance: (100/selectedObjs.length).toFixed(1), included: true };
        const isDisabled = !cfg.included;
        objectivesHtml += `<tr class="${isDisabled ? 'row-disabled' : ''}">
            <td>${col}</td>
            <td><select class="bo-o-type" data-col="${col}" onchange="syncObjectiveConfig('${col}', 'bo'); toggleRowTarget(this);" ${isDisabled ? 'disabled' : ''}>
                <option value="maximize" ${cfg.type === 'maximize' ? 'selected' : ''}>maximize</option>
                <option value="minimize" ${cfg.type === 'minimize' ? 'selected' : ''}>minimize</option>
                <option value="target"   ${cfg.type === 'target'   ? 'selected' : ''}>target</option>
            </select></td>
            <td><input type="text" class="bo-o-target ${cfg.type === 'target' ? '' : 'hidden'}" data-col="${col}" value="${cfg.target}" placeholder="Target value" onchange="syncObjectiveConfig('${col}', 'bo', 'target')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="number" class="bo-o-importance" data-col="${col}" value="${cfg.importance}" onchange="syncObjectiveConfig('${col}', 'bo', 'importance')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="checkbox" class="bo-o-include" data-col="${col}" ${cfg.included ? 'checked' : ''} onchange="syncObjectiveConfig('${col}', 'bo', 'include')"></td>
        </tr>`;
    });
    objectivesList.innerHTML = objectivesHtml;
    updateAcqSelectForMultiObj();   // Disable/annotate acquisition type selector for MOO
    updateAllConstraintsDropdowns('bo');
}

window.syncBOFeatureConfig = (col, field) => {
    /**
     * Reads the current DOM state for a BO feature row and writes it back
     * to the global boFeatureConfigs store.
     */
    const tr = document.querySelector(`.bo-f-type[data-col="${col}"]`)?.closest('tr');
    if (!tr || !boFeatureConfigs[col]) return;

    if (field === 'type'    || !field) boFeatureConfigs[col].type     = tr.querySelector('.bo-f-type').value;
    if (field === 'range'   || !field) boFeatureConfigs[col].range    = tr.querySelector('.bo-f-range').value;
    if (field === 'include' || !field) boFeatureConfigs[col].included = tr.querySelector('.bo-f-include').checked;
    updateAllConstraintsDropdowns('bo');
};

function updateAcqSelectForMultiObj() {
    /**
     * When multiple objectives are included, disable the acquisition-function
     * selector (BO always uses NEHVI or Scalarized EI for MOO) and show a
     * contextual note explaining the current multi-objective mode.
     *
     * Discovery Mode: equal weights → explore full Pareto front (NEHVI).
     * Priority Mode:  unequal weights → focus on weighted-sum objective (Scal. EI).
     */
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

window.syncObjectiveConfig = (col, sourceModule, field) => {
    /**
     * Two-phase sync when an objective setting changes:
     *   1. Read ALL objective rows back to objectiveConfigs (ensures consistency).
     *   2. Perform importance rebalancing via rebalanceRemaining or rebalanceObjectiveConfigs.
     * Then re-render the setup panel and refresh plots.
     */
    if (sourceModule === 'doe') return;  // DoE objectives are DOM-only, not in objectiveConfigs

    const rolePrefix = sourceModule === 'bo' ? 'bo-o' : 'sa-o';
    const row        = document.querySelector(`.${rolePrefix}-type[data-col="${col}"]`).closest('tr');

    // Phase 1: Sync entire DOM → objectiveConfigs
    document.querySelectorAll(`.${rolePrefix}-include`).forEach(cb => {
        const c = cb.dataset.col;
        const r = cb.closest('tr');
        if (objectiveConfigs[c]) {
            objectiveConfigs[c].type      = r.querySelector(`.${rolePrefix}-type`).value;
            objectiveConfigs[c].target    = r.querySelector(`.${rolePrefix}-target`).value;
            if (field === 'importance') objectiveConfigs[c].importance = r.querySelector(`.${rolePrefix}-importance`).value;
            objectiveConfigs[c].included  = cb.checked;
        }
    });

    // Phase 2: Rebalance based on what changed
    if (field === 'importance') {
        const selector = `.${rolePrefix}-importance`;
        rebalanceRemaining(col, row.querySelector(selector).value, selector);
        // Sync back the rebalanced DOM values to global configs
        document.querySelectorAll(selector).forEach(el => {
            if (objectiveConfigs[el.dataset.col]) objectiveConfigs[el.dataset.col].importance = el.value;
        });
    } else if (field === 'include' || field === 'type') {
        rebalanceObjectiveConfigs();
    }

    renderSetup();
    updateAcqSelectForMultiObj();
    if (currentModule === 'bo') renderTrendPlot();
};


// ===========================================================================
// SA SETUP RENDERING
// ===========================================================================

function renderSASetup() {
    /**
     * Renders the Statistical Analysis configuration tables:
     *   - Features: type (Numerical/Discrete/Categorical) and include toggle.
     *   - Objectives: same as BO objectives table (shared objectiveConfigs).
     */
    const list    = document.getElementById('sa-features-list');
    const objList = document.getElementById('sa-objectives-list');

    let featHtml = '';
    currentColumns.filter(c => columnRoles[c] === 'feature').forEach(col => {
        const cfg = saFeatureConfigs[col] || { type: 'continuous', included: true };
        featHtml += `<tr>
            <td>${col}</td>
            <td><select class="sa-f-type" data-col="${col}" onchange="syncSAFeatureConfig('${col}', 'type')">
                <option value="continuous" ${cfg.type === 'continuous' ? 'selected' : ''}>Numerical</option>
                <option value="discrete"   ${cfg.type === 'discrete'   ? 'selected' : ''}>Discrete</option>
                <option value="categorical"${cfg.type === 'categorical'? 'selected' : ''}>Categorical</option>
            </select></td>
            <td><input type="checkbox" class="sa-f-include" data-col="${col}" ${cfg.included ? 'checked' : ''} onchange="syncSAFeatureConfig('${col}', 'include')"> Include</td>
        </tr>`;
    });
    list.innerHTML = featHtml;

    let objHtml    = '';
    const selectedObjs = currentColumns.filter(col => columnRoles[col] === 'objective');
    selectedObjs.forEach(col => {
        const cfg       = objectiveConfigs[col] || { type: 'maximize', target: '', importance: (100/selectedObjs.length).toFixed(1), included: true };
        const isDisabled = !cfg.included;
        objHtml += `<tr class="${isDisabled ? 'row-disabled' : ''}">
            <td>${col}</td>
            <td><select class="sa-o-type" data-col="${col}" onchange="syncObjectiveConfig('${col}', 'sa', 'type'); toggleRowTarget(this)" ${isDisabled ? 'disabled' : ''}>
                <option value="maximize" ${cfg.type === 'maximize' ? 'selected' : ''}>maximize</option>
                <option value="minimize" ${cfg.type === 'minimize' ? 'selected' : ''}>minimize</option>
                <option value="target"   ${cfg.type === 'target'   ? 'selected' : ''}>target</option>
            </select></td>
            <td><input type="text" class="sa-o-target ${cfg.type === 'target' ? '' : 'hidden'}" data-col="${col}" value="${cfg.target || ''}" placeholder="Target" onchange="syncObjectiveConfig('${col}', 'sa', 'target')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="number" class="sa-o-importance" data-col="${col}" value="${cfg.importance}" onchange="syncObjectiveConfig('${col}', 'sa', 'importance')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="checkbox" class="sa-o-include" data-col="${col}" ${cfg.included ? 'checked' : ''} onchange="syncObjectiveConfig('${col}', 'sa', 'include')"> Include</td>
        </tr>`;
    });
    objList.innerHTML = objHtml;
}

window.syncSAFeatureConfig = (col, field) => {
    /** Read SA feature row DOM → saFeatureConfigs global store. */
    const tr = document.querySelector(`.sa-f-type[data-col="${col}"]`)?.closest('tr');
    if (!tr || !saFeatureConfigs[col]) return;

    if (field === 'type'    || !field) saFeatureConfigs[col].type     = tr.querySelector('.sa-f-type').value;
    if (field === 'include' || !field) saFeatureConfigs[col].included = tr.querySelector('.sa-f-include').checked;
};


// ===========================================================================
// UTILITY HELPERS
// ===========================================================================

function safeAddListener(id, event, callback) {
    /**
     * Attaches an event listener only if the element with the given ID exists.
     * Prevents silent failures during setup when optional elements are absent.
     */
    const el = document.getElementById(id);
    if (el) el.addEventListener(event, callback);
    else console.warn(`Element with ID "${id}" not found. skipping listener.`);
}


// ===========================================================================
// BUTTON WIRING — initButtons()
// ===========================================================================

function initButtons() {
    /**
     * Wires all action buttons that are present in index.html.
     * Organized into logical groups:
     *   1. DoE: add feature row, add objective row, run DoE.
     *   2. BO:  run optimization, commit results, export.
     *   3. SA:  run analysis, estimate, export report.
     *   4. Misc: shutdown, clear, CSV export.
     *   5. Column highlight on hover for export guidance.
     *   6. Results import from CSV.
     */

    // --- DoE: Dynamically add a feature row to the table ---
    safeAddListener('doe-add-feature-btn', 'click', () => {
        const list = document.getElementById('doe-features-list');
        const tr   = document.createElement('tr');
        tr.innerHTML = `
            <td><input type="text" class="doe-f-name" value="Feature_${list.children.length + 1}"></td>
            <td><select class="doe-f-type" onchange="updatePlaceholder(this)">
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

    // --- DoE: Dynamically add an objective row to the table ---
    safeAddListener('doe-add-objective-btn', 'click', () => {
        const list  = document.getElementById('doe-objectives-list');
        const count = list.children.length + 1;
        const tr    = document.createElement('tr');
        tr.innerHTML = `
            <td><input type="text" class="doe-o-name" value="Objective_${count}"></td>
            <td><select class="doe-o-type" onchange="toggleRowTarget(this)">
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

    // --- Primary action buttons ---
    safeAddListener('doe-run-btn',      'click', runDoE);
    safeAddListener('run-btn',          'click', runBO);
    safeAddListener('sa-run-btn',       'click', runSA);
    safeAddListener('sa-estimate-btn',  'click', runEstimate);
    safeAddListener('bo-estimate-btn',  'click', runBOEstimate);

    // --- Constraints Setup listeners ---
    safeAddListener('doe-add-num-constraint-btn', 'click', () => addNumericalConstraintRow('doe'));
    safeAddListener('doe-add-disallowed-constraint-btn', 'click', () => addDisallowedCombinationRow('doe'));
    safeAddListener('bo-add-num-constraint-btn', 'click', () => addNumericalConstraintRow('bo'));
    safeAddListener('bo-add-disallowed-constraint-btn', 'click', () => addDisallowedCombinationRow('bo'));

    const doeFeatList = document.getElementById('doe-features-list');
    if (doeFeatList) {
        doeFeatList.addEventListener('input', () => updateAllConstraintsDropdowns('doe'));
        doeFeatList.addEventListener('change', () => updateAllConstraintsDropdowns('doe'));
    }

    // --- Shutdown button: confirm → POST /shutdown → show stopped message ---
    safeAddListener('shutdown-btn', 'click', async () => {
        if (confirm('Are you sure you want to shut down the EDOS server?')) {
            try {
                await fetch('/shutdown', { method: 'POST' });
            } catch (_) { /* expected — server is stopping, response may not arrive */ }
            document.body.innerHTML = `
                <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;background:#0f172a;color:#e2e8f0;font-family:sans-serif;gap:20px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18.4 6.1a9 9 0 1 1-12.8 0"/><line x1="12" y1="2" x2="12" y2="12"/></svg>
                    <h1 style="font-size:2rem;margin:0;">Server Stopped</h1>
                    <p style="color:#64748b;max-width:400px;text-align:center;">The EDOS server has been shut down. You can safely close this browser tab.</p>
                </div>`;
        }
    });

    // --- CSV export for the main data table ---
    safeAddListener('export-main-btn', 'click', () => exportData(currentData, currentColumns));

    // --- Commit BO suggestions to the main dataset ---
    safeAddListener('commit-suggestions-btn', 'click', (e) => {
        if (e.target.disabled) return;
        const rows    = document.querySelectorAll('#results-table-body tr');
        const featCols = currentColumns.filter(c => columnRoles[c] === 'feature');
        const objCols  = currentColumns.filter(c => columnRoles[c] === 'objective');

        let addedCount = 0;
        rows.forEach(tr => {
            const cells  = tr.querySelectorAll('td');
            const newRow = new Array(currentColumns.length).fill('');

            // Copy feature values from the suggestion cells
            featCols.forEach((col, i) => {
                newRow[currentColumns.indexOf(col)] = cells[i].textContent.trim();
            });
            // Copy objective values; 'N/D' means not determined (excluded from BO run)
            objCols.forEach((col, i) => {
                const val = cells[featCols.length + i].textContent.trim();
                newRow[currentColumns.indexOf(col)] = (val === 'N/D') ? '' : val;
            });

            currentData.push(newRow);
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

    // --- Commit DoE suggestions to the main dataset and switch to BO module ---
    safeAddListener('doe-commit-suggestions-btn', 'click', () => {
        if (!suggestionsData || suggestionsData.length === 0) return;
        const rows = document.querySelectorAll('#doe-results-body tr');
        if (rows.length === 0) return;

        const feats = Object.keys(suggestionsData[0]);
        const objs  = currentDoEObjectives.map(o => o.name);

        // Initialize the dataset if it doesn't exist yet
        if (!currentData || currentColumns.length === 0) {
            currentColumns = [...feats, ...objs];
            currentData    = [];
            feats.forEach(col => { columnRoles[col] = 'feature'; });
            objs.forEach(col  => { columnRoles[col] = 'objective'; });
        }

        // Propagate DoE feature/objective configs to BO/SA stores
        currentDoEFeatures.forEach(f => {
            if (currentColumns.includes(f.name)) {
                boFeatureConfigs[f.name] = { type: f.type, range: f.range, included: true };
                saFeatureConfigs[f.name] = { type: f.type, included: true };
            }
        });
        currentDoEObjectives.forEach(o => {
            if (currentColumns.includes(o.name) && !objectiveConfigs[o.name]) {
                objectiveConfigs[o.name] = { name: o.name, type: 'maximize', target: '', importance: (100/objs.length).toFixed(1), included: true };
            }
        });

        // Validate that all column names exist in the current dataset
        const colIndices = {};
        let missing = [];
        [...feats, ...objs].forEach(name => {
            const idx = currentColumns.indexOf(name);
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
            const newRow = new Array(currentColumns.length).fill('');

            feats.forEach((name, i) => { newRow[colIndices[name]] = cells[i].textContent.trim(); });
            objs.forEach((name,  i) => {
                const val = cells[feats.length + i].textContent.trim();
                newRow[colIndices[name]] = (val === 'N/D') ? '' : val;
            });

            currentData.push(newRow);
            addedCount++;
        });

        if (addedCount > 0) {
            renderMainTable();
            renderSetup();
            alert(`Added ${addedCount} experiments to the main dataset.`);
            document.getElementById('doe-results-section').classList.add('hidden');
            document.getElementById('data-section').classList.remove('hidden');
            switchModule('bo');   // Automatically navigate to BO for the next step
        }
    });

    // --- DoE export: full table (features + empty objective columns) ---
    safeAddListener('doe-export-btn', 'click', () => {
        if (!suggestionsData || suggestionsData.length === 0) return;
        const rows  = document.querySelectorAll('#doe-results-body tr');
        if (rows.length === 0) return;
        const feats = Object.keys(suggestionsData[0]);
        const cols  = [...feats, ...currentDoEObjectives.map(o => o.name)];
        const exportArr = Array.from(rows).map(tr => Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim()));
        exportData(exportArr, cols);
    });

    // --- BO export: full results table ---
    safeAddListener('export-results-btn', 'click', () => {
        if (!suggestionsData || suggestionsData.length === 0) return;
        const rows     = document.querySelectorAll('#results-table-body tr');
        if (rows.length === 0) return;
        const featCols = currentColumns.filter(c => columnRoles[c] === 'feature');
        const objCols  = currentColumns.filter(c => columnRoles[c] === 'objective');
        const cols     = [...featCols, ...objCols.map(c => `${c} (Result)`)];
        const exportArr = Array.from(rows).map(tr => Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim()));
        exportData(exportArr, cols);
    });

    // --- Pareto export: only the non-dominated rows ---
    safeAddListener('pareto-export-btn', 'click', () => {
        const objCols      = currentColumns.filter(c => columnRoles[c] === 'objective');
        const paretoIndices = detectParetoIndices(currentData, objCols);
        const paretoData   = currentData.filter((_, i) => paretoIndices.includes(i));
        exportData(paretoData, currentColumns);
    });

    // --- Column highlight on hover (visual guide for selective export buttons) ---
    const applyHighlight = (tableId, type) => {
        const table = document.getElementById(tableId);
        if (!table) return;
        const ths = table.querySelectorAll('thead th');
        ths.forEach((th, idx) => {
            let matches = false;
            if (type === 'feature') {
                if (tableId === 'doe-results-table') {
                    const feats = suggestionsData && suggestionsData.length > 0 ? Object.keys(suggestionsData[0]) : [];
                    matches = idx < feats.length;
                } else {
                    matches = !th.textContent.includes('(Result)');
                }
            } else if (type === 'objective') {
                if (tableId === 'doe-results-table') {
                    const feats = suggestionsData && suggestionsData.length > 0 ? Object.keys(suggestionsData[0]) : [];
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

    // Hover listeners for column highlighting
    safeAddListener('doe-export-features-btn', 'mouseenter', () => applyHighlight('doe-results-table', 'feature'));
    safeAddListener('doe-export-features-btn', 'mouseleave', () => removeHighlight('doe-results-table'));
    safeAddListener('export-features-btn',     'mouseenter', () => applyHighlight('results-table', 'feature'));
    safeAddListener('export-features-btn',     'mouseleave', () => removeHighlight('results-table'));
    safeAddListener('doe-import-results-btn',  'mouseenter', () => applyHighlight('doe-results-table', 'objective'));
    safeAddListener('doe-import-results-btn',  'mouseleave', () => removeHighlight('doe-results-table'));
    safeAddListener('import-results-btn',      'mouseenter', () => applyHighlight('results-table', 'objective'));
    safeAddListener('import-results-btn',      'mouseleave', () => removeHighlight('results-table'));

    // --- Selective Feature Export (features columns only) ---
    safeAddListener('doe-export-features-btn', 'click', () => {
        if (!suggestionsData || suggestionsData.length === 0) return;
        const rows  = document.querySelectorAll('#doe-results-body tr');
        if (rows.length === 0) return;
        const feats = Object.keys(suggestionsData[0]);
        const exportArr = Array.from(rows).map(tr =>
            Array.from(tr.querySelectorAll('td')).slice(0, feats.length).map(td => td.textContent.trim())
        );
        exportData(exportArr, feats);
    });

    safeAddListener('export-features-btn', 'click', () => {
        if (!suggestionsData || suggestionsData.length === 0) return;
        const rows     = document.querySelectorAll('#results-table-body tr');
        if (rows.length === 0) return;
        const featCols = currentColumns.filter(c => columnRoles[c] === 'feature');
        const exportArr = Array.from(rows).map(tr =>
            Array.from(tr.querySelectorAll('td')).slice(0, featCols.length).map(td => td.textContent.trim())
        );
        exportData(exportArr, featCols);
    });

    // --- Import Experimental Results into the suggestion tables ---
    // Triggers a hidden file input, then parses the CSV and fills objective cells
    safeAddListener('doe-import-results-btn', 'click', () => document.getElementById('doe-import-results-input').click());
    safeAddListener('doe-import-results-input', 'change', (e) =>
        handleResultsImport(e, 'doe-results-table', currentDoEObjectives.map(o => o.name), false)
    );

    safeAddListener('import-results-btn',   'click', () => document.getElementById('import-results-input').click());
    safeAddListener('import-results-input', 'change', (e) =>
        handleResultsImport(e, 'results-table', currentColumns.filter(c => columnRoles[c] === 'objective'), true)
    );
}


// ===========================================================================
// CSV RESULTS IMPORT
// ===========================================================================

function handleResultsImport(e, tableId, expectedObjCols, autoCommit) {
    /**
     * Reads a CSV file and fills the objective cells in the active suggestion table.
     *
     * Header matching:
     *   1. By name: headers that match expectedObjCols (with or without " (Result)" suffix).
     *   2. Sequential fallback: if no names match but column count is ≥ expected, map by position.
     *
     * @param {Event}    e               - Change event from the hidden file input.
     * @param {string}   tableId         - ID of the HTML table to fill.
     * @param {string[]} expectedObjCols - Names of objective columns to populate.
     * @param {boolean}  autoCommit      - If true, click the commit button after import.
     */
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = '';  // Reset so the same file can be re-imported

    const reader = new FileReader();
    reader.onload = (event) => {
        const text    = event.target.result;
        const lines   = text.trim().split('\n');
        if (lines.length < 2) {
            alert('CSV should contain at least a header row and one data row.');
            return;
        }

        // Parse header and data rows (strip surrounding quotes)
        const headers  = lines[0].split(',').map(h => h.trim().replace(/^['"]|['"]$/g, ''));
        const dataRows = lines.slice(1).map(l => l.split(',').map(c => c.trim().replace(/^['"]|['"]$/g, '')));

        // Build column index map from CSV headers to expectedObjCols
        let colMap        = [];
        let useSequential = false;

        expectedObjCols.forEach(col => {
            const rawCol = col.replace(' (Result)', '');
            colMap.push(headers.findIndex(h => h === rawCol || h === col));
        });

        // Fallback to positional mapping if no headers matched
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
                const offset  = tds.length - expectedObjCols.length;  // Feature columns come first

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
                // Give the DOM a tick to process the input events before committing
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


// ===========================================================================
// SETTINGS LISTENERS — SA Model Visibility, BO Slider, Importance Norm.
// ===========================================================================

function initSettingsListeners() {
    /**
     * Wires up ancillary UI interactions:
     *   - SA model parameter visibility: show RF-specific or MLP-specific params.
     *   - BO exploration slider ↔ number field synchronization.
     *   - Objective importance auto-normalization on change events.
     */

    // SA: Show/hide model-specific parameters
    const saModel = document.getElementById('sa-model-select');
    const updateSAParamsVisibility = () => {
        const val = saModel.value;
        document.querySelectorAll('.sa-rf-param').forEach(el  => el.classList.toggle('hidden', val !== 'random_forest'));
        document.querySelectorAll('.sa-mlp-param').forEach(el => el.classList.toggle('hidden', val !== 'mlp'));
    };
    saModel.addEventListener('change', updateSAParamsVisibility);
    updateSAParamsVisibility();  // Apply immediately on page load

    safeAddListener('sa-export-report-btn', 'click', exportSAReport);

    // BO Exploration slider ↔ number input bidirectional sync
    const styleSlider = document.getElementById('style-slider');
    const styleNumber = document.getElementById('style-number');
    if (styleSlider && styleNumber) {
        styleSlider.addEventListener('input', (e) => styleNumber.value = e.target.value);
        styleNumber.addEventListener('input', (e) => styleSlider.value = e.target.value);
    }

    // Objective importance normalization: attach a delegated listener to document
    // that intercepts 'change' events from any importance input matching the selector
    const attachNorm = (selector) => {
        document.addEventListener('change', (e) => {
            if (e.target.matches(selector)) {
                let inputs = Array.from(document.querySelectorAll(selector));
                const isDoe = selector.includes('doe');
                // For BO/SA, normalize only among INCLUDED objectives
                if (!isDoe) {
                    inputs = inputs.filter(el => {
                        const col = el.dataset.col;
                        return objectiveConfigs[col] && objectiveConfigs[col].included;
                    });
                }
                rebalanceRemaining(e.target, e.target.value, selector);

                // Sync the rebalanced DOM values back to global state
                if (!isDoe) {
                    Array.from(document.querySelectorAll(selector)).forEach(el => {
                        const col = el.dataset.col;
                        if (col && objectiveConfigs[col]) objectiveConfigs[col].importance = el.value;
                    });
                }

                if (!isDoe && currentModule === 'bo') renderTrendPlot();
            }
        });
    };
    attachNorm('.doe-o-importance');
    attachNorm('.bo-o-importance');
    attachNorm('.sa-o-importance');
}


// ===========================================================================
// DELETE ROW HANDLER
// ===========================================================================

function attachDeleteEvents() {
    /**
     * Attaches onclick handlers to all '×' delete buttons in the DoE tables.
     * Cleans up objectiveConfigs if an objective row is deleted, then
     * rebalances importance weights.
     */
    document.querySelectorAll('.delete-row').forEach(btn => {
        btn.onclick = () => {
            const tr         = btn.closest('tr');
            const nameInput  = tr.querySelector('.doe-o-name, .doe-f-name');
            const isDoeRow   = tr.querySelector('.doe-o-name, .doe-f-name, .doe-o-type, .doe-f-type');

            if (nameInput && !isDoeRow) {
                const name = nameInput.value;
                if (objectiveConfigs[name]) {
                    delete objectiveConfigs[name];
                    rebalanceObjectiveConfigs();
                }
            }
            tr.remove();

            if (currentModule === 'doe') {
                rebalanceDoEDOM();
                updateAllConstraintsDropdowns('doe');
            } else {
                renderSetup();
                updateAllConstraintsDropdowns('bo');
            }
        };
    });
}


// ===========================================================================
// INLINE TARGET FIELD TOGGLE
// ===========================================================================

window.toggleRowTarget = (select) => {
    /**
     * Shows or hides the target value input in an objective row based on whether
     * the "target" option is selected.
     */
    const tr          = select.closest('tr');
    const targetInput = tr.querySelector('.bo-o-target, .doe-o-target, .sa-o-target');
    if (select.value === 'target') {
        targetInput.classList.remove('hidden');
    } else {
        targetInput.classList.add('hidden');
    }
};

window.updatePlaceholder = (select) => {
    /**
     * Updates the range input placeholder text when the feature type changes
     * in the DoE setup panel.
     */
    const tr    = select.closest('tr');
    const input = tr.querySelector('.doe-f-range');
    const type  = select.value;
    if      (type === 'continuous') input.placeholder = '[min, max]';
    else if (type === 'regular')    input.placeholder = '[min, max]steps';
    else if (type === 'discrete')   input.placeholder = 'xx, yy, zz,...';
    else if (type === 'categorical') input.placeholder = 'A, B, C,...';
};


// ===========================================================================
// DOE RUN — Collect config, POST to /doe, Render results
// ===========================================================================

async function runDoE() {
    /**
     * Collects the current DoE feature and objective configuration from the DOM,
     * sends it to /doe, and renders the resulting suggestion table with
     * quality metrics.
     */
    currentDoEFeatures = Array.from(document.querySelectorAll('#doe-features-list tr')).map(tr => ({
        name:  tr.querySelector('.doe-f-name').value,
        type:  tr.querySelector('.doe-f-type').value,
        range: tr.querySelector('.doe-f-range').value
    }));

    currentDoEObjectives = Array.from(document.querySelectorAll('#doe-objectives-list tr')).map(tr => ({
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

    currentAbortController = new AbortController();
    try {
        const res    = await fetch('/doe', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ features: currentDoEFeatures, tweaks }),
            signal: currentAbortController.signal
        });
        const result = await res.json();
        document.getElementById('loading-overlay').classList.add('hidden');
        if (result.error) return alert(result.error);

        suggestionsData = result.suggestions;
        renderDoEResults(result.metrics);
    } catch (err) {
        document.getElementById('loading-overlay').classList.add('hidden');
        if (err.name === 'AbortError') return console.log('DoE calculation aborted by user.');
        alert("DoE Error: " + err.message);
    } finally {
        currentAbortController = null;
    }
}


// ===========================================================================
// DOE RESULTS RENDERING
// ===========================================================================

function renderDoEResults(metrics) {
    /**
     * Renders the DoE results section with:
     *   - Quality metric dashboard (4 cards: Orthogonality, D-Efficiency, Resolution, Curvature).
     *   - Suggestion table with editable objective cells.
     *   - Real-time validation of the Commit button (all objective cells must be filled).
     */
    const section   = document.getElementById('doe-results-section');
    const dashboard = document.getElementById('doe-quality-dashboard');
    const head      = document.getElementById('doe-results-head');
    const body      = document.getElementById('doe-results-body');

    // Render quality metric cards with colored flag badges
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

    const feats = Object.keys(suggestionsData[0]);

    // Dynamic objective column headers
    const objHeaders = currentDoEObjectives.length > 0
        ? currentDoEObjectives.map(o => `<th>${o.name}</th>`).join('')
        : '<th>Target(s)</th>';

    head.innerHTML = '<tr>' + feats.map(f => `<th>${f}</th>`).join('') + objHeaders + '</tr>';

    // Build editable rows: feature cells (pre-filled) + objective cells (empty, green)
    body.innerHTML = suggestionsData.map(row => {
        const featCells = feats.map(f =>
            `<td contenteditable="true" class="feat-input-cell editable-cell" style="background:#f8fafc; border: 1px dashed #94a3b8;">${row[f]}</td>`
        ).join('');
        const objCells = currentDoEObjectives.length > 0
            ? currentDoEObjectives.map(() =>
                `<td contenteditable="true" class="obj-input-cell editable-cell" style="background:#f0fdf4; border: 1px dashed #22c55e;"></td>`
              ).join('')
            : `<td contenteditable="true" class="obj-input-cell editable-cell" style="background:#f0fdf4; border: 1px dashed #22c55e;"></td>`;

        return '<tr>' + featCells + objCells + '</tr>';
    }).join('');

    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth' });

    // Enable the Commit button only when ALL objective cells contain valid numbers
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


// ===========================================================================
// BO RUN — POST to /optimize, Render results
// ===========================================================================

async function runBO() {
    /**
     * Gathers the current BO configuration (features, objectives, tweaks),
     * sends them to /optimize, then renders:
     *   - GP performance table (cross-validated R²)
     *   - Editable suggestion table with commit/export controls
     *   - Estimator panel
     *   - Trend + Pareto plots
     */

    // Clear previous performance and estimator results to avoid stale display
    const perfInfo = document.getElementById('bo-performance-info');
    if (perfInfo) perfInfo.innerHTML = '';
    const estRes = document.getElementById('bo-estimate-results');
    if (estRes)   estRes.innerHTML = '';
    const boResContainer = document.getElementById('bo-results-container');
    if (boResContainer) boResContainer.classList.add('hidden');

    // Collect only the features that are checked 'included'
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

    currentAbortController = new AbortController();
    try {
        const res    = await fetch('/optimize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ data: currentData, columns: currentColumns, features, objectives, tweaks }),
            signal: currentAbortController.signal
        });
        const result = await res.json();
        document.getElementById('loading-overlay').classList.add('hidden');
        if (result.error) return alert("Optimization Error: " + result.error);

        suggestionsData = result.suggestions;

        // Render GP cross-validation performance table
        if (result.performance) renderBOPerformanceTable(result.performance);

        // Render the estimator input panel
        renderBOEstimator();

        const boResContainer = document.getElementById('bo-results-container');
        if (boResContainer) boResContainer.classList.remove('hidden');
    } catch (err) {
        document.getElementById('loading-overlay').classList.add('hidden');
        if (err.name === 'AbortError') return console.log('BO calculation aborted by user.');
        return alert("A critical error occurred during optimization: " + err.message);
    } finally {
        currentAbortController = null;
    }

    // Update objectiveConfigs with the latest run's settings (for Pareto detection)
    if (!objectiveConfigs) objectiveConfigs = {};
    objectives.forEach(obj => { objectiveConfigs[obj.name] = { ...obj, included: true }; });

    // Build suggestion table
    const section    = document.getElementById('results-section');
    const head       = document.getElementById('results-table-head');
    const body       = document.getElementById('results-table-body');
    const commitBtn  = document.getElementById('commit-suggestions-btn');

    const featCols    = currentColumns.filter(c => columnRoles[c] === 'feature');
    const objCols     = currentColumns.filter(c => columnRoles[c] === 'objective');
    const allObjCols  = currentColumns.filter(c => columnRoles[c] === 'objective');
    const optimizedCols = objectives.map(o => o.name);

    head.innerHTML = '<tr>' +
        featCols.map(c => `<th>${c}</th>`).join('') +
        objCols.map(c => `<th>${c} (Result)</th>`).join('') +
        '</tr>';

    // Validate commit readiness: all feature cells must be non-empty, all optimized objective cells must be numeric
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

    body.innerHTML = suggestionsData.map(row => {
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


// ===========================================================================
// SA RUN — POST to /sa, Render results
// ===========================================================================

async function runSA() {
    /**
     * Collects the SA feature/objective configuration, sends to /sa, then
     * passes the result to renderSAResults() for comprehensive visualization.
     */
    const features = Array.from(document.querySelectorAll('.sa-f-include:checked')).map(cb => {
        const row = cb.closest('tr');
        return {
            name:  cb.dataset.col,
            type:  row.querySelector('.sa-f-type').value,
            range: 'valid'   // SA doesn't require numeric range; flag for backend validation
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

    // Cache selected feature/objective names for use in the estimator panel
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
    document.getElementById('sa-estimate-results').innerHTML = '';   // Clear stale estimate

    currentAbortController = new AbortController();
    try {
        const res    = await fetch('/sa', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ data: currentData, columns: currentColumns, features, objectives, tweaks }),
            signal: currentAbortController.signal
        });
        const result = await res.json();
        document.getElementById('loading-overlay').classList.add('hidden');
        if (result.error) return alert("SA Error: " + result.error);

        window.lastSAResult = result;   // Cached for the report export function
        renderSAResults(result);
    } catch(err) {
        document.getElementById('loading-overlay').classList.add('hidden');
        if (err.name === 'AbortError') return console.log('SA calculation aborted by user.');
        alert("A critical error occurred during SA: " + err.message);
        console.error(err);
    } finally {
        currentAbortController = null;
    }
}


// ===========================================================================
// SA RESULTS RENDERING
// ===========================================================================

function renderSAResults(result) {
    /**
     * Renders all SA visualizations into their respective container divs:
     *   1. Numerical Correlation Matrix (Plotly heatmap).
     *   2. Categorical Interaction heatmaps (dynamically selected pair + objective).
     *   3. Parallel Coordinates plots (one per objective, colored by objective value).
     *   4. Feature Impact bar charts + SHAP beeswarm plots (side by side).
     *   5. Reliability table with R², status badge, and advice.
     *   6. Estimator input panel (numerical inputs or categorical dropdowns).
     */
    document.getElementById('sa-results-container').classList.remove('hidden');

    // --- 1. Numerical Correlation Matrix ---
    const numCorr  = result.numerical_correlation;
    const numFeats = result.num_features;
    const objNames = result.extended_obj_names;

    if (numFeats.length > 0) {
        // zNum[obj_index][feat_index] = Pearson correlation coefficient
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

    // --- 2. Categorical Interaction Heatmaps ---
    const catContainer = document.getElementById('sa-cat-interactions-container');
    const catControls  = document.getElementById('sa-cat-controls');
    catContainer.innerHTML = '';
    catControls.innerHTML  = '';

    const interactions = result.cat_interactions;
    const pairs        = Object.keys(interactions);

    if (pairs.length > 0) {
        // Dropdown controls to select which pair × objective to display
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
        updateInteractionPlot();   // Render the default selection immediately
    } else {
        catContainer.innerHTML = '<p class="text-secondary"><i>No categorical interactions possible (requires 2+ categorical features).</i></p>';
    }

    // --- 3. Parallel Coordinates Plots (one per objective) ---
    const parcoordsContainer = document.getElementById('sa-parcoords-container');
    parcoordsContainer.innerHTML = '';

    const featCols = (window.saSelectedFeatures || currentColumns.filter(c => columnRoles[c] === 'feature'));
    const objCols  = (window.saSelectedObjectives || currentColumns.filter(c => columnRoles[c] === 'objective'));

    if (featCols.length > 0 && objCols.length > 0 && currentData.length > 0) {
        objCols.forEach((objCol, idx) => {
            const plotCols = [...featCols, objCol];

            const dimensions = plotCols.map(col => {
                const colIdx  = currentColumns.indexOf(col);
                const rawVals = currentData.map(r => r[colIdx]);
                const numVals = rawVals.map(v => parseFloat(v));
                const isCat   = numVals.some(isNaN);

                if (isCat) {
                    // Encode categorical strings to integers for parcoords rendering
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

            // Color lines by the objective value (higher = warmer color)
            const objIdx    = currentColumns.indexOf(objCol);
            const colorVals = currentData.map(r => parseFloat(r[objIdx]) || 0);

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
                    // Plasma-like colorscale: dark purple → bright yellow
                    colorscale: [
                        [0, '#0d0887'], [0.11, '#46039f'], [0.22, '#7201a8'], [0.33, '#9c179e'],
                        [0.44, '#bd3786'], [0.55, '#d8576b'], [0.66, '#ed7953'], [0.77, '#fb9f3a'],
                        [0.88, '#fdca26'], [1, '#f0f921']
                    ],
                    showscale: true, reversescale: false, opacity: 0.85,
                    cmin, cmax,
                    colorbar: { title: objCol, thickness: 15 }
                },
                unselected: { line: { opacity: 0, color: 'rgba(0,0,0,0)' } },  // Hide non-selected lines
                dimensions
            }], {
                title: `Parallel Coordinates — Colored by: ${objCol}`,
                margin: { l: 80, r: 80, t: 120, b: 20 }
            }, { responsive: true });
        });
    } else {
        parcoordsContainer.innerHTML = '<p class="text-secondary"><i>Parallel coordinates require at least one feature and one objective with experiments loaded.</i></p>';
    }

    // --- 4. Feature Impact Bar Charts + SHAP Beeswarm Plots ---
    const impactContainer = document.getElementById('sa-impact-container');
    impactContainer.innerHTML = '';

    objNames.forEach((obj, idx) => {
        const res = result.results[obj];
        if (!res || !res.importance) return;

        const imp           = res.importance;
        const sortedEntries = Object.entries(imp).sort((a, b) => a[1] - b[1]);  // Ascending for horizontal bar
        const yFeats        = sortedEntries.map(e => e[0]);
        const xVals         = sortedEntries.map(e => e[1]);

        // Create a flex row container for side-by-side bar + SHAP plots
        const container   = document.createElement('div');
        container.classList.add('impact-row');
        container.style.cssText = 'display:flex;flex-wrap:wrap;gap:15px;margin-bottom:2.5rem;padding:1rem;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;';

        const titleDiv        = document.createElement('h3');
        titleDiv.textContent  = `Feature Impact: ${obj}`;
        titleDiv.style.cssText = 'width:100%;margin:0 0 1rem 0;';
        container.appendChild(titleDiv);

        // Left panel: Feature Importance Bar Chart
        const barId  = `sa-bar-${idx}`;
        const barDiv = document.createElement('div');
        barDiv.id    = barId;
        barDiv.style.cssText = 'flex:1;min-width:350px;height:400px;';
        container.appendChild(barDiv);

        // Right panel: SHAP Beeswarm Plot
        const shapId  = `sa-shap-${idx}`;
        const shapDiv = document.createElement('div');
        shapDiv.id    = shapId;
        shapDiv.style.cssText = 'flex:1.3;min-width:450px;height:400px;';
        container.appendChild(shapDiv);

        impactContainer.appendChild(container);

        const hasShap = res.shap_data && res.shap_data.features && res.shap_data.features.length > 0;

        // Render importance bar chart
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
            // Render SHAP beeswarm: each feature is a trace, points jittered vertically
            const shapTraces = [];
            const sd = res.shap_data;
            sd.features.forEach((feat, i) => {
                const sVals = sd.shap_values[i];
                const fVals = sd.feature_values[i];

                let numericFVals = fVals.map(v => parseFloat(v));
                let isCategorical = numericFVals.some(isNaN);
                let colorArray, cmin, cmax, colorscale, reversescale;

                if (isCategorical) {
                    // Map category strings to integer indices for coloring
                    const uniqueStr = [...new Set(fVals)];
                    colorArray   = fVals.map(v => uniqueStr.indexOf(v));
                    colorscale   = 'Portland'; reversescale = false;
                } else {
                    colorArray  = numericFVals;
                    colorscale  = 'RdBu';
                    cmin        = Math.min(...colorArray);
                    cmax        = Math.max(...colorArray);
                    reversescale = true;  // Red = low value, Blue = high value
                }

                const yBase = yFeats.indexOf(feat);
                shapTraces.push({
                    x: sVals,
                    y: sVals.map(() => yBase + (Math.random() - 0.5) * 0.4),  // Vertical jitter
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

    // --- 5. Reliability Table ---
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

    // --- 6. Estimator Input Panel ---
    const estDiv = document.getElementById('sa-estimator-inputs');
    const feats  = window.saSelectedFeatures || currentColumns.filter(c => columnRoles[c] === 'feature');

    estDiv.innerHTML = feats.map(f => {
        const colIdx     = currentColumns.indexOf(f);
        const uniqueVals = [...new Set(currentData.map(r => r[colIdx]).filter(v => v !== null && v !== ''))];
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


// ===========================================================================
// SA ESTIMATE — POST to /estimate
// ===========================================================================

async function runEstimate() {
    /**
     * Reads the SA estimator input panel values and calls /estimate to get
     * predictions from the fitted sklearn surrogate.
     * Displays per-objective predictions + success scores + global score.
     */
    const inputs  = {};
    document.querySelectorAll('.sa-est-input').forEach(el => inputs[el.dataset.col] = el.value);

    const featList = (window.saSelectedFeatures || currentColumns.filter(c => columnRoles[c] === 'feature'))
        .map(c => ({name: c, range: 'x'}));  // 'x' is a placeholder; SA doesn't use numeric range

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
                data: currentData, columns: currentColumns,
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


// ===========================================================================
// TREND PLOT (BO module)
// ===========================================================================

function renderTrendPlot() {
    /**
     * Renders a multi-line time-series chart of objective values over experiment
     * index (1, 2, 3, …) in the BO module.
     * Skips rows where the objective cell is empty or non-numeric.
     * Hides the entire BO visuals section if no data is plottable.
     */
    if (!currentData || currentData.length === 0) {
        try { Plotly.purge('trend-plot'); } catch(e) {}
        const visuals = document.getElementById('bo-visuals');
        if (visuals) visuals.classList.add('hidden');
        return;
    }

    const objCols = currentColumns.filter(c => columnRoles[c] === 'objective');

    const traces = objCols.map(col => {
        const colIdx = currentColumns.indexOf(col);
        const xDist  = [];
        const yDist  = [];

        currentData.forEach((row, i) => {
            const rawVal = row[colIdx];
            const parsed = parseFloat(rawVal);
            if (rawVal !== '' && !isNaN(parsed)) {
                xDist.push(i + 1);   // 1-indexed experiment number
                yDist.push(parsed);
            }
        });

        return { x: xDist, y: yDist, name: col, mode: 'lines+markers', line: { width: 3 }, marker: { size: 8 } };
    }).filter(t => t.x.length > 0);   // Skip objectives with no numeric data

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

    // Force resize after rendering to handle flex-layout containers
    setTimeout(() => { Plotly.Plots.resize('trend-plot'); }, 100);

    renderParetoPlot();   // Always refresh Pareto front alongside trend
}


// ===========================================================================
// PARETO FRONT DETECTION
// ===========================================================================

function detectParetoIndices(data, objCols) {
    /**
     * Identifies non-dominated (Pareto optimal) rows in the dataset.
     *
     * A point p1 is dominated by p2 if p2 is at least as good as p1 on ALL
     * objectives AND strictly better on at least one.  The scoring function
     * accounts for maximize / minimize / target objective types.
     *
     * @param {Array[]}  data    - currentData (array of arrays)
     * @param {string[]} objCols - names of objective columns to compare
     * @returns {number[]} Indices of non-dominated rows
     */
    const dataPoints = data.map(r => {
        const point = {};
        objCols.forEach(col => point[col] = parseFloat(r[currentColumns.indexOf(col)]) || 0);
        return point;
    });

    return dataPoints.map((p1, i) => {
        const isDominated = dataPoints.some((p2, j) => {
            if (i === j) return false;
            let p2AtLeastAsGoodAsP1   = true;
            let p2StrictlyBetterThanP1 = false;

            for (const col of objCols) {
                const config = objectiveConfigs[col] || { type: 'maximize', importance: 100 };
                const v1 = p1[col], v2 = p2[col];
                const target = parseFloat(config.target || 0);

                let v1Score, v2Score;
                if      (config.type === 'maximize') { v1Score = v1;               v2Score = v2; }
                else if (config.type === 'minimize') { v1Score = -v1;              v2Score = -v2; }
                else                                 { v1Score = -Math.abs(v1 - target); v2Score = -Math.abs(v2 - target); }

                if (v2Score < v1Score) p2AtLeastAsGoodAsP1   = false;
                if (v2Score > v1Score) p2StrictlyBetterThanP1 = true;
            }
            return p2AtLeastAsGoodAsP1 && p2StrictlyBetterThanP1;
        });
        return !isDominated;
    }).map((val, i) => val ? i : -1).filter(idx => idx !== -1);
}


// ===========================================================================
// PARETO PLOT (BO module)
// ===========================================================================

function renderParetoPlot() {
    /**
     * Renders a 2D or 3D Pareto front scatter plot, or a table for 4+ objectives.
     *
     * For "target" objectives, the axis shows distance from the target value
     * (lower = better) and the axis direction is reversed.
     * Only complete rows (no empty objective values) are included.
     */
    const objCols   = currentColumns.filter(c => columnRoles[c] === 'objective' && (objectiveConfigs[c] ? objectiveConfigs[c].included : true));
    const featCols  = currentColumns.filter(c => columnRoles[c] === 'feature');
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

    // Filter to only rows where ALL objective values are numeric
    const completeData = currentData.filter(row =>
        objCols.every(col => {
            const val = row[currentColumns.indexOf(col)];
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
        objCols.forEach(col  => { text += `${col}: ${completeData[idx][currentColumns.indexOf(col)]}<br>`; });
        text += `<br><i>Conditions:</i><br>`;
        featCols.forEach(col => { text += `${col}: ${completeData[idx][currentColumns.indexOf(col)]}<br>`; });
        return text;
    };

    const transformPoints = (indices, colsToUse) => {
        return indices.map(idx => {
            const p = {};
            colsToUse.forEach(col => {
                const cfg    = objectiveConfigs[col];
                const rawVal = parseFloat(completeData[idx][currentColumns.indexOf(col)]) || 0;
                // For "target" objectives, plot absolute distance from target (lower = better)
                p[col] = (cfg && cfg.type === 'target') ? Math.abs(rawVal - parseFloat(cfg.target || 0)) : rawVal;
            });
            return p;
        });
    };

    const getAxisTitle  = (col) => {
        const cfg = objectiveConfigs[col] || { type: 'maximize', target: 0, importance: 100 };
        return cfg.type === 'target'
            ? `Dist. to ${cfg.target || 0} (${col})`
            : `${col} (${cfg.type === 'maximize' ? 'Max' : 'Min'})`;
    };

    const getAxisLayout = (col) => {
        const cfg    = objectiveConfigs[col] || { type: 'maximize' };
        const layout = { title: getAxisTitle(col) };
        // Reversed axis: lower values appear at the top for minimize/target
        if (cfg.type === 'minimize' || cfg.type === 'target') layout.autorange = 'reversed';
        return layout;
    };

    if (objCols.length > 3) {
        // TABLE VIEW for 4+ objectives
        titleEl.textContent = "Pareto Optimal Conditions";
        plotEl.classList.add('hidden');
        tableCont.classList.remove('hidden');
        exportBtn.classList.remove('hidden');

        const head = document.getElementById('pareto-table-head');
        const body = document.getElementById('pareto-table-body');

        head.innerHTML = '<tr>' + currentColumns.map(c => `<th>${c}</th>`).join('') + '</tr>';
        body.innerHTML = paretoIndices.map(idx => {
            const row = completeData[idx];
            return '<tr>' + row.map(v => `<td>${v}</td>`).join('') + '</tr>';
        }).join('');

    } else {
        // PLOT VIEW for 2 or 3 objectives
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
            // 3D scatter for exactly 3 objectives
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


// ===========================================================================
// CSV EXPORT
// ===========================================================================

async function exportData(data, columns) {
    /**
     * POSTs data to /export to receive a CSV file and triggers a browser download.
     * Uses the Blob + object URL pattern to avoid page navigation.
     */
    const res  = await fetch('/export', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ data, columns })
    });
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'export.csv'; a.click();
}


// ===========================================================================
// SA HTML REPORT EXPORT
// ===========================================================================

async function exportSAReport() {
    /**
     * Generates a self-contained HTML report from the current SA session.
     *
     * Strategy:
     *   1. Capture all Plotly plot data (traces + layout) from the live DOM.
     *   2. Serialize the SA result data (categorical interactions, etc.).
     *   3. Build a complete HTML document with embedded Plotly CDN and inline
     *      scripts that re-render the plots from the serialized data.
     *   4. Download as a dated .html file.
     *
     * The resulting file is portable — it requires only internet access for
     * the Plotly CDN.
     */
    const overlay = document.getElementById('loading-overlay');
    document.getElementById('loading-text').textContent = "Generating Dynamic Report...";
    overlay.classList.remove('hidden');

    try {
        const timestamp   = new Date().toLocaleString();
        const activeModel = document.getElementById('sa-model-select').options[document.getElementById('sa-model-select').selectedIndex].text;

        // Capture all Plotly data from active plot elements
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
            capturePlot(el.id, el.layout?.title?.text || el.layout?.title || 'Feature Impact');
        });

        const reliabilityHtml  = document.getElementById('sa-reliability-info').innerHTML;
        const catInteractions  = window.lastSAResult ? window.lastSAResult.cat_interactions : {};
        const objNames         = window.lastSAResult ? window.lastSAResult.extended_obj_names : [];

        // Build the self-contained HTML report document
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
        // All plot data and categorical interactions are embedded as JSON literals
        const plotsData      = ${JSON.stringify(plotsData)};
        const catInteractions = ${JSON.stringify(catInteractions)};

        // 1. Numerical Correlation Heatmap
        const corrPlot = plotsData.find(p => p.id === 'sa-num-corr');
        if (corrPlot) {
            document.getElementById('num-corr-section').style.display = 'block';
            Plotly.newPlot('sa-num-corr', corrPlot.data, corrPlot.layout, { responsive: true });
        }

        // 2. Categorical Interaction Heatmap (dynamic selection)
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

        // 3. Parallel Coordinates Plots
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

        // 4. Feature Impact Groups (bar + SHAP pairs)
        const impactSection = document.getElementById('impact-section');
        const barPlots      = plotsData.filter(p => p.id.startsWith('sa-bar'));
        if (barPlots.length > 0) {
            const head = document.createElement('h2');
            head.textContent = 'Feature Impact Analysis';
            impactSection.appendChild(head);
            barPlots.forEach((p, idx) => {
                const group   = document.createElement('div');
                group.className = 'card';
                group.innerHTML = '<h3>' + (p.layout.title?.text || p.layout.title || 'Feature Impact Group') + '</h3><div class="impact-group"></div>';
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
                    // Show y-axis tick labels in the report (hidden in main app to save space)
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


// ===========================================================================
// BO PERFORMANCE TABLE
// ===========================================================================

function renderBOPerformanceTable(performance) {
    /**
     * Renders the cross-validated R² performance table below the BO suggestion table.
     * Provides status badges and actionable advice per objective.
     */
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


// ===========================================================================
// BO ESTIMATOR PANEL
// ===========================================================================

function renderBOEstimator() {
    /**
     * Dynamically builds the BO estimator input panel with:
     *   - Number inputs for continuous/regular/discrete features.
     *   - Select dropdowns for categorical features (populated from the range string).
     */
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


// ===========================================================================
// BO ESTIMATE — POST to /estimate_bo
// ===========================================================================

async function runBOEstimate() {
    /**
     * Reads the BO estimator input panel and calls /estimate_bo to get GP
     * predictions from the fitted surrogate model.
     * Displays per-objective predictions, success scores, and global score.
     */
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
            body: JSON.stringify({ data: currentData, columns: currentColumns, features, objectives, tweaks, inputs })
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

// ===========================================================================
// CONSTRAINTS HELPER LOGIC (DoE & BO)
// ===========================================================================

function getNumericalFeatures(modulePrefix) {
    if (modulePrefix === 'doe') {
        return Array.from(document.querySelectorAll('#doe-features-list tr'))
            .map(tr => ({
                name: tr.querySelector('.doe-f-name')?.value,
                type: tr.querySelector('.doe-f-type')?.value
            }))
            .filter(f => f.name && f.type !== 'categorical')
            .map(f => f.name);
    } else {
        return currentColumns.filter(col => {
            return columnRoles[col] === 'feature' && boFeatureConfigs[col] && boFeatureConfigs[col].type !== 'categorical';
        });
    }
}

function getCategoricalFeatures(modulePrefix) {
    if (modulePrefix === 'doe') {
        return Array.from(document.querySelectorAll('#doe-features-list tr'))
            .map(tr => {
                const name = tr.querySelector('.doe-f-name')?.value;
                const type = tr.querySelector('.doe-f-type')?.value;
                const range = tr.querySelector('.doe-f-range')?.value || "";
                const choices = range.split(',').map(c => c.trim()).filter(c => c);
                return { name, type, choices };
            })
            .filter(f => f.name && f.type === 'categorical');
    } else {
        return currentColumns
            .filter(col => columnRoles[col] === 'feature' && boFeatureConfigs[col] && boFeatureConfigs[col].type === 'categorical')
            .map(col => {
                const range = boFeatureConfigs[col].range || "";
                const choices = range.split(',').map(c => c.trim()).filter(c => c);
                return { name: col, choices };
            });
    }
}

window.addNumericalConstraintRow = (modulePrefix, equation = "") => {
    const list = document.getElementById(`${modulePrefix}-numerical-constraints-list`);
    if (!list) return;
    const tr = document.createElement('tr');
    tr.className = 'constraint-row';
    
    const numFeatures = getNumericalFeatures(modulePrefix);
    const options = numFeatures.map(f => `<option value="${f}">${f}</option>`).join('');
    
    tr.innerHTML = `
        <td>
            <input type="text" class="constraint-equation" value="${equation}" placeholder="e.g. A + B <= 10" style="width: 100%;" oninput="validateNumericalConstraintRow(this, '${modulePrefix}')">
            <span class="constraint-error" style="display: none; color: var(--danger-color); font-size: 0.8rem; margin-top: 3px;"></span>
        </td>
        <td>
            <select class="constraint-feature-helper" onchange="insertFeatureIntoEquation(this)" style="width: 100%;">
                <option value="">-- Insert Feature --</option>
                ${options}
            </select>
        </td>
        <td>
            <button class="btn secondary small delete-constraint" onclick="this.closest('tr').remove(); updateAllConstraintsDropdowns('${modulePrefix}');" type="button">×</button>
        </td>
    `;
    list.appendChild(tr);
};

window.validateNumericalConstraintRow = (inputEl, modulePrefix) => {
    const equation = inputEl.value;
    const errorSpan = inputEl.parentNode.querySelector('.constraint-error');
    const numFeatures = getNumericalFeatures(modulePrefix);
    
    let duplicates = [];
    numFeatures.forEach(f => {
        const regex = new RegExp(`\\b${escapeRegExp(f)}\\b`, 'g');
        const matches = equation.match(regex);
        if (matches && matches.length > 1) {
            duplicates.push(f);
        }
    });
    
    if (duplicates.length > 0) {
        inputEl.style.borderColor = 'var(--danger-color)';
        errorSpan.textContent = `Error: Feature(s) [${duplicates.join(', ')}] used more than once.`;
        errorSpan.style.display = 'block';
    } else {
        inputEl.style.borderColor = '';
        errorSpan.textContent = '';
        errorSpan.style.display = 'none';
    }
};

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

window.insertFeatureIntoEquation = (selectEl) => {
    const val = selectEl.value;
    if (!val) return;
    const tr = selectEl.closest('tr');
    const inputEl = tr.querySelector('.constraint-equation');
    
    const currentVal = inputEl.value;
    const separator = (currentVal === "" || /[\s+\-*/()]$/.test(currentVal)) ? "" : " ";
    inputEl.value = currentVal + separator + val;
    
    selectEl.value = "";
    inputEl.dispatchEvent(new Event('input', { bubbles: true }));
};

window.addDisallowedCombinationRow = (modulePrefix, conditions = []) => {
    const list = document.getElementById(`${modulePrefix}-disallowed-constraints-list`);
    if (!list) return;
    const tr = document.createElement('tr');
    tr.className = 'disallowed-row';
    
    tr.innerHTML = `
        <td>
            <div class="conditions-container" style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;"></div>
            <button class="btn secondary small add-condition-btn" style="margin-top: 5px;" type="button" onclick="addConditionToRow(this, '${modulePrefix}')">+ Add Variable Condition</button>
        </td>
        <td>
            <button class="btn secondary small delete-constraint" onclick="this.closest('tr').remove();" type="button">×</button>
        </td>
    `;
    list.appendChild(tr);
    
    const container = tr.querySelector('.conditions-container');
    if (conditions.length > 0) {
        conditions.forEach(c => {
            addConditionMarkup(container, modulePrefix, c.feature, c.value);
        });
    } else {
        addConditionMarkup(container, modulePrefix);
    }
};

window.addConditionToRow = (btnEl, modulePrefix) => {
    const container = btnEl.parentNode.querySelector('.conditions-container');
    addConditionMarkup(container, modulePrefix);
};

function addConditionMarkup(container, modulePrefix, selectedFeature = "", selectedValue = "") {
    const catFeatures = getCategoricalFeatures(modulePrefix);
    if (catFeatures.length === 0) {
        container.innerHTML = '<span class="text-secondary"><i>No categorical features available.</i></span>';
        return;
    }
    
    const conditionDiv = document.createElement('div');
    conditionDiv.className = 'condition-item';
    conditionDiv.style.cssText = 'display: flex; align-items: center; gap: 4px; background: var(--bg-secondary); padding: 4px 8px; border-radius: 6px; border: 1px solid var(--border-color);';
    
    const featureOptions = catFeatures.map(f => `<option value="${f.name}" ${f.name === selectedFeature ? 'selected' : ''}>${f.name}</option>`).join('');
    
    const currentFeatureName = selectedFeature || catFeatures[0].name;
    const currentFeature = catFeatures.find(f => f.name === currentFeatureName);
    const valOptions = currentFeature ? currentFeature.choices.map(c => `<option value="${c}" ${c === selectedValue ? 'selected' : ''}>${c}</option>`).join('') : '';
    
    const andLabel = container.children.length > 0 ? '<span class="and-label" style="font-weight: bold; color: var(--accent-color); margin-right: 4px;">AND</span>' : '';
    
    conditionDiv.innerHTML = `
        ${andLabel}
        <select class="condition-feature" onchange="updateConditionValues(this, '${modulePrefix}')" style="width: auto;">
            ${featureOptions}
        </select>
        <span style="font-weight: 500;">==</span>
        <select class="condition-value" style="width: auto;">
            ${valOptions}
        </select>
        <button type="button" class="btn secondary small" onclick="removeCondition(this)" style="padding: 0 4px; line-height: 1; min-height: auto;">×</button>
    `;
    container.appendChild(conditionDiv);
}

window.updateConditionValues = (selectEl, modulePrefix) => {
    const featureName = selectEl.value;
    const catFeatures = getCategoricalFeatures(modulePrefix);
    const feat = catFeatures.find(f => f.name === featureName);
    const valSelect = selectEl.parentNode.querySelector('.condition-value');
    if (feat && valSelect) {
        valSelect.innerHTML = feat.choices.map(c => `<option value="${c}">${c}</option>`).join('');
    }
};

window.removeCondition = (btnEl) => {
    const conditionItem = btnEl.closest('.condition-item');
    const container = conditionItem.parentNode;
    conditionItem.remove();
    
    if (container.children.length > 0) {
        const first = container.children[0];
        const andLabel = first.querySelector('.and-label');
        if (andLabel) andLabel.remove();
    }
};

window.updateAllConstraintsDropdowns = (modulePrefix) => {
    const numFeatures = getNumericalFeatures(modulePrefix);
    const helperSelects = document.querySelectorAll(`#${modulePrefix}-numerical-constraints-list .constraint-feature-helper`);
    helperSelects.forEach(select => {
        const currentVal = select.value;
        select.innerHTML = `<option value="">-- Insert Feature --</option>` + 
            numFeatures.map(f => `<option value="${f}">${f}</option>`).join('');
        select.value = currentVal;
    });

    const equations = document.querySelectorAll(`#${modulePrefix}-numerical-constraints-list .constraint-equation`);
    equations.forEach(eq => {
        validateNumericalConstraintRow(eq, modulePrefix);
    });

    const catFeatures = getCategoricalFeatures(modulePrefix);
    const conditionItems = document.querySelectorAll(`#${modulePrefix}-disallowed-constraints-list .condition-item`);
    conditionItems.forEach(item => {
        const featSelect = item.querySelector('.condition-feature');
        const valSelect = item.querySelector('.condition-value');
        if (featSelect && valSelect) {
            const currentFeat = featSelect.value;
            const currentVal = valSelect.value;
            
            featSelect.innerHTML = catFeatures.map(f => `<option value="${f.name}">${f.name}</option>`).join('');
            
            if (catFeatures.some(f => f.name === currentFeat)) {
                featSelect.value = currentFeat;
                const featObj = catFeatures.find(f => f.name === currentFeat);
                valSelect.innerHTML = featObj.choices.map(c => `<option value="${c}">${c}</option>`).join('');
                if (featObj.choices.includes(currentVal)) {
                    valSelect.value = currentVal;
                }
            } else {
                featSelect.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    });
};
