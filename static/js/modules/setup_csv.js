import { parseCategoricalChoices } from '../utils.js';
import { renderBOSetup } from './bo.js';
import { renderSASetup } from './sa.js';
import {
    addDisallowedCombinationRow,
    addNumericalConstraintRow,
    addObjectiveConstraintRow,
    updateAllConstraintsDropdowns,
} from './constraints.js';

const MODULES = ['doe', 'bo', 'sa'];

function rows(selector) {
    return Array.from(document.querySelectorAll(selector));
}

function field(row, selector) {
    return row.querySelector(selector)?.value ?? '';
}

function featureConstraints(module) {
    if (module === 'sa') return { numerical: [], disallowed: [] };
    const numerical = rows(`#${module}-numerical-constraints-list .constraint-row`)
        .map(row => field(row, '.constraint-equation'))
        .filter(expression => expression.trim());
    const disallowed = rows(`#${module}-disallowed-constraints-list .disallowed-row`)
        .map(row => rowsFrom(row, '.condition-item').map(item => ({
            feature: field(item, '.condition-feature'),
            value: field(item, '.condition-value'),
        })))
        .filter(conditions => conditions.length);
    return { numerical, disallowed };
}

function rowsFrom(element, selector) {
    return Array.from(element.querySelectorAll(selector));
}

function objectiveConstraints(module) {
    if (module === 'doe') return [];
    return rows(`#${module}-objective-constraints-list .objective-constraint-row`).map(row => ({
        objective: field(row, '.objective-constraint-objective'),
        operator: field(row, '.objective-constraint-operator'),
        value: field(row, '.objective-constraint-value'),
    }));
}

export function collectSetup(module) {
    if (!MODULES.includes(module)) throw new Error('Unknown setup module.');
    let features;
    let objectives;
    if (module === 'doe') {
        features = rows('#doe-features-list tr').map(row => ({
            name: field(row, '.doe-f-name'),
            type: field(row, '.doe-f-type'),
            range: field(row, '.doe-f-range'),
            included: true,
        }));
        objectives = rows('#doe-objectives-list tr').map(row => ({
            name: field(row, '.doe-o-name'),
            type: field(row, '.doe-o-type'),
            target: field(row, '.doe-o-target'),
            importance: field(row, '.doe-o-importance'),
            included: true,
        }));
    } else {
        const prefix = module === 'bo' ? 'bo' : 'sa';
        features = rows(`#${prefix}-features-list tr`).map(row => ({
            name: row.querySelector(`.${prefix}-f-type`)?.dataset.col ?? '',
            type: field(row, `.${prefix}-f-type`),
            range: module === 'bo' ? field(row, '.bo-f-range') : '',
            included: row.querySelector(`.${prefix}-f-include`)?.checked ?? true,
        }));
        objectives = rows(`#${prefix}-objectives-list tr`).map(row => ({
            name: row.querySelector(`.${prefix}-o-type`)?.dataset.col ?? '',
            type: field(row, `.${prefix}-o-type`),
            target: field(row, `.${prefix}-o-target`),
            importance: field(row, `.${prefix}-o-importance`),
            included: row.querySelector(`.${prefix}-o-include`)?.checked ?? true,
        }));
    }
    return {
        features,
        objectives,
        constraints: featureConstraints(module),
        objective_constraints: objectiveConstraints(module),
    };
}

function restoreDoE(setup) {
    document.getElementById('doe-features-list').replaceChildren();
    document.getElementById('doe-objectives-list').replaceChildren();
    setup.features.forEach(feature => {
        document.getElementById('doe-add-feature-btn').click();
        const row = document.getElementById('doe-features-list').lastElementChild;
        row.querySelector('.doe-f-name').value = feature.name;
        row.querySelector('.doe-f-type').value = feature.type;
        row.querySelector('.doe-f-range').value = feature.range ?? '';
        window.updatePlaceholder?.(row.querySelector('.doe-f-type'));
    });
    setup.objectives.forEach(objective => {
        document.getElementById('doe-add-objective-btn').click();
        const row = document.getElementById('doe-objectives-list').lastElementChild;
        row.querySelector('.doe-o-name').value = objective.name;
        row.querySelector('.doe-o-type').value = objective.type;
        row.querySelector('.doe-o-target').value = objective.target ?? '';
        window.toggleRowTarget?.(row.querySelector('.doe-o-type'));
    });
    rows('#doe-objectives-list tr').forEach((row, index) => {
        row.querySelector('.doe-o-importance').value = setup.objectives[index].importance ?? '';
    });
    updateAllConstraintsDropdowns('doe');
}

function restoreDataBacked(module, setup) {
    const roles = { ...window.columnRoles };
    const boFeatures = { ...window.boFeatureConfigs };
    const saFeatures = { ...window.saFeatureConfigs };
    const objectives = { ...window.objectiveConfigs };
    for (const feature of setup.features) {
        roles[feature.name] = 'feature';
        if (module === 'bo') {
            const previous = boFeatures[feature.name] || {};
            const range = feature.range || (previous.type === feature.type ? previous.range : '') || '';
            boFeatures[feature.name] = {
                type: feature.type,
                range,
                choices: feature.type === 'categorical' && range ? parseCategoricalChoices(range) : undefined,
                included: feature.included,
            };
        } else {
            saFeatures[feature.name] = { type: feature.type, included: feature.included };
        }
        delete objectives[feature.name];
    }
    for (const objective of setup.objectives) {
        roles[objective.name] = 'objective';
        objectives[objective.name] = { ...objective };
        delete boFeatures[objective.name];
        delete saFeatures[objective.name];
    }
    window.columnRoles = roles;
    window.boFeatureConfigs = boFeatures;
    window.saFeatureConfigs = saFeatures;
    window.objectiveConfigs = objectives;
    window.renderMainTable?.();
    if (module === 'bo') renderBOSetup();
    else renderSASetup();
}

function restoreConstraints(module, setup) {
    if (module !== 'sa') {
        document.getElementById(`${module}-numerical-constraints-list`).replaceChildren();
        document.getElementById(`${module}-disallowed-constraints-list`).replaceChildren();
        for (const expression of setup.constraints.numerical) {
            addNumericalConstraintRow(module);
            document.querySelector(`#${module}-numerical-constraints-list tr:last-child .constraint-equation`).value = expression;
        }
        for (const conditions of setup.constraints.disallowed) {
            addDisallowedCombinationRow(module, conditions);
        }
        updateAllConstraintsDropdowns(module);
    }
    if (module !== 'doe') {
        document.getElementById(`${module}-objective-constraints-list`).replaceChildren();
        for (const constraint of setup.objective_constraints) {
            addObjectiveConstraintRow(module, constraint);
        }
    }
}

export function applySetup(module, setup) {
    if (module === 'doe') restoreDoE(setup);
    else restoreDataBacked(module, setup);
    restoreConstraints(module, setup);
}

function setStatus(module, message, error = false) {
    const status = document.getElementById(`${module}-setup-csv-status`);
    if (!status) return;
    status.textContent = message;
    status.style.color = error ? 'var(--danger-color)' : 'var(--text-secondary)';
}

async function errorFrom(response) {
    try {
        const result = await response.json();
        return result.error || `Request failed (HTTP ${response.status}).`;
    } catch {
        return `Request failed (HTTP ${response.status}).`;
    }
}

async function downloadSetup(module) {
    try {
        const response = await fetch('/setup_csv/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ module, setup: collectSetup(module) }),
        });
        if (!response.ok) throw new Error(await errorFrom(response));
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `EDOS_${module === 'doe' ? 'DoE' : module.toUpperCase()}_setup.csv`;
        anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
        setStatus(module, 'Setup CSV downloaded.');
    } catch (error) {
        setStatus(module, `Could not download setup: ${error.message}`, true);
    }
}

async function uploadSetup(module, file) {
    if (!file) return;
    const columns = [...(window.currentColumns || [])];
    if (module !== 'doe' && !columns.length) {
        setStatus(module, 'Load an experimental dataset before uploading this setup.', true);
        return;
    }
    const form = new FormData();
    form.append('file', file);
    form.append('module', module);
    form.append('columns', JSON.stringify(columns));
    setStatus(module, 'Checking setup CSV...');
    try {
        const response = await fetch('/setup_csv/import', { method: 'POST', body: form });
        if (!response.ok) throw new Error(await errorFrom(response));
        const setup = await response.json();
        if (JSON.stringify(window.currentColumns || []) !== JSON.stringify(columns)) {
            throw new Error('The experimental dataset changed while the CSV was being checked. Upload it again.');
        }
        const previous = collectSetup(module);
        const previousRoles = { ...window.columnRoles };
        const previousBO = { ...window.boFeatureConfigs };
        const previousSA = { ...window.saFeatureConfigs };
        const previousObjectives = { ...window.objectiveConfigs };
        try {
            applySetup(module, setup);
        } catch (error) {
            window.columnRoles = previousRoles;
            window.boFeatureConfigs = previousBO;
            window.saFeatureConfigs = previousSA;
            window.objectiveConfigs = previousObjectives;
            applySetup(module, previous);
            throw error;
        }
        const summary = `Loaded ${setup.features.length} features, ${setup.objectives.length} objectives and their constraints.`;
        setStatus(module, setup.warnings?.length ? `${summary} ${setup.warnings.join(' ')}` : summary);
    } catch (error) {
        setStatus(module, `Could not upload setup: ${error.message}`, true);
    }
}

export function initSetupCSV() {
    MODULES.forEach(module => {
        document.getElementById(`${module}-setup-download-btn`)?.addEventListener('click', () => downloadSetup(module));
        const input = document.getElementById(`${module}-setup-upload-input`);
        document.getElementById(`${module}-setup-upload-btn`)?.addEventListener('click', () => input?.click());
        input?.addEventListener('change', event => {
            uploadSetup(module, event.target.files?.[0]);
            event.target.value = '';
        });
    });
}
