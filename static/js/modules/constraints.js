function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function getNumericalFeatures(modulePrefix) {
    if (modulePrefix === 'doe') {
        return Array.from(document.querySelectorAll('#doe-features-list tr'))
            .map(tr => ({
                name: tr.querySelector('.doe-f-name')?.value,
                type: tr.querySelector('.doe-f-type')?.value
            }))
            .filter(f => f.name && f.type !== 'categorical')
            .map(f => f.name);
    } else {
        return window.currentColumns.filter(col => {
            return window.columnRoles[col] === 'feature' && window.boFeatureConfigs[col] && window.boFeatureConfigs[col].type !== 'categorical';
        });
    }
}

export function getCategoricalFeatures(modulePrefix) {
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
        return window.currentColumns
            .filter(col => window.columnRoles[col] === 'feature' && window.boFeatureConfigs[col] && window.boFeatureConfigs[col].type === 'categorical')
            .map(col => {
                const range = window.boFeatureConfigs[col].range || "";
                const choices = range.split(',').map(c => c.trim()).filter(c => c);
                return { name: col, choices };
            });
    }
}

export function addNumericalConstraintRow(modulePrefix, equation = "") {
    const list = document.getElementById(`${modulePrefix}-numerical-constraints-list`);
    if (!list) return;
    const tr = document.createElement('tr');
    tr.className = 'constraint-row';
    
    const numFeatures = getNumericalFeatures(modulePrefix);
    const options = numFeatures.map(f => `<option value="${f}">${f}</option>`).join('');
    
    tr.innerHTML = `
        <td>
            <input type="text" class="constraint-equation" value="${equation}" placeholder="e.g. A + B <= 10" style="width: 100%;" oninput="window.validateNumericalConstraintRow(this, '${modulePrefix}')">
            <span class="constraint-error" style="display: none; color: var(--danger-color); font-size: 0.8rem; margin-top: 3px;"></span>
        </td>
        <td>
            <select class="constraint-feature-helper" onchange="window.insertFeatureIntoEquation(this)" style="width: 100%;">
                <option value="">-- Insert Feature --</option>
                ${options}
            </select>
        </td>
        <td>
            <button class="btn secondary small delete-constraint" onclick="this.closest('tr').remove(); window.updateAllConstraintsDropdowns('${modulePrefix}');" type="button">×</button>
        </td>
    `;
    list.appendChild(tr);
}

export function validateNumericalConstraintRow(inputEl, modulePrefix) {
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
}

export function insertFeatureIntoEquation(selectEl) {
    const val = selectEl.value;
    if (!val) return;
    const tr = selectEl.closest('tr');
    const inputEl = tr.querySelector('.constraint-equation');
    
    const currentVal = inputEl.value;
    const separator = (currentVal === "" || /[\s+\-*/()]$/.test(currentVal)) ? "" : " ";
    inputEl.value = currentVal + separator + val;
    
    selectEl.value = "";
    inputEl.dispatchEvent(new Event('input', { bubbles: true }));
}

export function addDisallowedCombinationRow(modulePrefix, conditions = []) {
    const list = document.getElementById(`${modulePrefix}-disallowed-constraints-list`);
    if (!list) return;
    const tr = document.createElement('tr');
    tr.className = 'disallowed-row';
    
    tr.innerHTML = `
        <td>
            <div class="conditions-container" style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;"></div>
            <button class="btn secondary small add-condition-btn" style="margin-top: 5px;" type="button" onclick="window.addConditionToRow(this, '${modulePrefix}')">+ Add Variable Condition</button>
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
}

export function addConditionToRow(btnEl, modulePrefix) {
    const container = btnEl.parentNode.querySelector('.conditions-container');
    addConditionMarkup(container, modulePrefix);
}

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
        <select class="condition-feature" onchange="window.updateConditionValues(this, '${modulePrefix}')" style="width: auto;">
            ${featureOptions}
        </select>
        <span style="font-weight: 500;">==</span>
        <select class="condition-value" style="width: auto;">
            ${valOptions}
        </select>
        <button type="button" class="btn secondary small" onclick="window.removeCondition(this)" style="padding: 0 4px; line-height: 1; min-height: auto;">×</button>
    `;
    container.appendChild(conditionDiv);
}

export function updateConditionValues(selectEl, modulePrefix) {
    const featureName = selectEl.value;
    const catFeatures = getCategoricalFeatures(modulePrefix);
    const feat = catFeatures.find(f => f.name === featureName);
    const valSelect = selectEl.parentNode.querySelector('.condition-value');
    if (feat && valSelect) {
        valSelect.innerHTML = feat.choices.map(c => `<option value="${c}">${c}</option>`).join('');
    }
}

export function removeCondition(btnEl) {
    const conditionItem = btnEl.closest('.condition-item');
    const container = conditionItem.parentNode;
    conditionItem.remove();
    
    if (container.children.length > 0) {
        const first = container.children[0];
        const andLabel = first.querySelector('.and-label');
        if (andLabel) andLabel.remove();
    }
}

export function updateAllConstraintsDropdowns(modulePrefix) {
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
}

// Bind to window for inline HTML events
window.addNumericalConstraintRow = addNumericalConstraintRow;
window.validateNumericalConstraintRow = validateNumericalConstraintRow;
window.insertFeatureIntoEquation = insertFeatureIntoEquation;
window.addDisallowedCombinationRow = addDisallowedCombinationRow;
window.addConditionToRow = addConditionToRow;
window.updateConditionValues = updateConditionValues;
window.removeCondition = removeCondition;
window.updateAllConstraintsDropdowns = updateAllConstraintsDropdowns;
