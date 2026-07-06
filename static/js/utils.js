export function safeAddListener(id, event, callback) {
    const el = document.getElementById(id);
    if (el) el.addEventListener(event, callback);
    else console.warn(`Element with ID "${id}" not found. skipping listener.`);
}

export function exportData(data, columns) {
    // Build CSV entirely client-side — no server round-trip needed
    const escape = (v) => {
        const s = (v === null || v === undefined) ? '' : String(v);
        return s.includes(',') || s.includes('"') || s.includes('\n')
            ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const rows = [
        columns.map(escape).join(','),
        ...data.map(row => row.map(escape).join(','))
    ];
    const csv  = rows.join('\r\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'export.csv'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}

export function detectParetoIndices(data, objCols) {
    const dataPoints = data.map(r => {
        const point = {};
        objCols.forEach(col => point[col] = parseFloat(r[window.currentColumns.indexOf(col)]) || 0);
        return point;
    });

    return dataPoints.map((p1, i) => {
        const isDominated = dataPoints.some((p2, j) => {
            if (i === j) return false;
            let p2AtLeastAsGoodAsP1   = true;
            let p2StrictlyBetterThanP1 = false;

            for (const col of objCols) {
                const config = window.objectiveConfigs[col] || { type: 'maximize', importance: 100 };
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

export function rebalanceRemaining(changedIdOrEl, newValue, selector) {
    const inputs = Array.from(document.querySelectorAll(selector));
    let activeInputs = inputs;

    if (selector.includes('bo') || selector.includes('sa')) {
        activeInputs = inputs.filter(el => {
            const col = el.dataset.col;
            return window.objectiveConfigs[col] && window.objectiveConfigs[col].included;
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

    const total = 1000;

    if (changedIdx < activeInputs.length - 1) {
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

export function rebalanceObjectiveConfigs() {
    const objCols      = Object.keys(window.objectiveConfigs);
    const includedCols = objCols.filter(c => window.objectiveConfigs[c].included);

    objCols.forEach(col => {
        if (!window.objectiveConfigs[col].included) window.objectiveConfigs[col].importance = (0).toFixed(1);
    });

    if (includedCols.length === 0) return;

    const total = 1000;
    const share = Math.floor(total / includedCols.length);
    let currentSum = 0;

    includedCols.forEach((col, idx) => {
        if (idx === includedCols.length - 1) {
            window.objectiveConfigs[col].importance = ((total - currentSum) / 10).toFixed(1);
        } else {
            window.objectiveConfigs[col].importance = (share / 10).toFixed(1);
            currentSum += share;
        }
    });
}
