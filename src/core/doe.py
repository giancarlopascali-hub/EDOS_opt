import re
import random
import numpy as np
import pyDOE2 as doe
from src.core.constraints import check_constraints

def generate_doe(features_config, tweaks):
    """
    Generates a Design of Experiments (DoE) table using classical statistical
    designs from the pyDOE2 library.
    """
    model_type = tweaks.get('model', 'bbdesign')   # Design algorithm
    max_runs   = int(float(tweaks.get('max_runs', 50)))   # Truncation limit

    # Parse and Validate Features
    valid_features = []
    for f in features_config:
        name    = f.get('name', 'Unnamed')
        f_type  = f.get('type', 'continuous')
        f_range = f.get('range', '')

        if not f_range:
            continue

        if f_type == 'categorical':
            choices = [x.strip() for x in f_range.split(',') if x.strip()]
            if choices:
                valid_features.append({'name': name, 'choices': choices, 'type': 'categorical', 'range': f_range})

        elif f_type == 'regular':
            matches = [float(m) for m in re.findall(r"[-+]?\d*\.?\d+", f_range)]
            if len(matches) >= 3:
                low, high, steps = sorted([matches[0], matches[1]]) + [int(matches[2])]
                valid_features.append({
                    'name': name, 'low': low, 'high': high, 'type': f_type,
                    'range': f_range, 'levels': np.linspace(low, high, steps).tolist()
                })
            elif len(matches) == 2:
                low, high = min(matches), max(matches)
                valid_features.append({
                    'name': name, 'low': low, 'high': high, 'type': f_type,
                    'range': f_range, 'levels': np.linspace(low, high, 10).tolist()
                })

        else:
            matches = [float(m) for m in re.findall(r"[-+]?\d*\.?\d+", f_range)]
            if len(matches) >= 2:
                low, high = min(matches), max(matches)
                valid_features.append({'name': name, 'low': low, 'high': high, 'type': f_type, 'range': f_range})

    n_factors = int(len(valid_features))

    if n_factors < 2:
        raise ValueError(f'DoE requires at least 2 features. Found {n_factors} valid features.')
    if model_type == 'bbdesign' and n_factors < 3:
        raise ValueError(f'Box-Behnken design requires at least 3 factors. Found {n_factors}.')

    # Generate Design Matrix (in [-1, 1] coded space)
    try:
        if model_type == 'ccdesign':
            design = doe.ccdesign(n_factors)
        elif model_type == 'bbdesign':
            design = doe.bbdesign(n_factors)
        elif model_type == 'fracfact':
            if n_factors <= 4:
                design = doe.fullfact([2] * n_factors).astype(float)
                design = design * 2 - 1
            else:
                design = doe.pbdesign(n_factors).astype(float)
        elif model_type in ('def_screening', 'pbdesign'):
            design = doe.pbdesign(n_factors).astype(float)
        elif model_type == 'lhs':
            from pyDOE2 import lhs as lhs_sampler
            n_numeric = sum(1 for f in valid_features if f['type'] != 'categorical')
            constraints_cfg = tweaks.get('constraints', {})
            has_constraints = bool(constraints_cfg.get('numerical') or constraints_cfg.get('disallowed'))
            n_samples = max(max_runs * 15, 1000) if has_constraints else max_runs
            if n_numeric > 0:
                lhs_numeric = lhs_sampler(n_numeric, samples=n_samples, criterion='maximin')
                lhs_numeric = lhs_numeric * 2 - 1
            else:
                lhs_numeric = np.empty((n_samples, 0))
            design = np.zeros((n_samples, len(valid_features)))
            num_col = 0
            for col_idx, feat in enumerate(valid_features):
                if feat['type'] != 'categorical':
                    design[:, col_idx] = lhs_numeric[:, num_col]
                    num_col += 1
        else:
            design = doe.bbdesign(n_factors)

    except Exception as design_err:
        print(f"Design generation error: {design_err}")
        design     = doe.pbdesign(n_factors).astype(float)
        model_type = "Plackett-Burman (Fallback)"

    # Truncate to max_runs if needed
    constraints_cfg = tweaks.get('constraints', {})
    has_constraints = bool(constraints_cfg.get('numerical') or constraints_cfg.get('disallowed'))
    if not has_constraints:
        original_len = len(design)
        if original_len > max_runs:
            design = design[:max_runs]

    # Convert Coded Design to Feature Space
    suggested_table = []
    for row_idx, row in enumerate(design):
        entry = {}
        for i, feat in enumerate(valid_features):
            if feat['type'] == 'categorical':
                if model_type == 'lhs':
                    n_choices = len(feat['choices'])
                    n_rows    = len(design)
                    strat     = [feat['choices'][j % n_choices] for j in range(n_rows)]
                    rng = random.Random(i)
                    rng.shuffle(strat)
                    entry[feat['name']] = strat[row_idx]
                else:
                    # Map coded value to category:
                    #   +1 → last choice (high level)
                    #   -1 → first choice (low level)
                    #   ~0 → rotate through all choices per row_idx (avoids all center-point
                    #        rows receiving the same fixed middle level)
                    if   row[i] >  0.5: idx = len(feat['choices']) - 1
                    elif row[i] < -0.5: idx = 0
                    else:               idx = row_idx % len(feat['choices'])
                    entry[feat['name']] = feat['choices'][idx]
            else:
                val = feat['low'] + (row[i] + 1) / 2 * (feat['high'] - feat['low'])

                if feat['type'] == 'discrete':
                    d_vals = sorted([float(x.strip()) for x in feat['range'].split(',') if x.strip()])
                    val    = min(d_vals, key=lambda x: abs(x - val)) if d_vals else round(float(val))
                elif feat['type'] == 'regular':
                    d_vals = feat.get('levels', [])
                    val    = min(d_vals, key=lambda x: abs(x - val)) if d_vals else round(float(val), 3)
                else:
                    val = round(float(val), 3)
                entry[feat['name']] = val
        suggested_table.append(entry)

    # Apply constraints
    num_rules = constraints_cfg.get('numerical', [])
    disallowed_rules = constraints_cfg.get('disallowed', [])
    
    if num_rules or disallowed_rules:
        filtered_table = []
        feasible_indices = []
        for idx, entry in enumerate(suggested_table):
            if check_constraints(entry, num_rules, disallowed_rules, valid_features):
                filtered_table.append(entry)
                feasible_indices.append(idx)
                
        if not filtered_table:
            raise ValueError('No generated design points satisfy the specified constraints.')
            
        feasible_indices = feasible_indices[:max_runs]
        suggested_table = [suggested_table[i] for i in feasible_indices]
        if len(design) > 0:
            design = design[feasible_indices]

    # Compute Design Quality Metrics
    try:
        X = design.astype(float)
        N_runs, k_factors = X.shape

        if model_type == 'lhs':
            ortho_score = "N/A"
        elif k_factors > 1:
            corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
            np.fill_diagonal(corr_matrix, 0)
            max_corr    = np.nanmax(corr_matrix) if not np.all(np.isnan(corr_matrix)) else 1.0
            ortho_score = max(0, 100 * (1 - max_corr))
        else:
            ortho_score = 100

        try:
            if model_type == 'lhs':
                d_eff = "N/A"
            else:
                X_full          = np.column_stack([np.ones(N_runs), X])
                k_with_intercept = X_full.shape[1]
                xtx             = np.dot(X_full.T, X_full)
                det             = np.linalg.det(xtx)
                d_eff = (det ** (1.0 / k_with_intercept)) / N_runs * 100 if det > 1e-12 else 0
        except Exception:
            d_eff = 0

        is_truncated = (len(design) < original_len) if 'original_len' in locals() else False
        design_with_intercept = np.column_stack([np.ones(N_runs), X])
        rank = np.linalg.matrix_rank(design_with_intercept)

        if model_type == 'lhs':
            resolution = "N/A (Space-Filling)"
        elif rank < (k_factors + 1):
            resolution = "Poor (Insuff. Runs)"
        elif is_truncated:
            resolution = "Reduced (Truncated)"
        elif model_type in ('bbdesign', 'ccdesign'):
            resolution = "V+ (High)"
        elif model_type == 'fracfact' or 'Plackett' in model_type:
            resolution = "III (Screening)"
        else:
            resolution = "IV (Moderate)"

        numeric_cols = [i for i, f in enumerate(valid_features) if f['type'] != 'categorical']
        if not numeric_cols:
            curvature = "N/A (Cat. only)"
        else:
            has_center = any(np.all(np.abs(r[numeric_cols]) < 0.15) for r in X)
            levels     = len(np.unique(X[:, numeric_cols[0]])) if numeric_cols else 2
            if   levels > 2 and has_center: curvature = "Excellent"
            elif levels > 2 or  has_center: curvature = "Partial"
            else:                           curvature = "None (Linear only)"

        metrics = {
            'orthogonality': round(ortho_score, 1) if isinstance(ortho_score, (int, float)) else ortho_score,
            'efficiency':    round(d_eff,        1) if isinstance(d_eff,        (int, float)) else d_eff,
            'resolution':    resolution,
            'curvature':     curvature,
            'runs':          len(suggested_table),
            'model':         model_type
        }
    except Exception as met_err:
        print(f"Metrics calculation error: {met_err}")
        metrics = {
            'orthogonality': 0, 'efficiency': 0, 'resolution': "Error",
            'curvature': "Error", 'runs': len(suggested_table), 'model': model_type
        }

    return suggested_table, metrics
