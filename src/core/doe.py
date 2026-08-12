import re
import random
import numpy as np
import pyDOE2 as doe
from src.core.constraints import check_constraints


def _doptimal_coordinate_exchange(candidates, N, model_order, max_iter=500, seed=42):
    """
    Coordinate-exchange D-optimal algorithm.

    Selects N rows from *candidates* (a coded [-1, 1] array of shape (M, k))
    that maximise det(X'X) for the specified model order.

    Parameters
    ----------
    candidates  : (M, k) ndarray  – candidate points in coded space
    N           : int             – desired run count
    model_order : str             – 'linear' | 'interaction' | 'quadratic'
    max_iter    : int             – coordinate-exchange iteration limit
    seed        : int             – random seed for reproducibility

    Returns
    -------
    design : (N, k) ndarray
    """
    rng    = np.random.default_rng(seed)
    idx    = rng.choice(len(candidates), size=N, replace=False)
    design = candidates[idx].copy()

    def build_model_matrix(X):
        """Expand X into a model matrix according to model_order."""
        n, k = X.shape
        cols = [np.ones((n, 1)), X]                           # intercept + main effects
        if model_order in ('interaction', 'quadratic'):
            for i in range(k):
                for j in range(i + 1, k):
                    cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
        if model_order == 'quadratic':
            cols.append(X ** 2)
        return np.hstack(cols)

    def det_XtX(X):
        M = build_model_matrix(X)
        return np.linalg.det(M.T @ M)

    best_det = det_XtX(design)
    for _ in range(max_iter):
        improved = False
        for row_i in range(N):
            best_row = design[row_i].copy()
            for cand_row in candidates:
                design[row_i] = cand_row
                d = det_XtX(design)
                if d > best_det + 1e-12:
                    best_det = d
                    best_row = cand_row.copy()
                    improved = True
            design[row_i] = best_row
        if not improved:
            break
    return design


def generate_doe(features_config, tweaks):
    """
    Generates a Design of Experiments (DoE) table using classical statistical
    designs from the pyDOE2 library, or a D-Optimal design via coordinate exchange.
    """
    model_type     = tweaks.get('model', 'bbdesign')
    max_runs       = int(float(tweaks.get('max_runs', 50)))
    use_exact_runs = bool(tweaks.get('use_exact_runs', False))
    oversampling_warning = False

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

    # D-Optimal minimum-N guard (checked before design generation)
    if model_type == 'doptimal':
        model_order = tweaks.get('dopt_model_order', 'quadratic')
        min_n_map = {
            'linear':      n_factors + 1,
            'interaction': n_factors + n_factors * (n_factors - 1) // 2 + 1,
            'quadratic':   (n_factors + 1) * (n_factors + 2) // 2,
        }
        min_n = min_n_map.get(model_order, n_factors + 1)
        if max_runs < min_n:
            raise ValueError(
                f"D-Optimal with '{model_order}' model and {n_factors} factors requires at least "
                f"{min_n} runs (requested {max_runs})."
            )

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
                model_type = 'fullfact'
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
        elif model_type == 'doptimal':
            from pyDOE2 import lhs as lhs_sampler
            model_order  = tweaks.get('dopt_model_order', 'quadratic')
            n_numeric    = sum(1 for f in valid_features if f['type'] != 'categorical')
            n_candidates = max(500, 20 * n_factors)
            if n_numeric > 0:
                candidates = lhs_sampler(n_numeric, samples=n_candidates, criterion='maximin') * 2 - 1
            else:
                candidates = np.empty((n_candidates, 0))
            numeric_design = _doptimal_coordinate_exchange(candidates, max_runs, model_order, seed=42)
            # Pad categorical columns (decoded later in the feature-space conversion step)
            if n_numeric < n_factors:
                cat_pad = np.zeros((max_runs, n_factors - n_numeric))
                design  = np.hstack([numeric_design, cat_pad])
            else:
                design  = numeric_design
        else:
            design = doe.bbdesign(n_factors)

    except Exception as design_err:
        print(f"Design generation error: {design_err}")
        design     = doe.pbdesign(n_factors).astype(float)
        model_type = "Plackett-Burman (Fallback)"

    # ------------------------------------------------------------------
    # Exact-N / oversampling logic (structured models only)
    # LHS and D-Optimal are inherently exact-N and never padded here.
    # ------------------------------------------------------------------
    constraints_cfg = tweaks.get('constraints', {})
    has_constraints = bool(constraints_cfg.get('numerical') or constraints_cfg.get('disallowed'))

    structured_models = {'bbdesign', 'ccdesign', 'fracfact', 'fullfact',
                         'def_screening', 'pbdesign', 'Plackett-Burman (Fallback)'}

    if model_type in structured_models:
        natural_len  = len(design)
        original_len = natural_len          # used later for is_truncated check

        if use_exact_runs:
            if max_runs > natural_len:
                # Oversampling: pad with jittered copies of existing rows
                oversampling_warning = True
                n_to_pad = max_runs - natural_len
                rng_pad  = np.random.default_rng(42)
                pad_idx  = rng_pad.integers(0, natural_len, size=n_to_pad)
                pad_rows = design[pad_idx].copy()
                jitter   = rng_pad.uniform(-0.02, 0.02, pad_rows.shape)
                pad_rows = np.clip(pad_rows + jitter, -1.0, 1.0)
                design   = np.vstack([design, pad_rows])
            elif max_runs < natural_len:
                # Trim to exact count (quality metrics will reflect the reduction)
                design = design[:max_runs]
        elif not has_constraints:
            # Max mode (original behaviour): truncate ceiling only
            if natural_len > max_runs:
                design = design[:max_runs]
    elif model_type == 'lhs' and not has_constraints:
        # LHS: already generated exactly max_runs samples (no further action needed)
        original_len = len(design)
    else:
        original_len = len(design)

    # Convert Coded Design to Feature Space
    suggested_table = []
    for row_idx, row in enumerate(design):
        entry = {}
        for i, feat in enumerate(valid_features):
            if feat['type'] == 'categorical':
                if model_type in ('lhs', 'doptimal'):
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
    num_rules        = constraints_cfg.get('numerical', [])
    disallowed_rules = constraints_cfg.get('disallowed', [])

    if num_rules or disallowed_rules:
        filtered_table   = []
        feasible_indices = []
        for idx, entry in enumerate(suggested_table):
            if check_constraints(entry, num_rules, disallowed_rules, valid_features):
                filtered_table.append(entry)
                feasible_indices.append(idx)

        if not filtered_table:
            raise ValueError('No generated design points satisfy the specified constraints.')

        feasible_indices = feasible_indices[:max_runs]
        suggested_table  = [suggested_table[i] for i in feasible_indices]
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
                X_full           = np.column_stack([np.ones(N_runs), X])
                k_with_intercept = X_full.shape[1]
                xtx              = np.dot(X_full.T, X_full)
                det              = np.linalg.det(xtx)
                d_eff = (det ** (1.0 / k_with_intercept)) / N_runs * 100 if det > 1e-12 else 0
        except Exception:
            d_eff = 0

        is_truncated = (len(design) < original_len) if 'original_len' in locals() else False
        design_with_intercept = np.column_stack([np.ones(N_runs), X])
        rank = np.linalg.matrix_rank(design_with_intercept)

        # Resolution
        if model_type == 'lhs':
            resolution = "N/A (Space-Filling)"
        elif model_type == 'doptimal':
            resolution = "D-Optimal"
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

        # Curvature
        if model_type == 'doptimal':
            dopt_order = tweaks.get('dopt_model_order', 'quadratic')
            curvature_map = {
                'quadratic':   'Excellent',
                'interaction': 'Partial',
                'linear':      'None (Linear only)',
            }
            curvature = curvature_map.get(dopt_order, 'Partial')
        else:
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
            'orthogonality':      round(ortho_score, 1) if isinstance(ortho_score, (int, float)) else ortho_score,
            'efficiency':         round(d_eff,        1) if isinstance(d_eff,        (int, float)) else d_eff,
            'resolution':         resolution,
            'curvature':          curvature,
            'runs':               len(suggested_table),
            'model':              model_type,
            'oversampling_warning': oversampling_warning,
        }
    except Exception as met_err:
        print(f"Metrics calculation error: {met_err}")
        metrics = {
            'orthogonality': 0, 'efficiency': 0, 'resolution': "Error",
            'curvature': "Error", 'runs': len(suggested_table), 'model': model_type,
            'oversampling_warning': oversampling_warning,
        }

    return suggested_table, metrics
