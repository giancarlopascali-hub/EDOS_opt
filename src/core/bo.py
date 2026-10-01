import ast
import re
import itertools
import numpy as np
import pandas as pd
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel

from botorch.models import SingleTaskGP, MixedSingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.optim.initializers import gen_batch_initial_conditions

# Acquisition functions
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qUpperConfidenceBound,
)
try:
    from botorch.acquisition import qLogExpectedImprovement
except ImportError:
    qLogExpectedImprovement = qExpectedImprovement

from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
    qExpectedHypervolumeImprovement,
)
try:
    from botorch.acquisition.multi_objective.monte_carlo import qLogNoisyExpectedHypervolumeImprovement
except ImportError:
    qLogNoisyExpectedHypervolumeImprovement = qNoisyExpectedHypervolumeImprovement

from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.normal import SobolQMCNormalSampler

from src.core.constraints import (
    check_constraints,
    check_objective_constraints,
    encode_categorical_series,
    get_categorical_choices,
)
from src.core.feasibility import (
    ProbabilityOfFeasibilityAcquisition,
    candidate_feasibility_report,
)


class SurfaceCalculationCancelled(RuntimeError):
    """Raised when a response-surface job is cancelled by the user."""


STANDARD_BO_MAX_BETA = 10.0


def validate_exploration(value):
    """Return a finite exploration setting constrained to the public 0-1 scale."""
    try:
        exploration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Exploration must be a number between 0 and 1.") from exc
    if not np.isfinite(exploration) or not 0.0 <= exploration <= 1.0:
        raise ValueError("Exploration must be a number between 0 and 1.")
    return exploration


def exploration_to_beta(exploration, max_beta=STANDARD_BO_MAX_BETA):
    """Map 0 to pure exploitation and 1 to the configured maximum exploration."""
    return validate_exploration(exploration) * float(max_beta)


def generate_feasible_initial_conditions(bounds_t, q, num_restarts, p_constraints, max_attempts=500):
    """
    Generates a tensor of shape (num_restarts, q, d) where all points satisfy
    the constraints specified in p_constraints.
    """
    d = bounds_t.shape[1]
    feasible_points = []
    
    for attempt in range(max_attempts):
        samples = draw_sobol_samples(bounds=bounds_t, n=1, q=2000).squeeze(0)
        
        mask = torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)
        for constraint_func, _ in p_constraints:
            val = constraint_func(samples)
            mask = mask & (val >= 0)
            
        feasible_samples = samples[mask]
        feasible_points.append(feasible_samples)
        
        all_feasible = torch.cat(feasible_points, dim=0)
        if all_feasible.shape[0] >= num_restarts * q:
            selected = all_feasible[:num_restarts * q]
            return selected.view(num_restarts, q, d)
            
    all_feasible = torch.cat(feasible_points, dim=0) if feasible_points else torch.empty((0, d), device=bounds_t.device)
    if all_feasible.shape[0] < num_restarts * q:
        if all_feasible.shape[0] == 0:
            return draw_sobol_samples(bounds=bounds_t, n=num_restarts, q=q)
        else:
            repeats = (num_restarts * q // all_feasible.shape[0]) + 1
            tiled = all_feasible.repeat(repeats, 1)[:num_restarts * q]
            return tiled.view(num_restarts, q, d)
            
    return all_feasible[:num_restarts * q].view(num_restarts, q, d)


def clean_dataframe(df, features_config, objectives_config):
    # 1. Feature filtering: Drop rows missing continuous/discrete numerical features.
    # For categorical features, treat missing/empty as 'None' level rather than dropping rows.
    df_clean = df.copy()
    valid_feat_mask = pd.Series(True, index=df_clean.index)
    for f in features_config:
        name = f['name']
        if name not in df_clean.columns:
            continue
        col_series = df_clean[name]
        if f.get('type') in ['continuous', 'discrete', 'regular']:
            num_col = pd.to_numeric(col_series, errors='coerce')
            valid_feat_mask = valid_feat_mask & num_col.notna()
        else:
            df_clean[name] = col_series.fillna('None').astype(str).replace(r'^\s*$', 'None', regex=True)
            
    df_clean = df_clean[valid_feat_mask].copy()
    
    if not objectives_config:
        return df_clean
        
    # 2. Objective filtering: Differentiate single vs multi-objective
    obj_df_numeric = pd.DataFrame()
    for obj in objectives_config:
        if obj['name'] in df_clean.columns:
            obj_df_numeric[obj['name']] = pd.to_numeric(df_clean[obj['name']], errors='coerce')
        else:
            obj_df_numeric[obj['name']] = pd.Series(np.nan, index=df_clean.index)
            
    if len(objectives_config) == 1:
        # Single-objective case: drop any row missing the objective value
        valid_obj_mask = obj_df_numeric[objectives_config[0]['name']].notna()
    else:
        # Multi-objective case: drop rows missing ALL objectives
        valid_obj_mask = obj_df_numeric.notna().any(axis=1)
        
    df_clean = df_clean[valid_obj_mask].copy()
    return df_clean


def infer_decimal_precision(col_series: pd.Series = None, range_str: str = '') -> int:
    """
    Return the maximum number of decimal places seen in col_series or range_str.
    Fallback chain:
      1. Maximum decimals across non-null dataset values
      2. Maximum decimals seen in range_str tokens (if provided)
      3. 1  (safe default – avoids integer-only snapping when no decimals present)
    """
    max_dec = 0
    if col_series is not None and hasattr(col_series, 'empty') and not col_series.empty:
        for raw in col_series.dropna().astype(str):
            m = re.search(r'\.(\d+)$', raw.strip())
            if m:
                max_dec = max(max_dec, len(m.group(1)))
    if max_dec == 0 and range_str:
        for tok in re.findall(r'[-+]?\d*\.?\d+', range_str):
            m = re.search(r'\.(\d+)$', tok)
            if m:
                max_dec = max(max_dec, len(m.group(1)))
    return max_dec if max_dec > 0 else 1


def run_bo_optimization(df, features_config, objectives_config, tweaks, device, compute_device):
    # Filter the dataframe to drop incomplete rows
    df_clean = clean_dataframe(df, features_config, objectives_config)

    # STEP 1 — Build Training Features (X)
    train_x_list    = []
    bounds          = []
    categorical_features = []
    feature_levels  = []

    for i, f in enumerate(features_config):
        name    = f['name']
        f_range = f['range']

        if f['type'] == 'continuous':
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            low, high = sorted([float(matches[0]), float(matches[1])]) if len(matches) >= 2 else (0.0, 1.0)
            if low == high:
                high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(0).values
            train_x_list.append(feat_vals)
            bounds.append([low, high])
            feature_levels.append(None)

        elif f['type'] == 'discrete':
            clean_range = f_range.replace('[', '').replace(']', '')
            values = sorted([float(x.strip()) for x in clean_range.split(',') if x.strip()])
            if not values:
                values = [0.0]
            low, high = values[0], values[-1]
            if low == high:
                high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(values[0]).values
            train_x_list.append(feat_vals)
            bounds.append([low, high])
            feature_levels.append(values)

        elif f['type'] == 'regular':
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            if len(matches) >= 3:
                low, high, steps = sorted([float(matches[0]), float(matches[1])]) + [int(float(matches[2]))]
                values = np.linspace(low, high, steps).tolist()
            elif len(matches) == 2:
                low, high = sorted([float(matches[0]), float(matches[1])])
                values = np.linspace(low, high, 10).tolist()
            else:
                values = [0.0]
            low, high = values[0], values[-1]
            if low == high:
                high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(values[0]).values
            train_x_list.append(feat_vals)
            bounds.append([low, high])
            feature_levels.append(values)

        elif f['type'] == 'categorical':
            encoded, choices = encode_categorical_series(df_clean[name], f)
            train_x_list.append(encoded)
            low, high = 0, len(choices) - 1
            if low == high:
                high = low + 0.1
            bounds.append([low, high])
            categorical_features.append(i)
            feature_levels.append(list(range(len(choices))))

    if not df_clean.empty:
        train_x  = torch.tensor(np.stack(train_x_list, axis=1), dtype=torch.float32).to(device)
    else:
        train_x  = torch.zeros((0, len(features_config)), dtype=torch.float32).to(device)
    bounds_t = torch.tensor(bounds, dtype=torch.float32).T.to(device)

    # Build per-feature decimal precision for continuous/regular/discrete features
    # (used later by get_fingerprint and decode_cand to round suggested values)
    feature_precision = {}
    for f in features_config:
        col_s = (
            df_clean[f['name']].astype(str)
            if (not df_clean.empty and f['name'] in df_clean.columns)
            else pd.Series(dtype=str)
        )
        feature_precision[f['name']] = infer_decimal_precision(col_s, f.get('range', ''))

    # STEP 2 — Build Training Objectives (Y)
    train_y_list = []
    weights_list = []

    for obj in objectives_config:
        weights_list.append(float(obj.get('importance', 1)))
        if not df_clean.empty:
            vals = pd.to_numeric(df_clean[obj['name']], errors='coerce').values
            if obj['type'] == 'minimize':
                train_y_list.append(-vals)
            elif obj['type'] == 'maximize':
                train_y_list.append(vals)
            elif obj['type'] == 'target':
                target   = float(obj.get('target', 0))
                valid_vals = vals[~np.isnan(vals)]
                variance = np.var(valid_vals) if len(valid_vals) > 0 and np.var(valid_vals) > 0 else 1.0
                train_y_list.append(-((vals - target) ** 2) / variance)

    if not df_clean.empty:
        train_y = torch.tensor(np.stack(train_y_list, axis=1), dtype=torch.float32).to(device)
    else:
        train_y = torch.zeros((0, len(objectives_config)), dtype=torch.float32).to(device)

    weights_t   = torch.tensor(weights_list, dtype=torch.float32).to(device)
    batch_size  = int(float(tweaks.get('batch_size', 1)))
    avoid_reval = tweaks.get('avoid_reval', True)
    # acq_type is ignored in multi-objective mode (which always uses EHVI or scalarized posterior transform)
    acq_type    = tweaks.get('acq_type', 'EI')
    exploration = validate_exploration(tweaks.get('exploration', 0.5))

    # Hoist constraint extraction so variables are always in scope (needed on cold start too)
    constraints_cfg  = tweaks.get('constraints', {})
    num_rules        = constraints_cfg.get('numerical', [])
    disallowed_rules = constraints_cfg.get('disallowed', [])
    objective_constraints = tweaks.get('objective_constraints', [])
    feasibility_models = {}

    # STEP 3 — Optimization Logic
    if df_clean.empty:
        # Cold start: generate Sobol candidates and filter by constraints if any
        n_pool = max(batch_size * 500, 2000)
        pool   = draw_sobol_samples(bounds=bounds_t, n=1, q=n_pool).squeeze(0)
        if num_rules or disallowed_rules:
            feasible = []
            for pt in pool:
                decoded = {}
                for idx_f, f in enumerate(features_config):
                    v = pt[idx_f].item()
                    if f['type'] == 'categorical':
                        choices = get_categorical_choices(f)
                        v = choices[max(0, min(len(choices) - 1, int(round(v))))]
                    elif f['type'] in ('discrete', 'regular'):
                        matches = re.findall(r"[-+]?\d*\.?\d+", f['range'])
                        if f['type'] == 'discrete':
                            d_vals = sorted([float(x.strip()) for x in f['range'].replace('[','').replace(']','').split(',') if x.strip()])
                        elif len(matches) >= 3:
                            lo, hi, st = sorted([float(matches[0]), float(matches[1])]) + [int(float(matches[2]))]
                            d_vals = np.linspace(lo, hi, st).tolist()
                        elif len(matches) == 2:
                            d_vals = np.linspace(float(matches[0]), float(matches[1]), 10).tolist()
                        else:
                            d_vals = []
                        if d_vals:
                            v = min(d_vals, key=lambda x: abs(x - v))
                    if isinstance(v, (int, float)):
                        v = round(float(v), 6)
                    decoded[f['name']] = v
                if check_constraints(decoded, num_rules, disallowed_rules, features_config):
                    feasible.append(pt)
                if len(feasible) >= batch_size:
                    break
            if feasible:
                candidates = torch.stack(feasible[:batch_size])
            else:
                candidates = draw_sobol_samples(bounds=bounds_t, n=1, q=batch_size).squeeze(0)
        else:
            candidates = draw_sobol_samples(bounds=bounds_t, n=1, q=batch_size).squeeze(0)
    else:
        kernel_name  = tweaks.get('kernel', 'matern52')
        is_noiseless = tweaks.get('noiseless', False)

        def get_kernel(k_name):
            if k_name == 'rbf':      return ScaleKernel(RBFKernel()).to(device)
            if k_name == 'matern32': return ScaleKernel(MaternKernel(nu=1.5)).to(device)
            if k_name == 'matern12': return ScaleKernel(MaternKernel(nu=0.5)).to(device)
            return ScaleKernel(MaternKernel(nu=2.5)).to(device)

        def build_single_model(tx, ty_slice):
            all_dims  = list(range(tx.shape[1]))
            cont_dims = [idx for idx in all_dims if idx not in categorical_features]

            input_tf = (
                Normalize(
                    d=tx.shape[1],
                    bounds=bounds_t.to(compute_device)[:, cont_dims],
                    indices=cont_dims
                ).to(compute_device)
                if cont_dims else None
            )

            if categorical_features and len(cont_dims) > 0:
                m = MixedSingleTaskGP(
                    tx.to(compute_device), ty_slice.to(compute_device),
                    cat_dims=categorical_features,
                    outcome_transform=Standardize(m=1),
                    input_transform=input_tf
                ).to(compute_device)
            else:
                m = SingleTaskGP(
                    tx.to(compute_device), ty_slice.to(compute_device),
                    covar_module=get_kernel(kernel_name).to(compute_device),
                    outcome_transform=Standardize(m=1),
                    input_transform=input_tf
                ).to(compute_device)

            if is_noiseless:
                # Pin noise to a small constant and freeze it so model fitting
                # cannot increase it. This enforces near-noiseless behavior rather
                # than just softly encouraging low noise via a prior.
                with torch.no_grad():
                    m.likelihood.noise_covar.noise = torch.tensor(1e-4, device=compute_device)
                m.likelihood.noise_covar.raw_noise.requires_grad_(False)

            return m.to(compute_device)

        if train_y.shape[1] > 1:
            models = []
            for idx in range(train_y.shape[1]):
                y_slice = train_y[:, idx:idx+1]
                non_nan_mask = ~torch.isnan(y_slice.squeeze(-1))
                tx_i = train_x[non_nan_mask]
                ty_i = y_slice[non_nan_mask]
                if tx_i.shape[0] == 0:
                    raise ValueError(f"No valid observations for objective '{objectives_config[idx]['name']}' after filtering missing values.")
                models.append(build_single_model(tx_i, ty_i))

            model   = ModelListGP(*models).to(compute_device)
            mll     = SumMarginalLogLikelihood(model.likelihood, model).to(compute_device)
        else:
            model = build_single_model(train_x, train_y).to(compute_device)
            mll   = ExactMarginalLogLikelihood(model.likelihood, model).to(compute_device)

        try:
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 100}})
        except Exception as e:
            print(f"Warning: Main model fit failed: {e}")

        OPT_DEVICE = device if device.type == 'cuda' else compute_device
        model = model.to(OPT_DEVICE)

        intensity = 2 if train_y.shape[1] > 1 else 1
        if device.type == 'cuda':
            intensity *= 6
        else:
            intensity *= 2

        is_mo      = train_y.shape[1] > 1
        is_uniform = (
            torch.allclose(weights_t, weights_t[0].expand_as(weights_t), atol=1e-2)
            if is_mo else True
        )

        base_mc_samples = 64 if is_mo else 256
        mc_samples = int(base_mc_samples * intensity)
        sampler    = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

        acq_is_logarithmic = False
        if is_mo:
            if is_uniform:
                mo_obj = WeightedMCMultiOutputObjective(weights=weights_t.to(OPT_DEVICE))
                ty_compute = train_y.to(OPT_DEVICE)

                mins_list = []
                maxs_list = []
                for idx in range(ty_compute.shape[1]):
                    valid_vals = ty_compute[:, idx][~torch.isnan(ty_compute[:, idx])]
                    if valid_vals.numel() > 0:
                        mins_list.append(valid_vals.min())
                        maxs_list.append(valid_vals.max())
                    else:
                        mins_list.append(torch.tensor(0.0, device=OPT_DEVICE))
                        maxs_list.append(torch.tensor(1.0, device=OPT_DEVICE))
                mins = torch.stack(mins_list)
                maxs = torch.stack(maxs_list)

                ref_point = (mins - 0.2 * (maxs - mins + 1e-6)) * weights_t.to(OPT_DEVICE)
                acq_func = qLogNoisyExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=ref_point,
                    X_baseline=train_x.to(OPT_DEVICE),
                    sampler=sampler,
                    objective=mo_obj,
                    prune_baseline=True
                ).to(OPT_DEVICE)
                acq_is_logarithmic = True
            else:
                post_tf = ScalarizedPosteriorTransform(weights=weights_t.to(OPT_DEVICE))
                
                # Fill NaNs with the minimum observed value for each objective
                train_y_filled = train_y.clone()
                for idx in range(train_y.shape[1]):
                    valid_vals = train_y[:, idx][~torch.isnan(train_y[:, idx])]
                    min_val = valid_vals.min().item() if valid_vals.numel() > 0 else 0.0
                    nan_mask = torch.isnan(train_y_filled[:, idx])
                    train_y_filled[nan_mask, idx] = min_val
                
                best_f = (train_y_filled.to(OPT_DEVICE) * weights_t.to(OPT_DEVICE)).sum(dim=-1).max().item()
                acq_func = qLogExpectedImprovement(
                    model=model,
                    best_f=best_f,
                    sampler=sampler,
                    posterior_transform=post_tf
                ).to(OPT_DEVICE)
                acq_is_logarithmic = True
        else:
            if acq_type == 'EI':
                acq_func = qLogExpectedImprovement(
                    model=model,
                    best_f=train_y.to(OPT_DEVICE).max().item(),
                    sampler=sampler,
                    objective=None
                ).to(OPT_DEVICE)
                acq_is_logarithmic = True
            elif acq_type == 'LCB':
                beta     = exploration_to_beta(exploration)
                acq_func = qUpperConfidenceBound(model, beta=beta, sampler=sampler).to(OPT_DEVICE)
            else:
                acq_func = qLogExpectedImprovement(
                    model=model,
                    best_f=train_y.to(OPT_DEVICE).max().item(),
                    sampler=sampler,
                    objective=None
                ).to(OPT_DEVICE)
                acq_is_logarithmic = True

        if objective_constraints:
            constrained_names = {c['objective'] for c in objective_constraints}
            for objective_name in constrained_names:
                raw_values = pd.to_numeric(
                    df_clean[objective_name], errors='coerce'
                ).values
                valid_mask_np = ~np.isnan(raw_values)
                valid_mask = torch.tensor(
                    valid_mask_np, dtype=torch.bool, device=train_x.device
                )
                raw_x = train_x[valid_mask]
                raw_y = torch.tensor(
                    raw_values[valid_mask_np], dtype=torch.float32
                ).unsqueeze(-1).to(device)
                if raw_x.shape[0] == 0:
                    raise ValueError(
                        f"No valid observations for constrained objective '{objective_name}'."
                    )
                raw_model = build_single_model(raw_x, raw_y)
                raw_mll = ExactMarginalLogLikelihood(
                    raw_model.likelihood, raw_model
                ).to(compute_device)
                try:
                    fit_gpytorch_mll(
                        raw_mll, optimizer_kwargs={"options": {"maxiter": 75}}
                    )
                except Exception as exc:
                    print(
                        f"Warning: Feasibility model fit failed for "
                        f"{objective_name}: {exc}"
                    )
                raw_model.eval()
                feasibility_models[objective_name] = raw_model.to(OPT_DEVICE)

            acq_func = ProbabilityOfFeasibilityAcquisition(
                acq_func,
                feasibility_models,
                objective_constraints,
                logarithmic=acq_is_logarithmic,
            ).to(OPT_DEVICE)

        strategy = tweaks.get('optimization_strategy', 'gradient_based')

        if strategy == 'exhaustive_grid':
            final_levels = []
            for idx, levels in enumerate(feature_levels):
                if levels is None:
                    low, high = bounds[idx]
                    final_levels.append(np.linspace(low, high, 10).tolist())
                else:
                    final_levels.append(levels)

            import math
            n_combos = math.prod(len(l) for l in final_levels)
            if n_combos > 100000:
                raise ValueError(
                    f"Exhaustive grid size ({n_combos:,} combinations) exceeds the safety cap of 100,000. "
                    "To optimize successfully, please reduce the number of steps in your 'regular' features "
                    "or limit the categories in 'categorical' features."
                )

            all_combos = list(itertools.product(*final_levels))
            train_fps  = {tuple(np.round(row, 6)) for row in train_x.cpu().numpy()}
            filtered_combos = [c for c in all_combos if tuple(np.round(c, 6)) not in train_fps]

            if not filtered_combos:
                raise ValueError('All points in the grid scope have already been evaluated.')

            candidates_grid = torch.tensor(filtered_combos, dtype=torch.float32).to(OPT_DEVICE)

            if num_rules or disallowed_rules:
                feasible_indices = []
                for idx, cand in enumerate(filtered_combos):
                    row_dict = {}
                    for col_idx, f in enumerate(features_config):
                        val = cand[col_idx]
                        if f['type'] == 'categorical':
                            choices = get_categorical_choices(f)
                            choice_idx = max(0, min(len(choices) - 1, int(round(val))))
                            val = choices[choice_idx]
                        elif f['type'] == 'discrete':
                            d_vals = sorted([float(x.strip()) for x in f['range'].split(',') if x.strip()])
                            val = min(d_vals, key=lambda x: abs(x - val)) if d_vals else val
                        elif f['type'] == 'regular':
                            matches = re.findall(r"[-+]?\d*\.?\d+", f['range'])
                            if len(matches) >= 3:
                                low, high, steps = sorted([float(matches[0]), float(matches[1])]) + [int(float(matches[2]))]
                                d_vals = np.linspace(low, high, steps).tolist()
                            elif len(matches) == 2:
                                low, high = sorted([float(matches[0]), float(matches[1])])
                                d_vals = np.linspace(low, high, 10).tolist()
                            else:
                                d_vals = []
                            val = min(d_vals, key=lambda x: abs(x - val)) if d_vals else val
                        
                        if isinstance(val, (int, float)):
                            prec = feature_precision.get(f['name'], infer_decimal_precision(None, f.get('range', '')))
                            val = round(float(val), prec)
                        row_dict[f['name']] = val

                    if check_constraints(row_dict, num_rules, disallowed_rules, features_config):
                        feasible_indices.append(idx)

                if not feasible_indices:
                    raise ValueError('No grid points satisfy the provided constraints.')

                filtered_combos = [filtered_combos[idx] for idx in feasible_indices]
                candidates_grid = torch.tensor(filtered_combos, dtype=torch.float32).to(OPT_DEVICE)

            if not objective_constraints and not is_mo and acq_type == 'EI':
                try:
                    from botorch.acquisition.analytic import LogExpectedImprovement
                    acq_func_grid = LogExpectedImprovement(model=model, best_f=train_y.to(OPT_DEVICE).max().item()).to(OPT_DEVICE)
                except ImportError:
                    from botorch.acquisition.analytic import ExpectedImprovement
                    acq_func_grid = ExpectedImprovement(model=model, best_f=train_y.to(OPT_DEVICE).max().item()).to(OPT_DEVICE)
            else:
                acq_func_grid = acq_func

            candidates, _ = optimize_acqf_discrete(
                acq_function=acq_func_grid,
                q=batch_size,
                choices=candidates_grid,
                unique=True
            )
            candidates = candidates.cpu()

        else:
            p_constraints = []
            if num_rules:
                indices = {f['name']: idx for idx, f in enumerate(features_config)}
                sorted_names = sorted(indices.keys(), key=len, reverse=True)

                def _eval_tensor_ast(node, X, ph_to_idx):
                    if isinstance(node, ast.Expression):
                        return _eval_tensor_ast(node.body, X, ph_to_idx)
                    elif isinstance(node, ast.Constant):
                        return float(node.value)
                    elif isinstance(node, ast.Num):
                        return float(node.n)
                    elif isinstance(node, ast.Name):
                        if node.id in ph_to_idx:
                            return X[..., ph_to_idx[node.id]]
                        raise ValueError(f"Unknown variable in constraint: {node.id}")
                    elif isinstance(node, ast.UnaryOp):
                        opnd = _eval_tensor_ast(node.operand, X, ph_to_idx)
                        if isinstance(node.op, ast.USub):
                            return -opnd
                        elif isinstance(node.op, ast.UAdd):
                            return opnd
                        raise ValueError(f"Unsupported unary operator: {type(node.op)}")
                    elif isinstance(node, ast.BinOp):
                        l = _eval_tensor_ast(node.left, X, ph_to_idx)
                        r = _eval_tensor_ast(node.right, X, ph_to_idx)
                        if isinstance(node.op, ast.Add):
                            return l + r
                        elif isinstance(node.op, ast.Sub):
                            return l - r
                        elif isinstance(node.op, ast.Mult):
                            return l * r
                        elif isinstance(node.op, ast.Div):
                            return l / r
                        elif isinstance(node.op, ast.Pow):
                            return l ** r
                        raise ValueError(f"Unsupported binary operator: {type(node.op)}")
                    raise ValueError(f"Unsupported constraint node: {type(node)}")

                for expr in num_rules:
                    try:
                        clean_expr = expr.strip()
                        clean_expr = re.sub(r'(?<![<>=!])=(?!=)', '==', clean_expr)
                        op_found = None
                        for op in ['<=', '>=', '==', '<', '>']:
                            if op in clean_expr:
                                op_found = op
                                break
                        if op_found:
                            parts = clean_expr.split(op_found)
                            lhs_str = parts[0].strip()
                            rhs_str = parts[1].strip()
                            
                            ph_to_idx = {}
                            for i, name in enumerate(sorted_names):
                                ph = f"_ph_{i}"
                                ph_to_idx[ph] = indices[name]
                                pattern = rf'(?<![\w]){re.escape(name)}(?![\w])'
                                lhs_str = re.sub(pattern, ph, lhs_str)
                                rhs_str = re.sub(pattern, ph, rhs_str)
                            
                            if op_found in ('<=', '<'):
                                vec_expr = f"({rhs_str}) - ({lhs_str})"
                                is_inequality = True
                            elif op_found == '==':
                                vec_expr = f"({lhs_str}) - ({rhs_str})"
                                is_inequality = False
                            else:
                                vec_expr = f"({lhs_str}) - ({rhs_str})"
                                is_inequality = True
                                
                            parsed_tree = ast.parse(vec_expr, mode='eval')
                            def constraint_func(X, tree=parsed_tree, mapping=ph_to_idx):
                                return _eval_tensor_ast(tree, X, mapping)
                            p_constraints.append((constraint_func, is_inequality))
                    except Exception as ex:
                        print(f"Error compiling continuous constraint '{expr}': {ex}")

            base_restarts    = 8  if is_mo else 32
            base_raw_samples = 128 if is_mo else 512

            batch_initial_conditions = None
            if p_constraints:
                try:
                    batch_initial_conditions = generate_feasible_initial_conditions(
                        bounds_t=bounds_t.to(OPT_DEVICE),
                        q=batch_size,
                        num_restarts=int(base_restarts * intensity),
                        p_constraints=p_constraints
                    )
                except Exception as e:
                    print(f"Error generating feasible initial conditions: {e}")

            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds_t.to(OPT_DEVICE),
                q=batch_size,
                num_restarts=int(base_restarts * intensity),
                raw_samples=int(base_raw_samples * intensity),
                nonlinear_inequality_constraints=p_constraints if p_constraints else None,
                batch_initial_conditions=batch_initial_conditions
            )
            candidates = candidates.cpu()

    # Decode Candidate Tensors
    feat_names = [f['name'] for f in features_config]

    def get_fingerprint(row_dict):
        parts = []
        for name in feat_names:
            val = row_dict[name]
            if isinstance(val, (int, float)):
                prec = feature_precision.get(name, 6)
                parts.append(f"{float(val):.{prec}f}")
            else:
                parts.append(str(val))
        return "|".join(parts)



    existing_fps = set()
    if not df_clean.empty:
        for _, row in df_clean[feat_names].iterrows():
            existing_fps.add(get_fingerprint(row.to_dict()))

    def decode_cand(c_tensor):
        row = {}
        for idx, f in enumerate(features_config):
            val = c_tensor[idx].item()

            if f['type'] == 'discrete':
                d_vals = sorted([float(x.strip()) for x in f['range'].split(',') if x.strip()])
                val    = min(d_vals, key=lambda x: abs(x - val)) if d_vals else val

            elif f['type'] == 'regular':
                matches = re.findall(r"[-+]?\d*\.?\d+", f['range'])
                if len(matches) >= 3:
                    low, high, steps = sorted([float(matches[0]), float(matches[1])]) + [int(float(matches[2]))]
                    d_vals = np.linspace(low, high, steps).tolist()
                elif len(matches) == 2:
                    low, high = sorted([float(matches[0]), float(matches[1])])
                    d_vals = np.linspace(low, high, 10).tolist()
                else:
                    d_vals = []
                val = min(d_vals, key=lambda x: abs(x - val)) if d_vals else val

            elif f['type'] == 'categorical':
                choices = get_categorical_choices(f)
                idx_choice = max(0, min(len(choices) - 1, int(round(val))))
                val     = choices[idx_choice]

            if isinstance(val, (int, float)):
                prec = feature_precision.get(f['name'], infer_decimal_precision(None, f.get('range', '')))
                val = round(float(val), prec)
            row[f['name']] = val
        return row

    suggestions = []
    duplicate_count = 0  # tracks candidates rejected solely due to fingerprint collision
    for cand in candidates:
        suggestion = decode_cand(cand)

        if avoid_reval or num_rules or disallowed_rules:
            attempts = 0
            t_cand   = cand.clone()
            while (get_fingerprint(suggestion) in existing_fps or not check_constraints(suggestion, num_rules, disallowed_rules, features_config)) and attempts < 40:
                attempts += 1
                for idx, f in enumerate(features_config):
                    if f['type'] == 'continuous':
                        span     = bounds[idx][1] - bounds[idx][0]
                        t_cand[idx] = torch.clamp(
                            t_cand[idx] + (torch.rand(1).item() - 0.5) * span * 0.1,
                            bounds[idx][0], bounds[idx][1]
                        )
                    elif f['type'] == 'discrete':
                        d_vals  = sorted([float(x.strip()) for x in f['range'].split(',') if x.strip()])
                        if len(d_vals) > 1:
                            cur_idx   = d_vals.index(suggestion[f['name']])
                            t_cand[idx] = d_vals[(cur_idx + attempts) % len(d_vals)]
                    elif f['type'] == 'regular':
                        matches = re.findall(r"[-+]?\d*\.?\d+", f['range'])
                        if len(matches) >= 3:
                            low, high, steps = sorted([float(matches[0]), float(matches[1])]) + [int(float(matches[2]))]
                            d_vals = np.linspace(low, high, steps).tolist()
                        elif len(matches) == 2:
                            low, high = sorted([float(matches[0]), float(matches[1])])
                            d_vals = np.linspace(low, high, 10).tolist()
                        else:
                            d_vals = []
                        if len(d_vals) > 1:
                            cur_idx   = d_vals.index(suggestion[f['name']])
                            t_cand[idx] = d_vals[(cur_idx + attempts) % len(d_vals)]
                    elif f['type'] == 'categorical':
                        choices = get_categorical_choices(f)
                        if len(choices) > 1:
                            cur_idx   = choices.index(suggestion[f['name']])
                            t_cand[idx] = float((cur_idx + attempts) % len(choices))
                suggestion = decode_cand(t_cand)

        # If avoid_reval is on and the candidate is still a known fingerprint after
        # all retry attempts, count it as a duplicate and skip (do not append).
        if avoid_reval and get_fingerprint(suggestion) in existing_fps:
            duplicate_count += 1
            continue

        suggestions.append(suggestion)
        existing_fps.add(get_fingerprint(suggestion))

    if num_rules or disallowed_rules:
        suggestions = [s for s in suggestions if check_constraints(s, num_rules, disallowed_rules, features_config)]

    # Sobol gap-filling
    if len(suggestions) < batch_size:
        gap = batch_size - len(suggestions)
        try:
            n_pool = max(gap * 200, 2000)
            pool = draw_sobol_samples(bounds=bounds_t.to(OPT_DEVICE), n=1, q=n_pool).squeeze(0)
            for pt in pool:
                if len(suggestions) >= batch_size:
                    break
                decoded = decode_cand(pt.cpu())
                fp = get_fingerprint(decoded)
                if fp in existing_fps:
                    continue
                if not check_constraints(decoded, num_rules, disallowed_rules, features_config):
                    continue
                suggestions.append(decoded)
                existing_fps.add(fp)
        except Exception as e:
            print(f"Error in Sobol gap-filling: {e}")

    # Exhaustion flag: if avoid_reval is active and every candidate was a
    # precision-rounded duplicate (not a constraint failure), we will signal
    # exhaustion after computing performance metrics (so the user still gets R² info).
    is_exhausted = avoid_reval and len(suggestions) == 0 and duplicate_count > 0

    if not is_exhausted and not suggestions:
        raise ValueError('No suggestions satisfy the specified constraints.')

    # Cross-Validation
    performance_results = {}
    if not df_clean.empty and train_x.shape[0] >= 5:
        from sklearn.model_selection import KFold
        n_splits = 5
        kf       = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for idx, obj_cfg in enumerate(objectives_config):
            obj_name  = obj_cfg['name']
            y_all_np  = train_y[:, idx:idx+1].cpu().numpy()
            x_all_np  = train_x.cpu().numpy()

            nan_mask = np.isnan(y_all_np.flatten())
            y_all_np_clean = y_all_np[~nan_mask]
            x_all_np_clean = x_all_np[~nan_mask]

            if len(y_all_np_clean) < 5:
                performance_results[obj_name] = {
                    'r2': 0.0,
                    'status': 'Poor'
                }
                continue

            all_preds = []
            all_true  = []

            for train_idx, test_idx in kf.split(x_all_np_clean):
                x_tr = torch.tensor(x_all_np_clean[train_idx], dtype=torch.float32).to(device)
                x_te = torch.tensor(x_all_np_clean[test_idx],  dtype=torch.float32).to(device)
                y_tr = torch.tensor(y_all_np_clean[train_idx], dtype=torch.float32).to(device)

                try:
                    m_tmp   = build_single_model(x_tr, y_tr)
                    mll_tmp = ExactMarginalLogLikelihood(m_tmp.likelihood, m_tmp).to(compute_device)
                    fit_gpytorch_mll(mll_tmp, optimizer_kwargs={"options": {"maxiter": 25}})
                    m_tmp.eval()
                    with torch.no_grad():
                        post  = m_tmp.posterior(x_te.to(compute_device))
                        preds = post.mean.cpu().numpy().flatten()
                        all_preds.extend(preds)
                        all_true.extend(y_all_np_clean[test_idx].flatten())
                except Exception as ex:
                    print(f"Error in CV for {obj_name}: {ex}")
                    all_preds.extend([np.mean(y_all_np_clean[train_idx])] * len(test_idx))
                    all_true.extend(y_all_np_clean[test_idx].flatten())

            all_preds = np.array(all_preds)
            all_true  = np.array(all_true)
            ss_res    = np.sum((all_true - all_preds) ** 2)
            ss_tot    = np.sum((all_true - np.mean(all_true)) ** 2)
            r2        = 1 - (ss_res / (ss_tot + 1e-9))
            avg_r2    = max(0, r2)

            performance_results[obj_name] = {
                'r2':     round(float(avg_r2), 3),
                'status': 'Good' if avg_r2 > 0.7 else ('Limited' if avg_r2 >= 0.4 else 'Poor')
            }

        if train_y.shape[1] > 1:
            weights_np = weights_t.cpu().numpy()
            
            # Fill NaNs in train_y for global success metric calculation
            train_y_filled_np = train_y.cpu().numpy()
            for idx in range(train_y_filled_np.shape[1]):
                valid_vals = train_y_filled_np[:, idx][~np.isnan(train_y_filled_np[:, idx])]
                min_val = valid_vals.min() if len(valid_vals) > 0 else 0.0
                train_y_filled_np[np.isnan(train_y_filled_np[:, idx]), idx] = min_val
                
            y_min, y_max = train_y_filled_np.min(axis=0), train_y_filled_np.max(axis=0)
            y_norm     = (train_y_filled_np - y_min) / (y_max - y_min + 1e-9)
            global_y   = np.average(y_norm, axis=1, weights=weights_np)

            all_preds_g = []
            all_true_g  = []
            for train_idx, test_idx in kf.split(x_all_np):
                x_tr   = torch.tensor(x_all_np[train_idx], dtype=torch.float32).to(device)
                x_te   = torch.tensor(x_all_np[test_idx],  dtype=torch.float32).to(device)
                y_tr_g = torch.tensor(global_y[train_idx], dtype=torch.float32).unsqueeze(-1).to(device)

                try:
                    m_tmp   = build_single_model(x_tr, y_tr_g)
                    mll_tmp = ExactMarginalLogLikelihood(m_tmp.likelihood, m_tmp).to(compute_device)
                    fit_gpytorch_mll(mll_tmp, optimizer_kwargs={"options": {"maxiter": 25}})
                    m_tmp.eval()
                    with torch.no_grad():
                        post  = m_tmp.posterior(x_te.to(compute_device))
                        preds = post.mean.cpu().numpy().flatten()
                        all_preds_g.extend(preds)
                        all_true_g.extend(global_y[test_idx].flatten())
                except Exception:
                    all_preds_g.extend([np.mean(global_y[train_idx])] * len(test_idx))
                    all_true_g.extend(global_y[test_idx].flatten())

            ss_res_g = np.sum((np.array(all_true_g) - np.array(all_preds_g)) ** 2)
            ss_tot_g = np.sum((np.array(all_true_g) - np.mean(all_true_g)) ** 2)
            r2_g     = 1 - (ss_res_g / (ss_tot_g + 1e-9))
            avg_r2_g = max(0, r2_g)

            performance_results['global_success'] = {
                'r2':     round(float(avg_r2_g), 3),
                'status': 'Good' if avg_r2_g > 0.7 else ('Limited' if avg_r2_g >= 0.4 else 'Poor')
            }

    constraint_status = {
        'constrained': bool(objective_constraints),
        'all_likely_feasible': True,
        'candidates': [],
    }
    if objective_constraints and df_clean.empty:
        constraint_status.update({
            'cold_start': True,
            'all_likely_feasible': None,
            'message': (
                'Cold Start: objective constraints will be evaluated once the '
                'first experimental observation is recorded.'
            ),
        })
    elif objective_constraints and suggestions:
        encoded_suggestions = []
        for suggestion in suggestions:
            encoded = []
            for feature in features_config:
                value = suggestion[feature['name']]
                if feature['type'] == 'categorical':
                    choices = get_categorical_choices(feature)
                    value = choices.index(str(value))
                encoded.append(float(value))
            encoded_suggestions.append(encoded)
        report_tensor = torch.tensor(
            encoded_suggestions, dtype=torch.float32, device=OPT_DEVICE
        )
        constraint_status = candidate_feasibility_report(
            feasibility_models,
            objective_constraints,
            report_tensor,
        )
        if not constraint_status['all_likely_feasible']:
            best_probability = max(
                (item['probability'] for item in constraint_status['candidates']),
                default=0.0,
            )
            constraint_status['message'] = (
                '⚠️ Feasibility Warning: at least one recommended candidate '
                'does not confidently satisfy every objective constraint '
                f'(best predicted feasibility: {best_probability:.1%}).'
            )

    # Return exhaustion sentinel after CV so performance table is still populated
    if is_exhausted:
        return None, performance_results, constraint_status

    return suggestions, performance_results, constraint_status


def run_bo_estimation(df, input_values, features_config, objectives_config, tweaks, device, compute_device):
    # Filter the dataframe to drop incomplete rows
    df_clean = clean_dataframe(df, features_config, objectives_config)
    if df_clean.empty:
        raise ValueError('No valid data points left after filtering missing features/objectives.')

    # Build Feature Matrix (X)
    train_x_list      = []
    bounds            = []
    categorical_features = []

    for idx, f in enumerate(features_config):
        name    = f['name']
        f_range = f['range']

        if f['type'] == 'continuous':
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            low, high = sorted([float(matches[0]), float(matches[1])]) if len(matches) >= 2 else (0.0, 1.0)
            if low == high: high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(0).values
            train_x_list.append(feat_vals)
            bounds.append([low, high])

        elif f['type'] in ['discrete', 'regular']:
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            if f['type'] == 'discrete':
                values = sorted([float(x.strip()) for x in f_range.replace('[', '').replace(']', '').split(',') if x.strip()])
            else:
                if len(matches) >= 3:
                    low, high, steps = sorted([float(matches[0]), float(matches[1])]) + [int(float(matches[2]))]
                    values = np.linspace(low, high, steps).tolist()
                else:
                    values = [0.0]
            low, high = values[0], values[-1]
            if low == high: high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(values[0]).values
            train_x_list.append(feat_vals)
            bounds.append([low, high])

        elif f['type'] == 'categorical':
            encoded, choices = encode_categorical_series(df_clean[name], f)
            train_x_list.append(encoded)
            low, high = 0, len(choices) - 1
            if low == high: high = low + 0.1
            bounds.append([low, high])
            categorical_features.append(idx)

    train_x  = torch.tensor(np.stack(train_x_list, axis=1), dtype=torch.float32).to(device)
    bounds_t = torch.tensor(bounds, dtype=torch.float32).T.to(device)

    # Build Objective Matrix (Y) in optimization-transformed space
    train_y_list = []
    weights_list = []
    for obj in objectives_config:
        weights_list.append(float(obj.get('importance', 1)))
        vals = pd.to_numeric(df_clean[obj['name']], errors='coerce').values
        if   obj['type'] == 'minimize': train_y_list.append(-vals)
        elif obj['type'] == 'maximize': train_y_list.append(vals)
        elif obj['type'] == 'target':
            target   = float(obj.get('target', 0))
            valid_vals = vals[~np.isnan(vals)]
            variance = np.var(valid_vals) if len(valid_vals) > 0 and np.var(valid_vals) > 0 else 1.0
            train_y_list.append(-((vals - target) ** 2) / variance)

    train_y  = torch.tensor(np.stack(train_y_list, axis=1), dtype=torch.float32).to(device)

    kernel_name = tweaks.get('kernel', 'matern52')

    def get_kernel(k_name):
        if k_name == 'rbf':      return ScaleKernel(RBFKernel()).to(device)
        if k_name == 'matern32': return ScaleKernel(MaternKernel(nu=1.5)).to(device)
        if k_name == 'matern12': return ScaleKernel(MaternKernel(nu=0.5)).to(device)
        return ScaleKernel(MaternKernel(nu=2.5)).to(device)

    def build_single_model(tx, ty_slice):
        all_dims  = list(range(tx.shape[1]))
        cont_dims = [idx for idx in all_dims if idx not in categorical_features]
        input_tf  = (
            Normalize(d=tx.shape[1], bounds=bounds_t.to(compute_device)[:, cont_dims], indices=cont_dims).to(compute_device)
            if cont_dims else None
        )
        if categorical_features and len(cont_dims) > 0:
            return MixedSingleTaskGP(tx.to(compute_device), ty_slice.to(compute_device),
                                      cat_dims=categorical_features, outcome_transform=Standardize(m=1),
                                      input_transform=input_tf).to(compute_device)
        return SingleTaskGP(tx.to(compute_device), ty_slice.to(compute_device),
                             covar_module=get_kernel(kernel_name).to(compute_device),
                             outcome_transform=Standardize(m=1), input_transform=input_tf).to(compute_device)

    if train_y.shape[1] > 1:
        models = []
        for idx in range(train_y.shape[1]):
            y_slice = train_y[:, idx:idx+1]
            non_nan_mask = ~torch.isnan(y_slice.squeeze(-1))
            tx_i = train_x[non_nan_mask]
            ty_i = y_slice[non_nan_mask]
            if tx_i.shape[0] == 0:
                raise ValueError(f"No valid observations for objective '{objectives_config[idx]['name']}' after filtering missing values.")
            models.append(build_single_model(tx_i, ty_i))
        model   = ModelListGP(*models).to(compute_device)
        mll     = SumMarginalLogLikelihood(model.likelihood, model).to(compute_device)
    else:
        model   = build_single_model(train_x, train_y)
        mll     = ExactMarginalLogLikelihood(model.likelihood, model).to(compute_device)

    try:
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 100}})
    except Exception as e:
        print(f"Warning: Main model fit failed in estimate_bo: {e}")
    model.eval()

    # Encode Query Point
    input_point_list = []
    for idx, f in enumerate(features_config):
        val = input_values[f['name']]
        if f['type'] == 'categorical':
            choices = get_categorical_choices(f)
            if str(val) not in choices:
                raise ValueError(f"Feature '{f['name']}' has unknown categorical value {val!r}.")
            val = choices.index(str(val))
        input_point_list.append(float(val))

    test_x = torch.tensor([input_point_list], dtype=torch.float32).to(compute_device)

    # Fit RAW (un-transformed) models to predict values in original units
    raw_predictions = {}
    raw_std_errors  = {}
    for idx, obj in enumerate(objectives_config):
        name  = obj['name']
        prec  = infer_decimal_precision(df_clean[name])
        y_raw = pd.to_numeric(df_clean[name], errors='coerce').values
        
        # Filter NaNs for raw model fitting
        non_nan_mask = ~np.isnan(y_raw)
        train_x_i = train_x[non_nan_mask]
        y_raw_t = torch.tensor(y_raw[non_nan_mask], dtype=torch.float32).unsqueeze(-1).to(device)

        if train_x_i.shape[0] == 0:
            raise ValueError(f"No valid observations for objective '{name}' after filtering missing values.")

        m_raw   = build_single_model(train_x_i, y_raw_t)
        mll_raw = ExactMarginalLogLikelihood(m_raw.likelihood, m_raw).to(compute_device)
        try:
            fit_gpytorch_mll(mll_raw, optimizer_kwargs={"options": {"maxiter": 100}})
        except Exception as e:
            print(f"Warning: Raw model fit failed in estimate_bo: {e}")
        m_raw.eval()

        with torch.no_grad():
            post_raw   = m_raw.posterior(test_x)
            pred_raw   = post_raw.mean.item()
            std_raw    = post_raw.stddev.item()
            raw_predictions[name] = round(float(pred_raw), prec)
            raw_std_errors[name]  = round(float(std_raw), prec + 2)

    # Compute Per-Objective and Global Success Scores (0-100%)
    success_scores    = {}
    processed_success = []

    for idx, obj in enumerate(objectives_config):
        name        = obj['name']
        val         = raw_predictions[name]
        train_vals  = pd.to_numeric(df_clean[name], errors='coerce').values
        
        # Clean training values for scoring bounds
        valid_train = train_vals[~np.isnan(train_vals)]
        if len(valid_train) == 0:
            success_scores[name] = 0.0
            continue

        if   obj['type'] == 'maximize': s = val;       t_s = valid_train
        elif obj['type'] == 'minimize': s = -val;      t_s = -valid_train
        else:
            target = float(obj.get('target', 0))
            s = -abs(val - target)
            t_s = -np.abs(valid_train - target)

        s_min, s_max = t_s.min(), t_s.max()
        norm_s = (s - s_min) / (s_max - s_min + 1e-9) if s_max > s_min else 1.0
        success_scores[name] = round(float(np.clip(norm_s, 0, 1) * 100), 1)
        processed_success.append(np.clip(norm_s, 0, 1))

    _, violations = check_objective_constraints(
        raw_predictions, tweaks.get('objective_constraints', [])
    )
    violated_objectives = {item['objective'] for item in violations}
    for objective_name in violated_objectives:
        if objective_name in success_scores:
            success_scores[objective_name] = 0.0

    weighted_scores = [
        float(success_scores.get(obj['name'], 0.0)) / 100.0
        for obj in objectives_config
    ]
    global_score = (
        np.average(weighted_scores, weights=weights_list)
        if weighted_scores else 0.0
    )

    return (
        raw_predictions,
        raw_std_errors,
        success_scores,
        round(float(global_score * 100), 1),
        violations,
    )


def run_bo_surface(
    df,
    features_config,
    objectives_config,
    tweaks,
    x_feat,
    y_feat,
    z_axis=None,
    color_obj=None,
    fixed_values=None,
    resolution=25,
    device=torch.device('cpu'),
    compute_device=torch.device('cpu'),
    cancel_event=None,
    progress_callback=None,
):
    """
    Computes a 2D prediction grid (response surface) using the fitted GP surrogate
    for contour and 3D surface visualizations.
    """
    def check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise SurfaceCalculationCancelled('Response-surface calculation cancelled by user.')

    def report(phase, percent):
        if progress_callback is not None:
            progress_callback(phase, percent)

    check_cancelled()
    report('Preparing response-surface data', 10)
    df_clean = clean_dataframe(df, features_config, objectives_config)
    if df_clean.empty:
        raise ValueError('No valid data points left after filtering missing features/objectives.')

    feat_names = [f['name'] for f in features_config if f['name'] in df_clean.columns]
    if len(feat_names) < 2:
        raise ValueError('At least two included numerical features are required for response surface.')

    if not x_feat or x_feat not in feat_names:
        x_feat = feat_names[0]
    if not y_feat or y_feat not in feat_names or y_feat == x_feat:
        remaining = [f for f in feat_names if f != x_feat]
        y_feat = remaining[0] if remaining else feat_names[1]

    obj_names = [o['name'] for o in objectives_config if o['name'] in df_clean.columns]
    if not obj_names:
        raise ValueError('At least one included objective is required for response surface.')

    if not color_obj or color_obj not in obj_names:
        color_obj = obj_names[0]
    # z_axis can be an objective OR a numerical feature (not x_feat or y_feat)
    num_feat_names = [
        f['name'] for f in features_config
        if f['name'] in df_clean.columns and f.get('type', 'continuous') != 'categorical'
    ]
    valid_z_options = obj_names + [f for f in num_feat_names if f != x_feat and f != y_feat]
    if not z_axis or z_axis not in valid_z_options:
        z_axis = color_obj

    if fixed_values is None:
        fixed_values = {}

    # Build Feature Matrix (X) and bounds
    train_x_list = []
    bounds = []
    categorical_features = []
    feat_ranges = {}

    for idx, f in enumerate(features_config):
        name = f['name']
        f_range = f.get('range', '')

        if f['type'] == 'continuous':
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            low, high = sorted([float(matches[0]), float(matches[1])]) if len(matches) >= 2 else (0.0, 1.0)
            if low == high:
                high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(0).values
            train_x_list.append(feat_vals)
            bounds.append([low, high])
            feat_ranges[name] = (low, high, 'continuous', None)

        elif f['type'] in ['discrete', 'regular']:
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            if f['type'] == 'discrete':
                clean_range = f_range.replace('[', '').replace(']', '')
                values = sorted([float(x.strip()) for x in clean_range.split(',') if x.strip()])
                if not values:
                    values = [0.0]
            else:
                if len(matches) >= 3:
                    low, high, steps = sorted([float(matches[0]), float(matches[1])]) + [int(float(matches[2]))]
                    values = np.linspace(low, high, steps).tolist()
                elif len(matches) == 2:
                    low, high = sorted([float(matches[0]), float(matches[1])])
                    values = np.linspace(low, high, 10).tolist()
                else:
                    values = [0.0]
            low, high = values[0], values[-1]
            if low == high:
                high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(values[0]).values
            train_x_list.append(feat_vals)
            bounds.append([low, high])
            feat_ranges[name] = (low, high, f['type'], values)

        elif f['type'] == 'categorical':
            encoded, choices = encode_categorical_series(df_clean[name], f)
            train_x_list.append(encoded)
            low, high = 0, len(choices) - 1
            if low == high:
                high = low + 0.1
            bounds.append([low, high])
            categorical_features.append(idx)
            feat_ranges[name] = (0, len(choices) - 1, 'categorical', choices)

    train_x = torch.tensor(np.stack(train_x_list, axis=1), dtype=torch.float32).to(device)
    bounds_t = torch.tensor(bounds, dtype=torch.float32).T.to(device)

    res = max(5, min(int(resolution), 50))

    x_info = feat_ranges.get(x_feat, (0.0, 1.0, 'continuous', None))
    x_vals = np.linspace(x_info[0], x_info[1], res)

    y_info = feat_ranges.get(y_feat, (0.0, 1.0, 'continuous', None))
    y_vals = np.linspace(y_info[0], y_info[1], res)

    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
    x_flat = X_mesh.ravel()
    y_flat = Y_mesh.ravel()
    n_points = len(x_flat)

    eval_matrix = np.zeros((n_points, len(features_config)), dtype=np.float32)
    for idx, f in enumerate(features_config):
        fname = f['name']
        if fname == x_feat:
            eval_matrix[:, idx] = x_flat
        elif fname == y_feat:
            eval_matrix[:, idx] = y_flat
        else:
            raw_fixed = fixed_values.get(fname, None)
            f_inf = feat_ranges.get(fname, (0.0, 1.0, 'continuous', None))
            if raw_fixed is None:
                if f_inf[2] == 'categorical' and f_inf[3]:
                    val = 0.0
                else:
                    val = (f_inf[0] + f_inf[1]) / 2.0
            else:
                if f_inf[2] == 'categorical' and f_inf[3]:
                    val = float(f_inf[3].index(str(raw_fixed))) if str(raw_fixed) in f_inf[3] else 0.0
                else:
                    try:
                        val = float(raw_fixed)
                    except Exception:
                        val = (f_inf[0] + f_inf[1]) / 2.0
            eval_matrix[:, idx] = val

    eval_x = torch.tensor(eval_matrix, dtype=torch.float32).to(compute_device)
    check_cancelled()
    report('Building response-surface grid', 25)

    kernel_name = tweaks.get('kernel', 'matern52')
    def get_kernel(k_name):
        if k_name == 'rbf':      return ScaleKernel(RBFKernel()).to(device)
        if k_name == 'matern32': return ScaleKernel(MaternKernel(nu=1.5)).to(device)
        if k_name == 'matern12': return ScaleKernel(MaternKernel(nu=0.5)).to(device)
        return ScaleKernel(MaternKernel(nu=2.5)).to(device)

    def build_single_model(tx, ty_slice):
        all_dims = list(range(tx.shape[1]))
        cont_dims = [i for i in all_dims if i not in categorical_features]
        input_tf = (
            Normalize(d=tx.shape[1], bounds=bounds_t.to(compute_device)[:, cont_dims], indices=cont_dims).to(compute_device)
            if cont_dims else None
        )
        if categorical_features and len(cont_dims) > 0:
            return MixedSingleTaskGP(tx.to(compute_device), ty_slice.to(compute_device),
                                     cat_dims=categorical_features, outcome_transform=Standardize(m=1),
                                     input_transform=input_tf).to(compute_device)
        return SingleTaskGP(tx.to(compute_device), ty_slice.to(compute_device),
                            covar_module=get_kernel(kernel_name).to(compute_device),
                            outcome_transform=Standardize(m=1), input_transform=input_tf).to(compute_device)

    obj_dict = {o['name']: o for o in objectives_config}

    def fit_and_predict_obj(obj_name, position, total):
        check_cancelled()
        report(f'Fitting surrogate for {obj_name}', 35 + int(45 * (position / max(total, 1))))
        prec = infer_decimal_precision(df_clean[obj_name])
        y_raw = pd.to_numeric(df_clean[obj_name], errors='coerce').values
        non_nan_mask = ~np.isnan(y_raw)
        tx_i = train_x[non_nan_mask]
        y_t = torch.tensor(y_raw[non_nan_mask], dtype=torch.float32).unsqueeze(-1).to(device)
        if tx_i.shape[0] == 0:
            return np.zeros((res, res), dtype=float)
        m = build_single_model(tx_i, y_t)
        mll = ExactMarginalLogLikelihood(m.likelihood, m).to(compute_device)
        try:
            fit_gpytorch_mll(
                mll,
                optimizer_kwargs={
                    "options": {"maxiter": 35},
                    "callback": lambda *_: check_cancelled(),
                },
            )
        except SurfaceCalculationCancelled:
            raise
        except Exception:
            pass
        check_cancelled()
        m.eval()
        with torch.no_grad():
            post = m.posterior(eval_x)
            pred = post.mean.squeeze(-1).cpu().numpy()
        pred_grid = np.round(pred.reshape(res, res), prec)
        return pred_grid

    predicted_objectives = [color_obj]
    if z_axis in obj_dict and z_axis != color_obj:
        predicted_objectives.append(z_axis)

    color_grid = fit_and_predict_obj(
        color_obj, 0, len(predicted_objectives)
    ).tolist()

    if z_axis in obj_dict:
        # z_axis is an objective — predict with GP
        if z_axis == color_obj:
            z_grid = color_grid
        else:
            z_grid = fit_and_predict_obj(
                z_axis, 1, len(predicted_objectives)
            ).tolist()
    else:
        # z_axis is a numerical feature — use the actual feature values from the eval grid
        feat_idx_map = {f['name']: idx for idx, f in enumerate(features_config)}
        if z_axis in feat_idx_map:
            col_idx = feat_idx_map[z_axis]
            z_flat = eval_matrix[:, col_idx]
            prec_z = infer_decimal_precision(df_clean[z_axis])
            z_grid = np.round(z_flat.reshape(res, res), prec_z).tolist()
        else:
            z_grid = color_grid

    x_col_raw = df_clean[x_feat].tolist()
    y_col_raw = df_clean[y_feat].tolist()
    color_col_raw = df_clean[color_obj].tolist() if color_obj in df_clean.columns else []
    z_col_raw = df_clean[z_axis].tolist() if z_axis and z_axis in df_clean.columns else color_col_raw

    # Determine remaining features and their target slice values (nearest for numeric, exact for categorical)
    remaining_feats = [f for f in features_config if f['name'] != x_feat and f['name'] != y_feat]
    slice_targets = {}

    for f in remaining_feats:
        fname = f['name']
        ftype = f.get('type', 'continuous')
        raw_fixed = fixed_values.get(fname, None)
        f_inf = feat_ranges.get(fname, (0.0, 1.0, 'continuous', None))

        if ftype == 'categorical':
            choices = f_inf[3] or []
            selected_cat = str(raw_fixed) if raw_fixed is not None and str(raw_fixed) in choices else (choices[0] if choices else '')
            slice_targets[fname] = {
                "type": "categorical",
                "target_val": selected_cat,
                "label": selected_cat
            }
        else:
            try:
                selected_num = float(raw_fixed) if raw_fixed is not None else (f_inf[0] + f_inf[1]) / 2.0
            except Exception:
                selected_num = (f_inf[0] + f_inf[1]) / 2.0
            
            # Find nearest unique value in the actual dataset
            col_vals = pd.to_numeric(df_clean[fname], errors='coerce').dropna().values
            if len(col_vals) > 0:
                unique_vals = np.unique(col_vals)
                nearest_idx = np.argmin(np.abs(unique_vals - selected_num))
                nearest_val = float(unique_vals[nearest_idx])
            else:
                nearest_val = selected_num
            
            slice_targets[fname] = {
                "type": "numerical",
                "target_val": nearest_val,
                "selected_val": selected_num,
                "label": f"{nearest_val:g}"
            }

    report('Classifying experiments for the selected slice', 85)
    experiments = []
    for i in range(len(df_clean)):
        if i % 256 == 0:
            check_cancelled()
        try:
            exp_x = float(x_col_raw[i]) if pd.notna(x_col_raw[i]) else None
            exp_y = float(y_col_raw[i]) if pd.notna(y_col_raw[i]) else None
            exp_c = float(color_col_raw[i]) if i < len(color_col_raw) and pd.notna(color_col_raw[i]) else None
            exp_z = float(z_col_raw[i]) if i < len(z_col_raw) and pd.notna(z_col_raw[i]) else exp_c

            if exp_x is not None and exp_y is not None:
                # Check if this row matches slice conditions
                is_slice_match = True
                row_remaining_info = []

                for f in remaining_feats:
                    fname = f['name']
                    target_info = slice_targets[fname]
                    row_val_raw = df_clean.iloc[i][fname]

                    if target_info['type'] == 'categorical':
                        row_val_str = str(row_val_raw)
                        row_remaining_info.append(f"{fname}={row_val_str}")
                        if row_val_str != target_info['target_val']:
                            is_slice_match = False
                    else:
                        try:
                            row_val_num = float(row_val_raw)
                            row_remaining_info.append(f"{fname}={row_val_num:g}")
                            if abs(row_val_num - target_info['target_val']) > 1e-6:
                                is_slice_match = False
                        except Exception:
                            is_slice_match = False
                            row_remaining_info.append(f"{fname}={row_val_raw}")

                experiments.append({
                    "exp_id": i + 1,
                    "x": exp_x,
                    "y": exp_y,
                    "z": exp_z,
                    "color": exp_c,
                    "in_slice": is_slice_match,
                    "conditions": ", ".join(row_remaining_info) if row_remaining_info else ""
                })
        except Exception:
            pass

    check_cancelled()
    report('Finalizing response surface', 95)
    prec_x = infer_decimal_precision(df_clean[x_feat])
    prec_y = infer_decimal_precision(df_clean[y_feat])

    return {
        "x": [round(float(v), prec_x) for v in x_vals],
        "y": [round(float(v), prec_y) for v in y_vals],
        "z": z_grid,
        "color": color_grid,
        "x_feat": x_feat,
        "y_feat": y_feat,
        "z_axis": z_axis or color_obj,
        "color_obj": color_obj,
        "experiments": experiments,
        "slice_targets": {k: v['label'] for k, v in slice_targets.items()}
    }
