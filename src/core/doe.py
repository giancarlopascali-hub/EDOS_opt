"""Design-of-experiments generation and quality assessment."""

from __future__ import annotations

import itertools
import math
import random
import re

import numpy as np
import pydoe as doe
from scipy.linalg import qr

from src.core.bo import infer_decimal_precision
from src.core.constraints import check_constraints


SUPPORTED_MODELS = {
    "bbdesign",
    "ccdesign",
    "fracfact",
    "def_screening",
    "lhs",
    "doptimal",
}

MODEL_LABELS = {
    "bbdesign": "Box-Behnken",
    "ccdesign": "Central Composite",
    "fracfact": "Fractional Factorial",
    "def_screening": "Definitive Screening",
    "lhs": "Latin Hypercube",
    "doptimal": "D-Optimal",
}


def _as_bool(value) -> bool:
    """Parse request booleans without treating the string ``"false"`` as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_features(features_config):
    valid_features = []
    for feature in features_config:
        name = feature.get("name", "Unnamed")
        feature_type = feature.get("type", "continuous")
        range_text = feature.get("range", "")
        if not range_text:
            continue

        if feature_type == "categorical":
            choices = [item.strip() for item in range_text.split(",") if item.strip()]
            if choices:
                valid_features.append(
                    {
                        "name": name,
                        "choices": choices,
                        "type": feature_type,
                        "range": range_text,
                    }
                )
            continue

        numbers = [float(match) for match in re.findall(r"[-+]?\d*\.?\d+", range_text)]
        if feature_type == "discrete":
            levels = sorted(set(numbers))
            if len(levels) >= 2:
                valid_features.append(
                    {
                        "name": name,
                        "low": levels[0],
                        "high": levels[-1],
                        "levels": levels,
                        "type": feature_type,
                        "range": range_text,
                    }
                )
        elif feature_type == "regular" and len(numbers) >= 2:
            low, high = min(numbers[0], numbers[1]), max(numbers[0], numbers[1])
            steps = max(2, int(numbers[2])) if len(numbers) >= 3 else 10
            valid_features.append(
                {
                    "name": name,
                    "low": low,
                    "high": high,
                    "levels": np.linspace(low, high, steps).tolist(),
                    "type": feature_type,
                    "range": range_text,
                }
            )
        elif len(numbers) >= 2:
            low, high = min(numbers[0], numbers[1]), max(numbers[0], numbers[1])
            if high > low:
                valid_features.append(
                    {
                        "name": name,
                        "low": low,
                        "high": high,
                        "type": feature_type,
                        "range": range_text,
                    }
                )
    return valid_features


def _move_centers_to_end(design):
    """Place center rows last while preserving order within both partitions."""
    is_center = np.all(np.abs(design) < 1e-9, axis=1)
    centers = design[is_center]
    ordered = np.vstack([design[~is_center], centers]) if len(centers) else design.copy()
    return ordered, int(is_center.sum())


def _feature_codes(feature):
    if feature["type"] == "categorical":
        count = len(feature["choices"])
    elif feature["type"] in {"regular", "discrete"}:
        count = len(feature["levels"])
    else:
        return None
    return np.linspace(-1.0, 1.0, count) if count > 1 else np.array([0.0])


def _decode_row(row, features, model_type):
    entry = {}
    for column, feature in enumerate(features):
        coded_value = float(row[column])
        if feature["type"] == "categorical":
            codes = _feature_codes(feature)
            index = int(np.argmin(np.abs(codes - coded_value)))
            entry[feature["name"]] = feature["choices"][index]
            continue

        value = feature["low"] + (coded_value + 1.0) * 0.5 * (
            feature["high"] - feature["low"]
        )
        if feature["type"] in {"regular", "discrete"}:
            value = min(feature["levels"], key=lambda candidate: abs(candidate - value))

        precision = infer_decimal_precision(None, feature.get("range", ""))
        # Space-filling and optimal designs need enough precision to avoid
        # collapsing distinct continuous candidate points during decoding.
        if model_type in {"lhs", "doptimal"} and feature["type"] == "continuous":
            precision = max(precision, 6)
        entry[feature["name"]] = round(float(value), precision)
    return entry


def _encode_table(table, features):
    encoded = np.zeros((len(table), len(features)), dtype=float)
    for row_index, entry in enumerate(table):
        for column, feature in enumerate(features):
            value = entry[feature["name"]]
            if feature["type"] == "categorical":
                codes = _feature_codes(feature)
                encoded[row_index, column] = codes[feature["choices"].index(value)]
            else:
                encoded[row_index, column] = (
                    2.0 * (float(value) - feature["low"])
                    / (feature["high"] - feature["low"])
                    - 1.0
                )
    return encoded


def _main_effect_columns(coded_design, features):
    """Return model columns grouped by original factor for valid interactions."""
    groups = []
    for column, feature in enumerate(features):
        values = coded_design[:, column]
        if feature["type"] != "categorical":
            groups.append(values.reshape(-1, 1))
            continue

        choices = feature["choices"]
        codes = _feature_codes(feature)
        indices = np.argmin(np.abs(values[:, None] - codes[None, :]), axis=1)
        # Treatment coding (first level is the baseline) gives m-1 estimable
        # columns and works for categorical factors with any number of levels.
        groups.append(
            np.column_stack([(indices == level).astype(float) for level in range(1, len(choices))])
        )
    return groups


def _build_model_matrix(coded_design, features, model_spec):
    coded_design = np.asarray(coded_design, dtype=float)
    groups = _main_effect_columns(coded_design, features)
    columns = [np.ones((len(coded_design), 1))]
    columns.extend(groups)

    if model_spec in {"interaction", "quadratic", "response_surface"}:
        for first in range(len(groups)):
            for second in range(first + 1, len(groups)):
                columns.append(
                    np.einsum("ni,nj->nij", groups[first], groups[second]).reshape(
                        len(coded_design), -1
                    )
                )

    if model_spec in {"quadratic", "response_surface", "dsd"}:
        for index, feature in enumerate(features):
            if feature["type"] != "categorical":
                columns.append((coded_design[:, index] ** 2).reshape(-1, 1))

    return np.hstack([column for column in columns if column.shape[1] > 0])


def _d_efficiency(model_matrix):
    n_runs, n_parameters = model_matrix.shape
    if n_runs == 0 or np.linalg.matrix_rank(model_matrix) < n_parameters:
        return 0.0
    sign, log_determinant = np.linalg.slogdet(model_matrix.T @ model_matrix)
    if sign <= 0:
        return 0.0
    return float(math.exp(log_determinant / n_parameters) / n_runs * 100.0)


def _orthogonality(coded_design, features):
    groups = _main_effect_columns(coded_design, features)
    columns = np.hstack(groups) if groups else np.empty((len(coded_design), 0))
    varying = columns[:, np.ptp(columns, axis=0) > 1e-12]
    if varying.shape[1] <= 1:
        return 100.0
    correlations = np.abs(np.corrcoef(varying, rowvar=False))
    np.fill_diagonal(correlations, 0.0)
    return max(0.0, 100.0 * (1.0 - float(np.nanmax(correlations))))


def _greedy_doptimal_indices(model_matrix, n_points):
    """Select a deterministic, without-replacement sequential D-optimal subset."""
    n_candidates, n_parameters = model_matrix.shape
    if n_points > n_candidates:
        raise ValueError(
            f"Only {n_candidates} distinct feasible candidate points are available "
            f"for the requested {n_points}-run D-Optimal design."
        )
    if np.linalg.matrix_rank(model_matrix) < n_parameters:
        raise ValueError(
            "The candidate space cannot estimate every term in the selected D-Optimal model. "
            "Add factor levels or choose a simpler model order."
        )
    if n_points < n_parameters:
        raise ValueError(
            f"D-Optimal requires at least {n_parameters} runs for the selected factor types "
            f"and model (requested {n_points})."
        )

    _, _, pivots = qr(model_matrix.T, pivoting=True, mode="economic")
    selected = [int(index) for index in pivots[:n_parameters]]
    selected_set = set(selected)
    information = model_matrix[selected].T @ model_matrix[selected]

    while len(selected) < n_points:
        inverse = np.linalg.pinv(information)
        leverages = np.einsum("ij,jk,ik->i", model_matrix, inverse, model_matrix)
        leverages[list(selected_set)] = -np.inf
        best = int(np.argmax(leverages))
        selected.append(best)
        selected_set.add(best)
        information += np.outer(model_matrix[best], model_matrix[best])

    # A few Fedorov-style exchange passes materially improve the sequential
    # start while retaining unique experimental conditions.
    for _ in range(3):
        improved = False
        for position, current_index in enumerate(list(selected)):
            available = [index for index in range(n_candidates) if index not in selected_set]
            available.append(current_index)
            rows = model_matrix[available]
            inverse = np.linalg.inv(information)
            current_row = model_matrix[current_index]
            current_leverage = float(current_row @ inverse @ current_row)
            candidate_leverages = np.einsum("ij,jk,ik->i", rows, inverse, rows)
            cross_leverages = rows @ inverse @ current_row
            # Matrix determinant lemma for replacing u with v in X'X:
            # det(M - uu' + vv') / det(M).
            determinant_ratios = (
                (1.0 - current_leverage) * (1.0 + candidate_leverages)
                + cross_leverages**2
            )
            candidate_position = int(np.argmax(determinant_ratios))
            candidate_index = available[candidate_position]
            if (
                candidate_index != current_index
                and determinant_ratios[candidate_position] > 1.0 + 1e-10
            ):
                selected_set.remove(current_index)
                selected_set.add(candidate_index)
                selected[position] = candidate_index
                information += np.outer(
                    model_matrix[candidate_index], model_matrix[candidate_index]
                ) - np.outer(current_row, current_row)
                improved = True
        if not improved:
            break
    return selected


def _quality_subset(design, n_points, features, model_spec, center_count=0):
    """Reduce a classical design by D-optimal row selection, keeping centers last."""
    if n_points >= len(design):
        return design.copy()
    reserved_centers = min(center_count, n_points)
    noncenter_count = n_points - reserved_centers
    noncenter_design = design[: len(design) - center_count] if center_count else design
    centers = design[len(design) - center_count :] if center_count else np.empty((0, design.shape[1]))

    if noncenter_count:
        matrix = _build_model_matrix(noncenter_design, features, model_spec)
        # Rank-deficient subsets are allowed here by design: the user explicitly
        # requested fewer than the natural number of runs. Select the rows that
        # maximize incremental information and let the metrics expose the loss.
        selected = []
        information = np.eye(matrix.shape[1]) * 1e-10
        remaining = set(range(len(noncenter_design)))
        for _ in range(min(noncenter_count, len(noncenter_design))):
            inverse = np.linalg.pinv(information)
            candidates = sorted(remaining)
            scores = np.einsum(
                "ij,jk,ik->i", matrix[candidates], inverse, matrix[candidates]
            )
            best = candidates[int(np.argmax(scores))]
            selected.append(best)
            remaining.remove(best)
            information += np.outer(matrix[best], matrix[best])
        reduced = noncenter_design[selected]
    else:
        reduced = np.empty((0, design.shape[1]))

    if reserved_centers:
        reduced = np.vstack([reduced, centers[:reserved_centers]])
    return reduced


def _fractional_resolution(design):
    """Find the shortest defining word for a regular two-level design."""
    n_factors = design.shape[1]
    for word_length in range(3, n_factors + 1):
        for columns in itertools.combinations(range(n_factors), word_length):
            product = np.prod(design[:, columns], axis=1)
            if np.allclose(np.abs(product), 1.0) and np.allclose(product, product[0]):
                return word_length
    return None


def _doptimal_candidates(features, model_order, max_candidates=1500):
    levels = []
    for feature in features:
        codes = _feature_codes(feature)
        if codes is not None:
            levels.append(codes.tolist())
        else:
            count = 9 if model_order == "quadratic" else 7
            levels.append(np.linspace(-1.0, 1.0, count).tolist())

    total = math.prod(len(values) for values in levels)
    if total <= max_candidates:
        return np.asarray(list(itertools.product(*levels)), dtype=float)

    rng = np.random.default_rng(42)
    rows = {tuple(values[0] for values in levels), tuple(values[-1] for values in levels)}
    center = tuple(min(values, key=abs) for values in levels)
    rows.add(center)
    attempts = 0
    while len(rows) < max_candidates and attempts < max_candidates * 30:
        rows.add(tuple(values[int(rng.integers(len(values)))] for values in levels))
        attempts += 1
    return np.asarray(sorted(rows), dtype=float)


def _filter_unique_candidates(candidates, features, constraints, model_type):
    numerical = constraints.get("numerical", [])
    disallowed = constraints.get("disallowed", [])
    unique = {}
    for candidate in candidates:
        decoded = _decode_row(candidate, features, model_type)
        if (numerical or disallowed) and not check_constraints(
            decoded, numerical, disallowed, features
        ):
            continue
        key = tuple(decoded[feature["name"]] for feature in features)
        unique.setdefault(key, candidate)
    if not unique:
        raise ValueError("No candidate design points satisfy the specified constraints.")
    return np.asarray(list(unique.values()), dtype=float)


def _generate_lhs(features, n_runs):
    numeric_indices = [
        index for index, feature in enumerate(features) if feature["type"] != "categorical"
    ]
    if not numeric_indices:
        raise ValueError("Latin Hypercube Sampling requires at least one numerical factor.")
    design = np.zeros((n_runs, len(features)), dtype=float)
    numeric = doe.lhs(
        len(numeric_indices), samples=n_runs, criterion="maximin", seed=42
    ) * 2.0 - 1.0
    for numeric_column, design_column in enumerate(numeric_indices):
        feature = features[design_column]
        if feature["type"] in {"regular", "discrete"}:
            codes = _feature_codes(feature)
            strata = np.floor((numeric[:, numeric_column] + 1.0) * 0.5 * len(codes)).astype(int)
            strata = np.clip(strata, 0, len(codes) - 1)
            design[:, design_column] = codes[strata]
        else:
            design[:, design_column] = numeric[:, numeric_column]

    for column, feature in enumerate(features):
        if feature["type"] != "categorical":
            continue
        codes = _feature_codes(feature)
        balanced = np.resize(codes, n_runs).copy()
        random.Random(42 + column).shuffle(balanced)
        design[:, column] = balanced
    return design


def _validate_design_compatibility(model_type, features, constraints):
    categorical = [feature for feature in features if feature["type"] == "categorical"]
    has_constraints = bool(constraints.get("numerical") or constraints.get("disallowed"))

    if model_type in {"bbdesign", "ccdesign"} and categorical:
        raise ValueError(
            f"{MODEL_LABELS[model_type]} supports numerical factors only. "
            "Use D-Optimal for mixed numerical/categorical experiments."
        )
    if model_type in {"fracfact", "def_screening"}:
        invalid = [feature["name"] for feature in categorical if len(feature["choices"]) != 2]
        if invalid:
            raise ValueError(
                f"{MODEL_LABELS[model_type]} supports only two-level categorical factors; "
                f"check: {', '.join(invalid)}."
            )
    if has_constraints and model_type != "doptimal":
        raise ValueError(
            "Constraints are supported with D-Optimal designs only. Filtering a completed "
            f"{MODEL_LABELS[model_type]} design would invalidate its statistical properties."
        )


def generate_doe(features_config, tweaks):
    """Generate one of the six DoE designs exposed by the EDOS front end."""
    model_type = str(tweaks.get("model", "bbdesign"))
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported DoE model '{model_type}'. Select one of: "
            + ", ".join(MODEL_LABELS[model] for model in SUPPORTED_MODELS)
            + "."
        )

    try:
        requested_runs = int(float(tweaks.get("max_runs", 50)))
    except (TypeError, ValueError) as exc:
        raise ValueError("The experiment count must be a positive integer.") from exc
    if requested_runs < 1:
        raise ValueError("The experiment count must be at least 1.")

    features = _parse_features(features_config)
    n_factors = len(features)
    if n_factors < 2:
        raise ValueError(f"DoE requires at least 2 features. Found {n_factors} valid features.")
    if model_type == "bbdesign" and n_factors < 3:
        raise ValueError(f"Box-Behnken design requires at least 3 factors. Found {n_factors}.")

    constraints = dict(tweaks.get("constraints") or {})
    constraints["numerical"] = constraints.get("numerical") or []
    constraints["disallowed"] = constraints.get("disallowed") or []
    _validate_design_compatibility(model_type, features, constraints)

    exact_only = model_type in {"lhs", "doptimal"}
    use_exact_runs = exact_only or _as_bool(tweaks.get("use_exact_runs", False))
    natural_runs = requested_runs
    center_count = 0
    added_replicates = 0
    design_status = "Natural"
    design_warning = None
    fractional_resolution = None

    if model_type == "bbdesign":
        design, center_count = _move_centers_to_end(doe.bbdesign(n_factors))
        model_spec = "response_surface"
    elif model_type == "ccdesign":
        # Face-centred CCD keeps every proposed experiment within the user's bounds.
        design, center_count = _move_centers_to_end(doe.ccdesign(n_factors, face="ccf"))
        model_spec = "response_surface"
    elif model_type == "fracfact":
        try:
            requested_resolution = int(tweaks.get("fracfact_resolution", 4))
        except (TypeError, ValueError) as exc:
            raise ValueError("Fractional Factorial resolution must be III, IV, or V.") from exc
        if requested_resolution not in {3, 4, 5}:
            raise ValueError("Fractional Factorial resolution must be III, IV, or V.")
        design = doe.fracfact_by_res(n_factors, requested_resolution).astype(float)
        fractional_resolution = _fractional_resolution(design)
        model_spec = "linear"
    elif model_type == "def_screening":
        numeric_indices = [
            index for index, feature in enumerate(features) if feature["type"] != "categorical"
        ]
        categorical_indices = [
            index for index, feature in enumerate(features) if feature["type"] == "categorical"
        ]
        grouped = doe.definitive_screening_design(
            len(numeric_indices), n_cat=len(categorical_indices), min_13=True
        ).astype(float)
        design = np.zeros((len(grouped), n_factors), dtype=float)
        for source, target in enumerate(numeric_indices + categorical_indices):
            design[:, target] = grouped[:, source]
        model_spec = "dsd"
    elif model_type == "lhs":
        design = _generate_lhs(features, requested_runs)
        model_spec = "linear"
    else:  # doptimal
        model_order = str(tweaks.get("dopt_model_order", "quadratic"))
        if model_order not in {"linear", "interaction", "quadratic"}:
            raise ValueError("D-Optimal model order must be linear, interaction, or quadratic.")
        candidates = _doptimal_candidates(features, model_order)
        candidates = _filter_unique_candidates(candidates, features, constraints, model_type)
        candidate_matrix = _build_model_matrix(candidates, features, model_order)
        selected = _greedy_doptimal_indices(candidate_matrix, requested_runs)
        design = candidates[selected]
        model_spec = model_order

    natural_runs = len(design)
    if not exact_only:
        if requested_runs < natural_runs:
            design = _quality_subset(
                design, requested_runs, features, model_spec, center_count=center_count
            )
            design_status = "Reduced"
            design_warning = (
                f"The requested {requested_runs} runs are below the natural {natural_runs}-run "
                f"{MODEL_LABELS[model_type]} design. A quality-optimized reduced subset is shown; "
                "it is not the complete classical design."
            )
        elif use_exact_runs and requested_runs > natural_runs:
            added_replicates = requested_runs - natural_runs
            repeated = np.vstack([design[index % natural_runs] for index in range(added_replicates)])
            design = np.vstack([design, repeated])
            design_status = "Replicated"
            design_warning = (
                f"The exact request exceeds the natural {natural_runs}-run "
                f"{MODEL_LABELS[model_type]} design. {added_replicates} repeat run(s) were "
                "added at the end of the table."
            )

    suggested_table = [_decode_row(row, features, model_type) for row in design]
    actual_coded = _encode_table(suggested_table, features)

    if model_type in {"bbdesign", "ccdesign"}:
        # Added exact-N repetitions are reported separately, even when one of
        # those repeated rows happens to be a center point.
        base_length = len(actual_coded) - added_replicates
        center_count = int(
            np.all(np.abs(actual_coded[:base_length]) < 1e-9, axis=1).sum()
        )
    else:
        center_count = 0

    if model_type == "lhs":
        orthogonality = "N/A"
        efficiency = "N/A"
        resolution = "N/A"
        curvature = "N/A"
        rank = "N/A"
        n_parameters = "N/A"
    else:
        metric_matrix = _build_model_matrix(actual_coded, features, model_spec)
        rank = int(np.linalg.matrix_rank(metric_matrix))
        n_parameters = int(metric_matrix.shape[1])
        orthogonality = round(_orthogonality(actual_coded, features), 1)
        efficiency = round(_d_efficiency(metric_matrix), 1)

        if model_type == "fracfact":
            if fractional_resolution is None:
                resolution = "Full factorial"
            elif fractional_resolution >= 5:
                resolution = f"{fractional_resolution} (V+)"
            elif fractional_resolution == 4:
                resolution = "4 (IV)"
            else:
                resolution = "3 (III)"
            curvature = "None (Linear only)"
        else:
            resolution = "N/A"
            if model_type in {"bbdesign", "ccdesign", "def_screening"}:
                curvature = "Excellent" if rank == n_parameters else "Partial"
            elif model_type == "doptimal" and model_spec == "quadratic":
                curvature = "Excellent" if rank == n_parameters else "Partial"
            else:
                curvature = "None (Linear only)"

        if rank < n_parameters:
            rank_warning = (
                f" The returned model matrix has rank {rank} for {n_parameters} terms, so not "
                "all intended effects can be estimated independently."
            )
            design_warning = (design_warning or "Design quality is limited.") + rank_warning

    metrics = {
        "orthogonality": orthogonality,
        "efficiency": efficiency,
        "resolution": resolution,
        "curvature": curvature,
        "runs": len(suggested_table),
        "model": model_type,
        "model_label": MODEL_LABELS[model_type],
        "design_status": design_status,
        "natural_runs": natural_runs,
        "requested_runs": requested_runs,
        "design_warning": design_warning,
        "reduction_warning": design_status == "Reduced",
        # Retained for API compatibility; it now means replicated exact-N output.
        "oversampling_warning": added_replicates > 0,
        "added_replicates": added_replicates,
        "center_replicates": center_count,
        "rank": rank,
        "parameters": n_parameters,
    }
    return suggested_table, metrics
