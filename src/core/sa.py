import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import shap

def clean_dataframe_sa(df, features_config, objectives_config):
    # 1. Feature filtering: Drop rows missing any required feature value
    valid_feat_mask = pd.Series(True, index=df.index)
    for f in features_config:
        name = f['name']
        if name not in df.columns:
            continue
        col_series = df[name]
        if f['type'] in ['continuous', 'discrete', 'regular']:
            num_col = pd.to_numeric(col_series, errors='coerce')
            valid_feat_mask = valid_feat_mask & num_col.notna()
        else:
            valid_feat_mask = valid_feat_mask & col_series.notna() & (col_series.astype(str).str.strip() != "")
            
    df_clean = df[valid_feat_mask].copy()
    
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


def run_sa(df, features_config, objectives_config, tweaks):
    """
    Performs statistical analysis of experimental data to understand
    feature-outcome relationships.
    """
    # Filter the dataframe first
    df_clean = clean_dataframe_sa(df, features_config, objectives_config)

    # Ensure categorical columns are stored as strings to prevent mixed-type issues
    for f in features_config:
        if f.get('type') == 'categorical':
            df_clean[f['name']] = df_clean[f['name']].astype(str)

    feat_names = [f['name'] for f in features_config if f.get('range')]
    obj_names  = [o['name'] for o in objectives_config]

    if not feat_names or not obj_names:
        raise ValueError('Missing features or objectives for analysis.')

    if df_clean.empty:
        raise ValueError('No valid data points left after filtering missing features/objectives.')

    X_raw = df_clean[feat_names].copy()
    Y_raw = df_clean[obj_names].copy()

    # STEP 1 — Goal Transformation → Normalized Success Scores
    success_data      = {}
    processed_success = []
    weights           = []

    for obj in objectives_config:
        name = obj['name']
        w    = float(obj.get('importance', 100))
        weights.append(w)
        
        # Get raw objective values (keep NaNs)
        raw_y = pd.to_numeric(Y_raw[name], errors='coerce').values
        
        if   obj['type'] == 'maximize': transformed = raw_y
        elif obj['type'] == 'minimize': transformed = -raw_y
        else:
            t          = float(obj.get('target', 0))
            transformed = -np.abs(raw_y - t)

        # Normalize per-objective ignoring NaNs
        valid_trans = transformed[~np.isnan(transformed)]
        if len(valid_trans) > 0:
            s_min, s_max = valid_trans.min(), valid_trans.max()
            if s_max > s_min:
                normalized = (transformed - s_min) / (s_max - s_min)
            else:
                normalized = np.ones_like(transformed)
        else:
            normalized = np.zeros_like(transformed)

        success_data[name] = normalized
        processed_success.append(normalized)

    # Compute weighted Global Success Index when multiple objectives present
    if len(processed_success) >= 2:
        succ_matrix = np.column_stack(processed_success)
        w_arr       = np.array(weights)
        
        global_success = []
        for row_idx in range(succ_matrix.shape[0]):
            row_vals = succ_matrix[row_idx]
            nan_mask = np.isnan(row_vals)
            
            valid_vals = row_vals[~nan_mask]
            valid_weights = w_arr[~nan_mask]
            
            if len(valid_vals) > 0 and np.sum(valid_weights) > 0:
                g_succ = np.average(valid_vals, weights=valid_weights)
            elif len(valid_vals) > 0:
                g_succ = np.mean(valid_vals)
            else:
                g_succ = 0.0
            global_success.append(g_succ)
            
        success_data['global_success'] = np.array(global_success)
    else:
        global_success = None

    Y_success         = pd.DataFrame(success_data)
    extended_obj_names = list(success_data.keys())

    # STEP 2 — Feature Encoding and Surrogate Modeling
    X_encoded       = pd.get_dummies(X_raw, drop_first=False)
    encoded_feat_names = list(X_encoded.columns)
    X = X_encoded.fillna(0).astype(float)
    Y = Y_success.apply(pd.to_numeric, errors='coerce').astype(float) # Keep NaNs in Y!

    # Correlation matrix in success space (fill NaNs with 0 for joint corr structure, or calculate pairwise)
    corr_matrix = pd.concat([X, Y], axis=1).corr().fillna(0).to_dict()

    model_name = tweaks.get('model', 'linear')
    params     = tweaks.get('params', {})

    results = {}
    for obj in extended_obj_names:
        y = Y[obj]
        
        # Smart Per-Objective Splitting: Filter out rows where objective is NaN
        valid_mask = y.notna()
        X_obj = X[valid_mask].reset_index(drop=True)
        y_obj = y[valid_mask].reset_index(drop=True)
        X_raw_obj = X_raw[valid_mask].reset_index(drop=True)

        if len(y_obj) < 5:
            results[obj] = {
                'reliability': None,
                'importance': {f_name: 0.0 for f_name in feat_names},
                'shap_data': None,
                'fit_flag': 'Poor'
            }
            continue

        if model_name == 'random_forest':
            model = ensemble.RandomForestRegressor(
                n_estimators=int(params.get('n_estimators', 100)),
                max_depth=int(params.get('max_depth', 10)) if params.get('max_depth') else None
            )
        elif model_name == 'mlp':
            hidden_layers = str(params.get('mlp_layers', '100'))
            layers = [int(x.strip()) for x in hidden_layers.split(',') if x.strip()]
            model  = MLPRegressor(
                hidden_layer_sizes=tuple(layers),
                max_iter=int(params.get('mlp_iter', 500)),
                random_state=42
            )
        else:
            model = lm.LinearRegression()

        ss     = ShuffleSplit(n_splits=min(5, len(y_obj)), test_size=0.2, random_state=42)
        scores = []

        X_data = X_obj
        if model_name == 'mlp':
            scaler = StandardScaler()
            X_data = pd.DataFrame(scaler.fit_transform(X_obj), columns=X_obj.columns)

        for train_idx, test_idx in ss.split(X_data):
            X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
            y_train, y_test = y_obj.iloc[train_idx], y_obj.iloc[test_idx]
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))

        reliability = np.mean(scores)
        model.fit(X_data, y_obj)

        if model_name == 'linear':
            importance = np.abs(model.coef_)
        elif model_name == 'mlp':
            imp_res    = permutation_importance(model, X_data, y_obj, n_repeats=5, random_state=42)
            importance = imp_res.importances_mean
        else:
            importance = model.feature_importances_

        imp_sum        = np.sum(importance)
        importance_pct = (importance / imp_sum * 100).tolist() if imp_sum > 0 else [0.0] * len(importance)

        grouped_importance = {}
        for f_name in feat_names:
            rel_indices = [i for i, c in enumerate(encoded_feat_names) if c == f_name or f"{f_name}_" in c]
            grouped_importance[f_name] = round(sum(importance_pct[i] for i in rel_indices), 2)

        shap_data = None
        try:
            if model_name == 'random_forest':
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_data)
            elif model_name == 'mlp':
                background = shap.kmeans(X_data, min(50, len(X_data)))
                explainer  = shap.KernelExplainer(model.predict, background)
                shap_vals  = explainer.shap_values(X_data)
            elif model_name == 'linear':
                explainer = shap.LinearExplainer(model, X_data)
                shap_vals = explainer.shap_values(X_data)

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]

            grouped_shap = {f: np.zeros(len(X_data)) for f in feat_names}
            for f_name in feat_names:
                rel_indices = [i for i, c in enumerate(encoded_feat_names) if c == f_name or f"{f_name}_" in c]
                for idx in rel_indices:
                    grouped_shap[f_name] += shap_vals[:, idx]

            shap_data = {
                'features':      feat_names,
                'shap_values':   [grouped_shap[f].tolist() for f in feat_names],
                'feature_values': [X_raw_obj[f].tolist() for f in feat_names]
            }
        except Exception as e:
            print(f"SHAP Explainer execution failed on {model_name}: {e}")

        results[obj] = {
            'reliability': None if np.isnan(reliability) else round(float(reliability), 3),
            'importance':  grouped_importance,
            'shap_data':   shap_data,
            'fit_flag':    'Good' if reliability > 0.7 else ('Limited' if reliability >= 0.4 else 'Poor')
        }

    # STEP 3 — Categorical Interaction Pivot Tables
    cat_features = [f['name'] for f in features_config if f.get('type') == 'categorical']
    num_features = [f['name'] for f in features_config if f.get('type') in ['continuous', 'discrete']]

    cat_interactions = {}
    for i in range(len(cat_features)):
        for j in range(i + 1, len(cat_features)):
            f1, f2   = cat_features[i], cat_features[j]
            pair_key = f"{f1} vs {f2}"
            cat_interactions[pair_key] = {}
            for obj in extended_obj_names:
                combo_df = pd.concat([df_clean[[f1, f2]], Y_success[[obj]]], axis=1)
                pivot    = combo_df.groupby([f1, f2])[obj].max().unstack().fillna(0)
                cat_interactions[pair_key][obj] = {
                    'z': pivot.values.tolist(),
                    'x': pivot.columns.tolist(),
                    'y': pivot.index.tolist()
                }

    # Numerical Pearson correlation.
    # Y_success is used directly (no NaN fill) — pandas.corr() uses pairwise complete
    # observations, so rows where an objective is missing are excluded per pair rather
    # than biasing the correlation toward zero.
    num_df = df_clean[num_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    num_corr_matrix = (
        pd.concat([num_df, Y_success], axis=1)
        .corr().loc[extended_obj_names, num_features].fillna(0).to_dict()
    )

    return num_corr_matrix, cat_interactions, results, cat_features, num_features, extended_obj_names


def estimate_sa(df, input_values, features_config, objectives_config, model_name, params):
    """
    Predicts objective values at a user-specified operating point using the
    same sklearn surrogate model fitted in run_sa().
    """
    df_clean = clean_dataframe_sa(df, features_config, objectives_config)
    if df_clean.empty:
        raise ValueError('No valid data points left after filtering missing features/objectives.')

    feat_names = [f['name'] for f in features_config if f.get('range')]
    obj_names  = [o['name'] for o in objectives_config]

    # Build Training Feature Matrix
    X_raw_train    = df_clean[feat_names].copy()
    X_train_encoded = pd.get_dummies(X_raw_train)
    encoded_cols   = X_train_encoded.columns
    X_train        = X_train_encoded.fillna(0).astype(float)

    for o in obj_names:
        if df_clean[o].dtype == object or str(df_clean[o].dtype) in ['string', 'category']:
            df_clean[o] = df_clean[o].astype(str).astype('category').cat.codes
    Y = df_clean[obj_names].apply(pd.to_numeric, errors='coerce').astype(float) # Keep NaNs

    # Encode the Query Point
    input_data_raw = pd.DataFrame([input_values], columns=feat_names)

    for col in feat_names:
        if df_clean[col].dtype == object or str(df_clean[col].dtype) in ['string', 'category']:
            unique_train_vals  = df_clean[col].astype(str).unique()
            input_data_raw[col] = pd.Categorical(input_data_raw[col].astype(str), categories=unique_train_vals)

    input_df_encoded = pd.get_dummies(input_data_raw)
    input_df_aligned = input_df_encoded.reindex(columns=encoded_cols, fill_value=0).astype(float)

    # Scale Features
    X_train_data = X_train
    input_data   = input_df_aligned
    if model_name == 'mlp':
        scaler       = StandardScaler()
        X_train_data = pd.DataFrame(scaler.fit_transform(X_train), columns=encoded_cols)
        input_data   = pd.DataFrame(scaler.transform(input_df_aligned), columns=encoded_cols)

    # Fit Model and Predict per-objective using valid data
    predictions = {}
    for obj in obj_names:
        y_obj = Y[obj]
        valid_mask = y_obj.notna()
        X_tr_obj = X_train_data[valid_mask]
        y_tr_obj = y_obj[valid_mask]

        if len(y_tr_obj) == 0:
            predictions[obj] = 0.0
            continue

        if model_name == 'random_forest':
            model = ensemble.RandomForestRegressor(
                n_estimators=int(params.get('n_estimators', 100)),
                max_depth=int(params.get('max_depth', 10)) if params.get('max_depth') else None
            )
        elif model_name == 'mlp':
            hidden_layers = str(params.get('mlp_layers', '100'))
            layers = [int(x.strip()) for x in hidden_layers.split(',') if x.strip()]
            model  = MLPRegressor(
                hidden_layer_sizes=tuple(layers),
                max_iter=int(params.get('mlp_iter', 500)),
                random_state=42
            )
        else:
            model = lm.LinearRegression()

        model.fit(X_tr_obj, y_tr_obj)
        pred             = model.predict(input_data)[0]
        predictions[obj] = round(float(pred), 3)

    # Compute Success Scores
    success_feedback  = {}
    processed_success = []
    weights           = []

    for obj in objectives_config:
        name = obj['name']
        val  = predictions[name]
        w    = float(obj.get('importance', 100))
        weights.append(w)

        if   obj['type'] == 'maximize': s = val
        elif obj['type'] == 'minimize': s = -val
        else:
            t = float(obj.get('target', 0))
            s = -abs(val - t)

        train_raw = Y[name].values
        valid_train = train_raw[~np.isnan(train_raw)]
        
        if len(valid_train) == 0:
            success_feedback[name] = 0.0
            processed_success.append(np.nan)
            continue

        if   obj['type'] == 'maximize': train_s = valid_train
        elif obj['type'] == 'minimize': train_s = -valid_train
        else:               train_s = -np.abs(valid_train - float(obj.get('target', 0)))

        s_min, s_max = train_s.min(), train_s.max()
        norm_s = (s - s_min) / (s_max - s_min) if s_max > s_min else 1.0

        success_feedback[name] = round(float(norm_s * 100), 1)
        processed_success.append(norm_s)

    # Calculate global success with dynamic renormalization of weights
    if processed_success:
        succ_arr = np.array(processed_success)
        w_arr = np.array(weights)
        nan_mask = np.isnan(succ_arr)
        
        valid_succ = succ_arr[~nan_mask]
        valid_weights = w_arr[~nan_mask]
        
        if len(valid_succ) > 0 and np.sum(valid_weights) > 0:
            global_score = np.average(valid_succ, weights=valid_weights)
        elif len(valid_succ) > 0:
            global_score = np.mean(valid_succ)
        else:
            global_score = 0.0
    else:
        global_score = 0

    return predictions, success_feedback, round(float(global_score * 100), 1)
