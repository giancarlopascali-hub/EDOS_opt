import os
import sys
import torch
import pandas as pd
from flask import Blueprint, request, jsonify, render_template

# Import core modules
from src.core.doe import generate_doe
from src.core.bo import run_bo_optimization, run_bo_estimation
from src.core.sa import run_sa, estimate_sa

# Create the blueprint
api_bp = Blueprint('api', __name__)

# Device configuration (CUDA GPU > DirectML > CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_DEVICE = DEVICE

try:
    import torch_directml
    if torch_directml.is_available():
        DEVICE = torch_directml.device()
        COMPUTE_DEVICE = torch.device("cpu")
except Exception:
    pass

import math
import numpy as np

def clean_json_data(data):
    if isinstance(data, dict):
        return {k: clean_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    elif isinstance(data, np.floating):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.ndarray):
        return clean_json_data(data.tolist())
    else:
        return data


@api_bp.route('/')
def index():
    """Serves the main single-page application (SPA)."""
    return render_template('index.html')


@api_bp.route('/guide')
def guide():
    """Serves the quick-start guide page."""
    return render_template('guide.html')


@api_bp.route('/shutdown', methods=['POST'])
def shutdown():
    """Gracefully shuts down the Flask development server."""
    import threading

    def _stop():
        import time, signal
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_stop, daemon=True).start()
    return jsonify({'status': 'shutting down'})


@api_bp.route('/upload', methods=['POST'])
def upload():
    """Accepts a multipart CSV file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)
        data = df.to_dict(orient='split')
        return jsonify(clean_json_data(data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/optimize', methods=['POST'])
def optimize():
    """Core Bayesian Optimization endpoint."""
    try:
        req_data = request.get_json()
        data             = req_data.get('data')
        columns          = req_data.get('columns')
        features_config  = req_data.get('features')
        objectives_config= req_data.get('objectives')
        tweaks           = req_data.get('tweaks', {})

        df = pd.DataFrame(data, columns=columns)
        
        suggestions, performance_results = run_bo_optimization(
            df=df,
            features_config=features_config,
            objectives_config=objectives_config,
            tweaks=tweaks,
            device=DEVICE,
            compute_device=COMPUTE_DEVICE
        )

        return jsonify(clean_json_data({'suggestions': suggestions, 'performance': performance_results}))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/doe', methods=['POST'])
def run_doe():
    """Generates a Design of Experiments (DoE) table using pyDOE2."""
    try:
        req_data       = request.get_json()
        features_config = req_data.get('features', [])
        tweaks         = req_data.get('tweaks', {})

        suggested_table, metrics = generate_doe(
            features_config=features_config,
            tweaks=tweaks
        )

        return jsonify(clean_json_data({'suggestions': suggested_table, 'metrics': metrics}))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/estimate_bo', methods=['POST'])
def estimate_bo():
    """Predicts objective values using the BO GP model."""
    try:
        req_data         = request.get_json()
        data             = req_data.get('data')
        columns          = req_data.get('columns')
        input_values     = req_data.get('inputs')
        features_config  = req_data.get('features', [])
        objectives_config= req_data.get('objectives', [])
        tweaks           = req_data.get('tweaks', {})

        df = pd.DataFrame(data, columns=columns)
        if df.empty:
            return jsonify({'error': 'Cannot estimate with no data points.'}), 400

        raw_predictions, success_scores, global_score = run_bo_estimation(
            df=df,
            input_values=input_values,
            features_config=features_config,
            objectives_config=objectives_config,
            tweaks=tweaks,
            device=DEVICE,
            compute_device=COMPUTE_DEVICE
        )

        return jsonify(clean_json_data({
            'predictions':    raw_predictions,
            'success_scores': success_scores,
            'global_success': global_score
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/sa', methods=['POST'])
def run_sa_endpoint():
    """Performs statistical analysis of experimental data."""
    try:
        req_data         = request.get_json()
        data             = req_data.get('data')
        columns          = req_data.get('columns')
        features_config  = req_data.get('features', [])
        objectives_config= req_data.get('objectives', [])
        tweaks           = req_data.get('tweaks', {})

        df = pd.DataFrame(data, columns=columns)
        
        num_corr_matrix, cat_interactions, results, cat_features, num_features, extended_obj_names = run_sa(
            df=df,
            features_config=features_config,
            objectives_config=objectives_config,
            tweaks=tweaks
        )

        return jsonify(clean_json_data({
            'numerical_correlation': num_corr_matrix,
            'cat_interactions':      cat_interactions,
            'results':               results,
            'cat_features':          cat_features,
            'num_features':          num_features,
            'extended_obj_names':    extended_obj_names
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/estimate', methods=['POST'])
def estimate_sa_endpoint():
    """Predicts objective values using the SA model."""
    try:
        req_data         = request.get_json()
        data             = req_data.get('data')
        columns          = req_data.get('columns')
        input_values     = req_data.get('inputs')
        features_config  = req_data.get('features', [])
        objectives_config= req_data.get('objectives', [])
        model_name       = req_data.get('model', 'linear')
        params           = req_data.get('params', {})

        df         = pd.DataFrame(data, columns=columns)
        
        predictions, success_feedback, global_score = estimate_sa(
            df=df,
            input_values=input_values,
            features_config=features_config,
            objectives_config=objectives_config,
            model_name=model_name,
            params=params
        )

        return jsonify(clean_json_data({
            'predictions':    predictions,
            'success_scores': success_feedback,
            'global_success': global_score
        }))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


