import os
import sys
import threading
import time
import uuid
import torch
import pandas as pd
from flask import Blueprint, request, jsonify, render_template

# Import core modules
from src.core.doe import generate_doe
from src.core.bo import run_bo_optimization, run_bo_estimation, run_bo_surface
from src.core.sa import run_sa, estimate_sa, AnalysisCancelled
from src.core.constraints import (
    validate_feature_constraints,
    validate_feature_ranges,
    validate_objective_constraints,
)

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

_SA_JOBS = {}
_SA_JOBS_LOCK = threading.Lock()
_SA_JOB_TTL_SECONDS = 3600


def _cleanup_sa_jobs():
    cutoff = time.time() - _SA_JOB_TTL_SECONDS
    with _SA_JOBS_LOCK:
        expired = [
            job_id for job_id, job in _SA_JOBS.items()
            if job.get('finished_at', float('inf')) < cutoff
        ]
        for job_id in expired:
            _SA_JOBS.pop(job_id, None)


def _update_sa_job(job_id, **updates):
    with _SA_JOBS_LOCK:
        job = _SA_JOBS.get(job_id)
        if job is not None:
            job.update(updates)


def _validated_tweaks(raw_tweaks, features_config, objectives_config=None):
    tweaks = dict(raw_tweaks or {})
    constraints = dict(tweaks.get('constraints') or {})
    numerical = constraints.get('numerical') or []
    if features_config and all('range' in feature for feature in features_config):
        validate_feature_ranges(features_config)
    validate_feature_constraints(numerical, features_config or [])
    constraints['numerical'] = numerical
    constraints['disallowed'] = constraints.get('disallowed') or []
    tweaks['constraints'] = constraints
    tweaks['objective_constraints'] = validate_objective_constraints(
        tweaks.get('objective_constraints') or [], objectives_config or []
    )
    return tweaks

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


def _format_sa_result(sa_output):
    (
        num_corr_matrix,
        cat_interactions,
        results,
        cat_features,
        num_features,
        extended_obj_names,
        metadata,
    ) = sa_output
    return clean_json_data({
        'numerical_correlation': num_corr_matrix,
        'cat_interactions': cat_interactions,
        'results': results,
        'cat_features': cat_features,
        'num_features': num_features,
        'extended_obj_names': extended_obj_names,
        'metadata': metadata,
    })


def _run_sa_job(job_id, payload):
    with _SA_JOBS_LOCK:
        job = _SA_JOBS.get(job_id)
        if job is None:
            return
        cancel_event = job['cancel_event']

    def progress_callback(phase, percent):
        _update_sa_job(job_id, phase=phase, percent=percent, updated_at=time.time())

    _update_sa_job(
        job_id, status='running', phase='Preparing analysis', percent=1,
        started_at=time.time()
    )
    try:
        dataframe = pd.DataFrame(payload['data'], columns=payload['columns'])
        output = run_sa(
            df=dataframe,
            features_config=payload['features'],
            objectives_config=payload['objectives'],
            tweaks=payload['tweaks'],
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        _update_sa_job(
            job_id, status='completed', phase='Completed', percent=100,
            result=_format_sa_result(output), finished_at=time.time()
        )
    except AnalysisCancelled as exc:
        _update_sa_job(
            job_id, status='cancelled', phase='Cancelled', error=str(exc),
            finished_at=time.time()
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _update_sa_job(
            job_id, status='failed', phase='Failed', error=str(exc),
            finished_at=time.time()
        )


def _shutdown_is_allowed():
    """Return True only for a local request in a non-hosted environment."""
    client_ip = request.remote_addr
    is_local = client_ip in ('127.0.0.1', '::1')
    environment = os.environ.get('ENVIRONMENT', '').strip().lower()
    hostname = (request.host or '').split(':', 1)[0].strip().lower()
    is_hosted = (
        environment in ('production', 'prod')
        or bool(os.environ.get('SPACE_ID'))
        or bool(os.environ.get('HF_SPACE'))
        or hostname.endswith('.hf.space')
    )
    return is_local and not is_hosted


@api_bp.route('/')
def index():
    """Serves the main single-page application (SPA)."""
    return render_template('index.html', show_shutdown=_shutdown_is_allowed())


@api_bp.route('/guide')
def guide():
    """Serves the quick-start guide page."""
    return render_template('guide.html')


@api_bp.route('/shutdown', methods=['POST'])
def shutdown():
    """Gracefully shuts down the local Flask server (local development only)."""
    if not _shutdown_is_allowed():
        return jsonify({'error': 'Shutdown endpoint is restricted to local development.'}), 403

    def _stop():
        import signal
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
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )

        df = pd.DataFrame(data, columns=columns)
        
        suggestions, performance_results, constraint_status = run_bo_optimization(
            df=df,
            features_config=features_config,
            objectives_config=objectives_config,
            tweaks=tweaks,
            device=DEVICE,
            compute_device=COMPUTE_DEVICE
        )

        # Exhaustion sentinel: all optimal conditions have already been tested
        if suggestions is None:
            return jsonify(clean_json_data({
                'exhausted': True,
                'performance': performance_results,
                'constraint_status': constraint_status
            }))

        return jsonify(clean_json_data({
            'suggestions': suggestions,
            'performance': performance_results,
            'constraint_status': constraint_status
        }))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
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
        tweaks = _validated_tweaks(req_data.get('tweaks', {}), features_config)

        suggested_table, metrics = generate_doe(
            features_config=features_config,
            tweaks=tweaks
        )

        return jsonify(clean_json_data({'suggestions': suggested_table, 'metrics': metrics}))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
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
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )

        df = pd.DataFrame(data, columns=columns)
        if df.empty:
            return jsonify({'error': 'Cannot estimate with no data points.'}), 400

        raw_predictions, raw_std_errors, success_scores, global_score, violations = run_bo_estimation(
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
            'std_errors':     raw_std_errors,
            'success_scores': success_scores,
            'global_success': global_score,
            'constraint_violated': bool(violations),
            'violations': violations
        }))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/bo_surface', methods=['POST'])
def bo_surface():
    """Computes a response surface grid using the BO GP model for contour/3D plots."""
    try:
        req_data          = request.get_json()
        data              = req_data.get('data')
        columns           = req_data.get('columns')
        features_config   = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )
        x_feat            = req_data.get('x_feat')
        y_feat            = req_data.get('y_feat')
        z_axis            = req_data.get('z_axis')
        color_obj         = req_data.get('color_obj')
        fixed_values      = req_data.get('fixed_values', {})
        resolution        = req_data.get('resolution', 25)

        df = pd.DataFrame(data, columns=columns)
        if df.empty:
            return jsonify({'error': 'Cannot generate surface with no data points.'}), 400

        result = run_bo_surface(
            df=df,
            features_config=features_config,
            objectives_config=objectives_config,
            tweaks=tweaks,
            x_feat=x_feat,
            y_feat=y_feat,
            z_axis=z_axis,
            color_obj=color_obj,
            fixed_values=fixed_values,
            resolution=resolution,
            device=DEVICE,
            compute_device=COMPUTE_DEVICE
        )

        return jsonify(clean_json_data(result))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/sa_submit', methods=['POST'])
def submit_sa_job():
    """Queue a long-running SA calculation and return immediately."""
    try:
        _cleanup_sa_jobs()
        req_data = request.get_json() or {}
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        data = req_data.get('data')
        columns = req_data.get('columns')
        if not isinstance(data, list) or not isinstance(columns, list):
            return jsonify({'error': 'A valid dataset and columns are required.'}), 400
        if not features_config or not objectives_config:
            return jsonify({'error': 'At least one feature and objective are required.'}), 400
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )

        max_jobs = max(1, int(os.environ.get('SA_MAX_CONCURRENT_JOBS', '1')))
        with _SA_JOBS_LOCK:
            active_jobs = sum(
                job.get('status') in ('queued', 'running', 'cancelling')
                for job in _SA_JOBS.values()
            )
            if active_jobs >= max_jobs:
                return jsonify({
                    'error': 'Another statistical analysis is already running. '
                             'Wait for it to finish or cancel it first.'
                }), 429
            job_id = str(uuid.uuid4())
            _SA_JOBS[job_id] = {
                'status': 'queued',
                'phase': 'Queued',
                'percent': 0,
                'created_at': time.time(),
                'updated_at': time.time(),
                'cancel_event': threading.Event(),
            }

        payload = {
            'data': data,
            'columns': columns,
            'features': features_config,
            'objectives': objectives_config,
            'tweaks': tweaks,
        }
        worker = threading.Thread(
            target=_run_sa_job,
            args=(job_id, payload),
            name=f'sa-job-{job_id[:8]}',
            daemon=True,
        )
        worker.start()
        return jsonify({'job_id': job_id, 'status': 'queued'}), 202
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@api_bp.route('/sa_status/<job_id>', methods=['GET'])
def sa_job_status(job_id):
    _cleanup_sa_jobs()
    with _SA_JOBS_LOCK:
        job = _SA_JOBS.get(job_id)
        if job is None:
            return jsonify({'error': 'Analysis job not found or expired.'}), 404
        response = {
            'job_id': job_id,
            'status': job.get('status'),
            'phase': job.get('phase'),
            'percent': job.get('percent', 0),
        }
        if job.get('status') == 'completed':
            response['result'] = job.get('result')
        elif job.get('status') in ('failed', 'cancelled'):
            response['error'] = job.get('error')
    return jsonify(clean_json_data(response))


@api_bp.route('/sa_cancel/<job_id>', methods=['POST'])
def cancel_sa_job(job_id):
    with _SA_JOBS_LOCK:
        job = _SA_JOBS.get(job_id)
        if job is None:
            return jsonify({'error': 'Analysis job not found or expired.'}), 404
        if job.get('status') in ('completed', 'failed', 'cancelled'):
            return jsonify({'job_id': job_id, 'status': job.get('status')})
        job['cancel_event'].set()
        job['status'] = 'cancelling'
        job['phase'] = 'Cancellation requested'
        job['updated_at'] = time.time()
    return jsonify({'job_id': job_id, 'status': 'cancelling'})


@api_bp.route('/sa', methods=['POST'])
def run_sa_endpoint():
    """Backward-compatible synchronous Statistical Analysis endpoint."""
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        columns = req_data.get('columns')
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )

        df = pd.DataFrame(data, columns=columns)
        output = run_sa(
            df=df,
            features_config=features_config,
            objectives_config=objectives_config,
            tweaks=tweaks
        )
        return jsonify(_format_sa_result(output))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/estimate', methods=['POST'])
def estimate_sa_endpoint():
    """Predicts objective values using the SA model."""
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        columns = req_data.get('columns')
        input_values = req_data.get('inputs')
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        model_name = req_data.get('model', 'linear')
        params = req_data.get('params', {})
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )

        df = pd.DataFrame(data, columns=columns)
        predictions, std_errors, success_feedback, global_score, violations = estimate_sa(
            df=df,
            input_values=input_values,
            features_config=features_config,
            objectives_config=objectives_config,
            model_name=model_name,
            params=params,
            tweaks=tweaks
        )

        return jsonify(clean_json_data({
            'predictions': predictions,
            'std_errors': std_errors,
            'success_scores': success_feedback,
            'global_success': global_score,
            'constraint_violated': bool(violations),
            'violations': violations
        }))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
