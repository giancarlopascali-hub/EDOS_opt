import os
import sys
import threading
import time
import torch
import pandas as pd
from flask import Blueprint, request, jsonify, render_template

# Import core modules
from src.core.doe import generate_doe
from src.core.bo import (
    SurfaceCalculationCancelled,
    validate_exploration,
    run_bo_optimization,
    run_bo_estimation,
    run_bo_surface,
)
from src.core.bo_policy import (
    load_bo_budget,
    normalized_objective_weights,
    select_bo_policy,
    validate_bo_data_mode,
    validate_multi_objective_method,
)
from src.api.bo_worker import (
    BOChildFailure,
    BOProcessCancelled,
    BOProcessTimeout,
    run_bo_in_subprocess,
)
from src.api.jobs import (
    cancel_background_job,
    get_background_job_response,
    get_cancel_event,
    queue_background_job,
    update_background_job,
)
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

BO_SURFACE_MAX_ROWS = 650


def _is_hosted_environment():
    environment = os.environ.get('ENVIRONMENT', '').strip().lower()
    hostname = (request.host or '').split(':', 1)[0].strip().lower()
    return (
        environment in ('production', 'prod')
        or bool(os.environ.get('SPACE_ID'))
        or bool(os.environ.get('HF_SPACE'))
        or hostname.endswith('.hf.space')
    )


def _validate_bo_surface_row_count(data):
    """Protect the local exact-GP surface calculation from huge datasets."""
    if isinstance(data, list) and len(data) > BO_SURFACE_MAX_ROWS:
        raise ValueError(
            f'BO response-surface plots are limited to {BO_SURFACE_MAX_ROWS:,} '
            f'dataset rows for reliable completion; this dataset has {len(data):,}. '
            'Filter or reduce the dataset before creating a contour or 3D plot.'
        )


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
    acquisition_type = str(tweaks.get('acq_type', 'EI')).upper()
    if acquisition_type not in {'EI', 'LCB'}:
        raise ValueError("Acquisition must be either 'EI' or 'LCB'.")
    tweaks['acq_type'] = acquisition_type
    tweaks['multi_objective_method'] = validate_multi_objective_method(
        tweaks.get('multi_objective_method', 'auto')
    )
    tweaks['data_mode'] = validate_bo_data_mode(tweaks.get('data_mode', 'auto'))
    strategy = str(tweaks.get('optimization_strategy', 'gradient_based'))
    if strategy not in {'gradient_based', 'exhaustive_grid'}:
        raise ValueError('Unknown Bayesian optimization strategy.')
    tweaks['optimization_strategy'] = strategy
    tweaks['exploration'] = validate_exploration(tweaks.get('exploration', 0.5))
    if objectives_config:
        weights = [
            objective.get('importance', 1) for objective in objectives_config
        ]
        normalized_objective_weights(weights)
        select_bo_policy(
            len(objectives_config),
            weights,
            acquisition_type,
            load_bo_budget(is_cuda=False).nehvi_max_objectives,
            requested_method=tweaks['multi_objective_method'],
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


def _format_bo_result(result):
    """Preserve the public BO response while adding bounded-run diagnostics."""

    serialized = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    response = {
        "performance": serialized.get("performance") or {},
        "constraint_status": serialized.get("constraint_status") or {},
        "diagnostics": serialized.get("diagnostics") or {},
    }
    if serialized.get("suggestions") is None:
        response["exhausted"] = True
    else:
        response["suggestions"] = serialized.get("suggestions")
    return clean_json_data(response)


def _run_bo_job(job_id, payload):
    """Supervise one process-isolated BO calculation from a lightweight thread."""

    cancel_event = get_cancel_event(job_id)
    if cancel_event is None:
        return

    def progress_callback(phase, percent, details):
        if cancel_event.is_set():
            update_background_job(
                job_id,
                status="cancelling",
                phase="Stopping optimization",
                percent=percent,
                details=details,
            )
            return
        updates = {
            "status": "running",
            "phase": phase,
            "percent": percent,
            "details": details,
        }
        if details.get("algorithm"):
            updates["algorithm"] = details["algorithm"]
        update_background_job(job_id, **updates)

    update_background_job(
        job_id,
        status="running",
        phase="Starting isolated BO worker",
        percent=1,
        started_at=time.time(),
    )
    try:
        result = run_bo_in_subprocess(payload, cancel_event, progress_callback)
        diagnostics = result.get("diagnostics") or {}
        update_background_job(
            job_id,
            status="completed",
            phase="Completed",
            percent=100,
            algorithm=diagnostics.get("algorithm"),
            details={
                "elapsed_seconds": diagnostics.get("elapsed_seconds"),
                "algorithm": diagnostics.get("algorithm"),
                "algorithm_label": diagnostics.get("algorithm_label"),
            },
            result=_format_bo_result(result),
            finished_at=time.time(),
        )
    except BOProcessCancelled as exc:
        update_background_job(
            job_id,
            status="cancelled",
            phase="Cancelled",
            error=str(exc),
            error_code="cancelled",
            finished_at=time.time(),
        )
    except BOProcessTimeout as exc:
        update_background_job(
            job_id,
            status="failed",
            phase="Stopped at hard time limit",
            error=str(exc),
            error_code="timeout",
            finished_at=time.time(),
        )
    except BOChildFailure as exc:
        update_background_job(
            job_id,
            status="failed",
            phase="Optimization failed",
            error=str(exc),
            error_code=exc.error_code,
            finished_at=time.time(),
        )
    except Exception:
        import traceback

        traceback.print_exc()
        update_background_job(
            job_id,
            status="failed",
            phase="Optimization failed",
            error="The BO supervisor failed unexpectedly. Please retry.",
            error_code="supervisor_error",
            finished_at=time.time(),
        )


def _run_sa_job(job_id, payload):
    cancel_event = get_cancel_event(job_id)
    if cancel_event is None:
        return

    def progress_callback(phase, percent):
        update_background_job(job_id, phase=phase, percent=percent)

    update_background_job(
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
        update_background_job(
            job_id, status='completed', phase='Completed', percent=100,
            result=_format_sa_result(output), finished_at=time.time()
        )
    except AnalysisCancelled as exc:
        update_background_job(
            job_id, status='cancelled', phase='Cancelled', error=str(exc),
            finished_at=time.time()
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        update_background_job(
            job_id, status='failed', phase='Failed', error=str(exc),
            finished_at=time.time()
        )


def _run_bo_surface_job(job_id, payload):
    cancel_event = get_cancel_event(job_id)
    if cancel_event is None:
        return
    update_background_job(
        job_id, status='running', phase='Preparing response surface', percent=5,
        started_at=time.time(),
    )
    try:
        dataframe = pd.DataFrame(payload['data'], columns=payload['columns'])
        if dataframe.empty:
            raise ValueError('Cannot generate surface with no data points.')
        result = run_bo_surface(
            df=dataframe,
            features_config=payload['features'],
            objectives_config=payload['objectives'],
            tweaks=payload['tweaks'],
            x_feat=payload.get('x_feat'),
            y_feat=payload.get('y_feat'),
            z_axis=payload.get('z_axis'),
            color_obj=payload.get('color_obj'),
            fixed_values=payload.get('fixed_values', {}),
            resolution=payload.get('resolution', 25),
            device=DEVICE,
            compute_device=COMPUTE_DEVICE,
            cancel_event=cancel_event,
            progress_callback=lambda phase, percent: update_background_job(
                job_id, status='running', phase=phase, percent=percent,
            ),
        )
        if cancel_event.is_set():
            raise SurfaceCalculationCancelled(
                'Response-surface calculation cancelled by user.'
            )
        update_background_job(
            job_id, status='completed', phase='Completed', percent=100,
            result=clean_json_data(result), finished_at=time.time(),
        )
    except SurfaceCalculationCancelled as exc:
        update_background_job(
            job_id, status='cancelled', phase='Cancelled', error=str(exc),
            finished_at=time.time(),
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        update_background_job(
            job_id, status='failed', phase='Failed', error=str(exc),
            finished_at=time.time(),
        )


def _shutdown_is_allowed():
    """Return True only for a local request in a non-hosted environment."""
    client_ip = request.remote_addr
    is_local = client_ip in ('127.0.0.1', '::1')
    return is_local and not _is_hosted_environment()


@api_bp.route('/')
def index():
    """Serves the main single-page application (SPA)."""
    return render_template(
        'index.html',
        show_shutdown=_shutdown_is_allowed(),
        hosted_environment=_is_hosted_environment(),
    )


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
    """Backward-compatible synchronous endpoint for small local/API jobs."""
    try:
        req_data = request.get_json()
        data             = req_data.get('data')
        columns          = req_data.get('columns')
        features_config  = req_data.get('features')
        objectives_config= req_data.get('objectives')
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )
        tweaks['_allow_unsafe_full'] = not _is_hosted_environment()

        raw_batch_size = float(tweaks.get('batch_size', 1))
        if not raw_batch_size.is_integer():
            raise ValueError('Batch size must be a whole number.')
        batch_size = int(raw_batch_size)
        sync_max_objectives = max(1, int(os.environ.get('BO_SYNC_MAX_OBJECTIVES', '2')))
        sync_max_batch = max(1, int(os.environ.get('BO_SYNC_MAX_BATCH_SIZE', '3')))
        if _is_hosted_environment() and (
            len(objectives_config or []) > sync_max_objectives
            or batch_size > sync_max_batch
        ):
            return jsonify({
                'error': (
                    'This workload must use the background BO endpoint so it is '
                    'observable and is not interrupted by the web gateway.'
                ),
                'error_code': 'async_required',
            }), 422

        df = pd.DataFrame(data, columns=columns)
        result = run_bo_optimization(
            df=df,
            features_config=features_config,
            objectives_config=objectives_config,
            tweaks=tweaks,
            device=DEVICE,
            compute_device=COMPUTE_DEVICE,
            compute_performance=bool(req_data.get('compute_performance', False)),
        )
        return jsonify(_format_bo_result(result))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Bayesian optimization failed unexpectedly. Review the server log.',
            'error_code': 'optimization_error',
        }), 500


@api_bp.route('/bo_submit', methods=['POST'])
def submit_bo_job():
    """Validate and queue a process-isolated Bayesian optimization job."""

    try:
        req_data = request.get_json() or {}
        data = req_data.get('data')
        columns = req_data.get('columns')
        features = req_data.get('features', [])
        objectives = req_data.get('objectives', [])
        if not isinstance(data, list) or not isinstance(columns, list):
            return jsonify({'error': 'A valid dataset and columns are required.'}), 400
        if any(not isinstance(row, (list, tuple)) or len(row) != len(columns) for row in data):
            return jsonify({'error': 'Every dataset row must match the columns.'}), 400
        if not features or not objectives:
            return jsonify({'error': 'At least one feature and objective are required.'}), 400
        tweaks = _validated_tweaks(req_data.get('tweaks', {}), features, objectives)
        # This is server-owned. A request cannot opt a hosted worker out of the
        # exact-GP safety boundary.
        tweaks['_allow_unsafe_full'] = not _is_hosted_environment()
        # Parse here so malformed and out-of-range requests fail before reserving a slot.
        raw_batch_size = float(tweaks.get('batch_size', 1))
        if not raw_batch_size.is_integer():
            raise ValueError('Batch size must be a whole number.')
        batch_size = int(raw_batch_size)
        max_batch_size = max(1, int(os.environ.get('BO_MAX_BATCH_SIZE', '20')))
        if not 1 <= batch_size <= max_batch_size:
            raise ValueError(
                f'Batch size must be between 1 and {max_batch_size}.'
            )
        try:
            random_seed = int(req_data.get('random_seed', 1729))
        except (TypeError, ValueError) as exc:
            raise ValueError('The BO random seed must be an integer.') from exc
        if not 0 <= random_seed < 2**63:
            raise ValueError('The BO random seed must be between 0 and 2^63 - 1.')
        payload = {
            'data': data,
            'columns': columns,
            'features': features,
            'objectives': objectives,
            'tweaks': tweaks,
            'compute_performance': bool(req_data.get('compute_performance', False)),
            'random_seed': random_seed,
        }
        job_id = queue_background_job(
            'bo-optimization', payload, _run_bo_job, 'Queued Bayesian optimization'
        )
        if job_id is None:
            return jsonify({
                'error': (
                    'Another analysis is already running. Wait for it to finish '
                    'or cancel it before starting BO.'
                ),
                'error_code': 'busy',
            }), 429
        return jsonify({'job_id': job_id, 'status': 'queued'}), 202
    except ValueError as exc:
        return jsonify({'error': str(exc), 'error_code': 'validation'}), 400
    except Exception:
        import traceback

        traceback.print_exc()
        return jsonify({
            'error': 'Unable to queue Bayesian optimization.',
            'error_code': 'submission_error',
        }), 500


@api_bp.route('/doe', methods=['POST'])
def run_doe():
    """Generate a validated Design of Experiments (DoE) table."""
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


@api_bp.route('/bo_surface_submit', methods=['POST'])
def bo_surface_submit():
    """Queue a cancellable BO response-surface calculation."""
    try:
        req_data = request.get_json() or {}
        data = req_data.get('data')
        columns = req_data.get('columns')
        features = req_data.get('features', [])
        objectives = req_data.get('objectives', [])
        if not isinstance(data, list) or not isinstance(columns, list):
            return jsonify({'error': 'A valid dataset and columns are required.'}), 400
        _validate_bo_surface_row_count(data)
        if len(features) < 2 or not objectives:
            return jsonify({'error': 'At least two features and one objective are required.'}), 400
        tweaks = _validated_tweaks(req_data.get('tweaks', {}), features, objectives)
        payload = {
            'data': data,
            'columns': columns,
            'features': features,
            'objectives': objectives,
            'x_feat': req_data.get('x_feat'),
            'y_feat': req_data.get('y_feat'),
            'z_axis': req_data.get('z_axis'),
            'color_obj': req_data.get('color_obj'),
            'fixed_values': req_data.get('fixed_values', {}),
            'resolution': req_data.get('resolution', 25),
            'tweaks': tweaks,
        }
        job_id = queue_background_job(
            'bo-surface', payload, _run_bo_surface_job,
            'Queued response surface',
        )
        if job_id is None:
            return jsonify({
                'error': 'Another analysis job is already running. '
                         'Wait for it to finish or cancel it first.'
            }), 429
        return jsonify({'job_id': job_id, 'status': 'queued'}), 202
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@api_bp.route('/sa_submit', methods=['POST'])
def submit_sa_job():
    """Queue a long-running SA calculation and return immediately."""
    try:
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

        payload = {
            'data': data,
            'columns': columns,
            'features': features_config,
            'objectives': objectives_config,
            'tweaks': tweaks,
        }
        job_id = queue_background_job(
            'sa-analysis', payload, _run_sa_job, 'Queued statistical analysis'
        )
        if job_id is None:
            return jsonify({
                'error': 'Another statistical analysis is already running. '
                         'Wait for it to finish or cancel it first.'
            }), 429
        return jsonify({'job_id': job_id, 'status': 'queued'}), 202
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@api_bp.route('/sa_status/<job_id>', methods=['GET'])
def sa_job_status(job_id):
    response = get_background_job_response(job_id)
    if response is None:
        return jsonify({'error': 'Analysis job not found or expired.'}), 404
    return jsonify(clean_json_data(response))


@api_bp.route('/sa_cancel/<job_id>', methods=['POST'])
def cancel_sa_job(job_id):
    response = cancel_background_job(job_id)
    if response is None:
        return jsonify({'error': 'Analysis job not found or expired.'}), 404
    return jsonify(response)


@api_bp.route('/job_status/<job_id>', methods=['GET'])
def background_job_status(job_id):
    response = get_background_job_response(job_id)
    if response is None:
        return jsonify({'error': 'Background job not found or expired.'}), 404
    return jsonify(clean_json_data(response))


@api_bp.route('/job_cancel/<job_id>', methods=['POST'])
def cancel_background_job_endpoint(job_id):
    response = cancel_background_job(job_id)
    if response is None:
        return jsonify({'error': 'Background job not found or expired.'}), 404
    return jsonify(response)


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
