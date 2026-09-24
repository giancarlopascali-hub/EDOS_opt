import threading
import time

from src.api import jobs
from app import app as flask_app


def test_background_job_lifecycle(monkeypatch):
    monkeypatch.setattr(jobs, '_JOBS', {})
    completed = threading.Event()

    def worker(job_id, _payload):
        jobs.update_background_job(job_id, status='running', phase='Working', percent=50)
        jobs.update_background_job(
            job_id, status='completed', phase='Completed', percent=100,
            result={'ok': True}, finished_at=time.time(),
        )
        completed.set()

    job_id = jobs.queue_background_job('test', {}, worker, 'Queued')
    assert job_id
    assert completed.wait(2)
    response = jobs.get_background_job_response(job_id)
    assert response['kind'] == 'test'
    assert response['status'] == 'completed'
    assert response['result'] == {'ok': True}


def test_cancelled_job_exposes_cancellation_event(monkeypatch):
    monkeypatch.setattr(jobs, '_JOBS', {})
    started = threading.Event()
    release = threading.Event()

    def worker(job_id, _payload):
        started.set()
        release.wait(2)

    job_id = jobs.queue_background_job('test', {}, worker, 'Queued')
    assert started.wait(2)
    assert jobs.cancel_background_job(job_id)['status'] == 'cancelling'
    assert jobs.get_cancel_event(job_id).is_set()
    release.set()


def test_bo_surface_submit_rejects_rows_above_surface_limit():
    flask_app.config['TESTING'] = True
    payload = {
        'data': [[0, 0, 0]] * 651,
        'columns': ['x', 'y', 'objective'],
        'features': [
            {'name': 'x', 'type': 'continuous', 'range': '[0, 1]'},
            {'name': 'y', 'type': 'continuous', 'range': '[0, 1]'},
        ],
        'objectives': [{'name': 'objective', 'type': 'maximize'}],
    }
    with flask_app.test_client() as client:
        response = client.post('/bo_surface_submit', json=payload)
    assert response.status_code == 400
    assert 'limited to 650' in response.get_json()['error']
