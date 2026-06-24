"""
tests/test_api.py
==================
Integration tests for the Flask API endpoints.
Uses Flask's built-in test client to verify that each route returns the
expected HTTP status codes and JSON response structures.
"""

import pytest
import json
import io
import sys
import os

# Ensure the app root is on the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app as flask_app


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def client():
    """Create a Flask test client with testing mode enabled."""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def sample_doe_payload():
    """A minimal valid DoE request payload."""
    return {
        'features': [
            {'name': 'A', 'type': 'continuous', 'range': '[0, 10]'},
            {'name': 'B', 'type': 'continuous', 'range': '[5, 20]'},
            {'name': 'C', 'type': 'continuous', 'range': '[100, 200]'},
        ],
        'tweaks': {'model': 'bbdesign', 'max_runs': 20}
    }


@pytest.fixture
def sample_csv_data():
    """A small in-memory CSV payload representing a 3-row dataset."""
    return (
        b"A,B,C,Y\n"
        b"1,5,100,0.5\n"
        b"5,12,150,0.8\n"
        b"9,18,190,0.3\n"
    )


# ==============================================================================
# Test: Static Pages
# ==============================================================================

class TestStaticPages:
    """Tests that the main application pages serve correctly."""

    def test_index_returns_200(self, client):
        """GET / should return HTTP 200."""
        resp = client.get('/')
        assert resp.status_code == 200

    def test_guide_returns_200(self, client):
        """GET /guide should return HTTP 200."""
        resp = client.get('/guide')
        assert resp.status_code == 200


# ==============================================================================
# Test: /upload Endpoint
# ==============================================================================

class TestUploadEndpoint:
    """Tests for the CSV file upload endpoint."""

    def test_upload_valid_csv(self, client, sample_csv_data):
        """POST /upload with a valid CSV should return data and columns."""
        data = {
            'file': (io.BytesIO(sample_csv_data), 'test.csv', 'text/csv')
        }
        resp = client.post('/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 200

        result = resp.get_json()
        assert 'columns' in result
        assert 'data' in result
        assert set(result['columns']) == {'A', 'B', 'C', 'Y'}

    def test_upload_no_file_returns_400(self, client):
        """POST /upload with no file attached should return HTTP 400."""
        resp = client.post('/upload', data={}, content_type='multipart/form-data')
        assert resp.status_code == 400

        result = resp.get_json()
        assert 'error' in result

    def test_upload_empty_filename_returns_400(self, client):
        """POST /upload with an empty filename should return HTTP 400."""
        data = {
            'file': (io.BytesIO(b''), '', 'text/csv')
        }
        resp = client.post('/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 400


# ==============================================================================
# Test: /doe Endpoint
# ==============================================================================

class TestDoeEndpoint:
    """Tests for the DoE generation endpoint."""

    def test_doe_returns_suggestions_and_metrics(self, client, sample_doe_payload):
        """POST /doe with a valid payload should return 'suggestions' and 'metrics'."""
        resp = client.post(
            '/doe',
            data=json.dumps(sample_doe_payload),
            content_type='application/json'
        )
        assert resp.status_code == 200

        result = resp.get_json()
        assert 'suggestions' in result
        assert 'metrics' in result
        assert len(result['suggestions']) > 0

    def test_doe_invalid_model_fallback(self, client):
        """POST /doe with only 1 feature should return HTTP 500 with error message."""
        payload = {
            'features': [
                {'name': 'A', 'type': 'continuous', 'range': '[0, 10]'},
            ],
            'tweaks': {'model': 'bbdesign'}
        }
        resp = client.post(
            '/doe',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert resp.status_code == 500
        result = resp.get_json()
        assert 'error' in result



