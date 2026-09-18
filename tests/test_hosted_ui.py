import os
import unittest
from unittest.mock import patch

from app import app


class HostedUiTests(unittest.TestCase):
    def test_shutdown_button_is_visible_for_local_desktop(self):
        with patch.dict(os.environ, {}, clear=True), app.test_client() as client:
            response = client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="shutdown-btn"', response.data)

    def test_shutdown_button_and_endpoint_are_disabled_on_hugging_face(self):
        with patch.dict(os.environ, {'SPACE_ID': 'realgcp/EDOS-app'}, clear=True), app.test_client() as client:
            page = client.get('/')
            shutdown = client.post('/shutdown')
        self.assertNotIn(b'id="shutdown-btn"', page.data)
        self.assertEqual(shutdown.status_code, 403)

    def test_hugging_face_hostname_also_hides_shutdown_button(self):
        with patch.dict(os.environ, {}, clear=True), app.test_client() as client:
            response = client.get('/', base_url='https://example.hf.space')
        self.assertNotIn(b'id="shutdown-btn"', response.data)


if __name__ == '__main__':
    unittest.main()
