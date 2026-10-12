import os
import unittest
from unittest.mock import patch

from app import app


class HostedUiTests(unittest.TestCase):
    def test_exploration_controls_start_disabled_for_default_ei(self):
        with app.test_client() as client:
            response = client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'exploration-control control-disabled', response.data)
        self.assertRegex(response.data, rb'id="style-slider"[^>]*disabled')
        self.assertRegex(response.data, rb'id="style-number"[^>]*disabled')

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

    def test_advanced_bo_controls_exist_but_start_hidden(self):
        with app.test_client() as client:
            response = client.get('/')
        self.assertIn(b'id="bo-mo-method-item" class="tweak-item hidden"', response.data)
        self.assertIn(b'id="bo-large-data-options" class="tweak-item full-width hidden"', response.data)
        self.assertIn(b'Auto (recommended)', response.data)

    def test_hosted_profile_is_exposed_to_disable_unsafe_full_mode(self):
        with patch.dict(os.environ, {'SPACE_ID': 'realgcp/autoEDOS'}, clear=True), app.test_client() as client:
            response = client.get('/')
        self.assertIn(b'data-hosted-environment="true"', response.data)


if __name__ == '__main__':
    unittest.main()
