"""
EDOS (Experimental Design & Optimization System) - Flask Backend Bootstrap
================================================================
Main application entry point. Resolves static/template resources for development
or standalone PyInstaller builds, registers blueprints, and starts the local server.
"""

import os
import json
import socket
import sys
import warnings

# Suppress expected BoTorch/GPyTorch runtime warnings
warnings.filterwarnings("ignore", message=".*The model inputs are of type torch.float32.*")
warnings.filterwarnings("ignore", message=".*Optimization failed in `gen_candidates_scipy`.*")

# Resolve paths for PyInstaller standalone vs. Dev Mode
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS          # PyInstaller temp dir
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR   = os.path.join(BASE_DIR, 'static')

# Secondary path check for --onedir bundles placing resources inside _internal/
if not os.path.exists(os.path.join(TEMPLATE_DIR, 'index.html')):
    _internal = os.path.join(BASE_DIR, '_internal')
    if os.path.exists(os.path.join(_internal, 'templates')):
        TEMPLATE_DIR = os.path.join(_internal, 'templates')
        STATIC_DIR   = os.path.join(_internal, 'static')

# Setup paths inside Python path for absolute imports inside src/
sys.path.insert(0, BASE_DIR)

from flask import Flask, jsonify
from src.api.routes import api_bp

# Initialize the Flask App
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

# Register the routes blueprint
app.register_blueprint(api_bp)


def _deployment_info():
    info = {
        'application': 'EDOS',
        'application_commit': 'development',
        'edos_foundation_commit': 'development',
        'deployed_at': None,
    }
    path = os.path.join(BASE_DIR, 'build-info.json')
    try:
        with open(path, encoding='utf-8') as handle:
            info.update(json.load(handle))
    except (OSError, ValueError, TypeError):
        pass
    return info


@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'application': 'EDOS'})


@app.get('/version')
def version():
    return jsonify(_deployment_info())

# Ensure template/static directories exist when running in dev mode
if not getattr(sys, 'frozen', False):
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

if __name__ == '__main__':
    import multiprocessing
    import threading
    import webbrowser
    import time

    multiprocessing.freeze_support()

    # Port selection: Standalone matches README (5000), deployment uses environment PORT
    default_port = 5000
    if not getattr(sys, 'frozen', False):
        default_port = int(os.environ.get("PORT", 7860))

    def _available_local_port(preferred_port):
        """Use the preferred local port unless another local app owns it."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('127.0.0.1', preferred_port))
                return preferred_port
            except OSError:
                probe.bind(('127.0.0.1', 0))
                return probe.getsockname()[1]

    requested_port = int(os.environ.get("PORT", default_port))
    port = requested_port if os.environ.get("PORT") else _available_local_port(requested_port)
    url  = f"http://127.0.0.1:{port}"

    print(f"\n" + "=" * 40)
    print(f"       EDOS v7.1 STANDALONE")
    print(f"=" * 40)
    print(f"Server is starting...")
    print(f"Local URL: {url}")
    print(f"Attempting to open browser automatically...")
    print(f"=" * 40 + "\n")

    def open_browser():
        time.sleep(2.0)
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Error opening browser: {e}")

    # Auto-open browser only on standalone builds or local dev on port 5000
    if getattr(sys, 'frozen', False) or port == 5000:
        threading.Thread(target=open_browser, daemon=True).start()

    host = '0.0.0.0' if os.environ.get("PORT") else '127.0.0.1'
    app.run(host=host, port=port, debug=False, use_reloader=False)
