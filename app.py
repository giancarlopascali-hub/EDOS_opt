"""
EDOS (Experimental Design & Optimization System) - Flask Backend Bootstrap
================================================================
Main application entry point. Resolves static/template resources for development
or standalone PyInstaller builds, registers blueprints, and starts the local server.
"""

import os
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

from flask import Flask
from src.api.routes import api_bp

# Initialize the Flask App
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

# Register the routes blueprint
app.register_blueprint(api_bp)

# Ensure template/static directories exist when running in dev mode
if not getattr(sys, 'frozen', False):
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

if __name__ == '__main__':
    import threading
    import webbrowser
    import time

    # Port selection: Standalone matches README (5000), deployment uses environment PORT
    default_port = 5000
    if not getattr(sys, 'frozen', False):
        default_port = int(os.environ.get("PORT", 7860))

    port = int(os.environ.get("PORT", default_port))
    url  = f"http://127.0.0.1:{port}"

    print(f"\n" + "=" * 40)
    print(f"       EDOS v6.0 STANDALONE")
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
