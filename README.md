---
title: EDOS
emoji: 🧪
favicon: static/favicon.ico
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# EDOS

EDOS is the canonical shared foundation for the independently maintained
AutoEDOS and TransEDOS repositories. The published foundation revision and
shared-file contract are recorded under `alignment/`.

Every push to `main` must pass the full test suite before a clean bundle is
deployed to [realgcp/EDOS-app](https://huggingface.co/spaces/realgcp/EDOS-app).
The live application exposes `/health` and `/version` for deployment
verification and source-revision traceability.


**EDOS** (Experimental Design and Optimization System) is a powerful platform designed for researchers to perform **Design of Experiments (DoE)**, **Bayesian Optimization (BO)** and **Statistical Analysis (SA)** with ease.

## Key Features

- **DoE Module**: Generate Box-Behnken, Central Composite, Fractional Factorial, mixed-level Full Factorial, Definitive Screening, LHS, and D-Optimal designs. Reduced budgets prioritize unique combinations; Exact counts add flagged repeats only after unique conditions are exhausted.
- **Scientific Quality Metrics**: Real-time calculation of Orthogonality, D-Efficiency, Resolution, and Curvature detection with a visual "Quality Dashboard".
- **Bayesian Optimizer**: Suggest the best experiments to perform next and improve your peformance, visualizing objective trends and Pareto front. It supports categorical parameters and multi-objective optimization.
- **Statistical Analysis**: Perform basic analysis on experimental data, including feature importance, correlation, and success-based optimization.
- **Standalone Capability**: Bundled into a Windows executable for easy deployment.

## Installation and Running (Windows Standalone)

1. **Web-app**: You can use the most updated version of the app online on the [EDOS](https://realgcp-edos-app.hf.space/) page.
2. **Download**: You can download the latest standalone version from the [Releases](https://github.com/giancarlopascali-hub/EDOS_opt/releases) page. This version is not up to date as the online one, but gives the freedom of no internet connection required to run. New releases will be available in the future, with several fixes and improvement, to align with the online version.
3. **Extract**: Right-click the `EDOS_v4.0_Standalone.zip` file and select "Extract All...".
4. **Run**: 
   - Open the extracted folder.
   - **Note**: The application lives entirely within its folder structure; do not move the `.exe` file out of the `EDOS_v4.0` folder.
   - Double-click **`EDOS_v4.0.exe`**.

5. **Access**: The application will automatically start a local server. If your browser doesn't open automatically, go to `http://127.0.0.1:5000`.
6. **Shutdown**: To close the application properly, press the **"Shutdown"** button within the web interface before closing the window.


## Technical Stack


- **Backend**: Python (Flask, BoTorch, GPyTorch, Scikit-Learn, PyDOE)
- **Frontend**: HTML5, Vanilla CSS3, JavaScript (ES6+)
- **Packaging**: PyInstaller

## Developer
Giancarlo Pascali
UNSW, Sydney

---
*Created with the help of Antigravity AI.*
