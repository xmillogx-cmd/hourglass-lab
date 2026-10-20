@echo off
rem Exp5 (bootstrap) — stability of the neck-level argmin under corpus resampling (GPU 1): all six models, N_BOOT random windows each.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 14 GiB free on GPU 1. Takes ~20-35 min (default 25 boots x 6 models; output_hidden_states is memory-light but not instant).
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp5_neck_bootstrap.py > data\exp5_neck_bootstrap_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp5_neck_bootstrap_run.log
) else (
    echo Script finished OK. Results: data\exp5_neck_bootstrap_metrics.json
)
powershell -NoProfile -Command "Get-Content 'data\exp5_neck_bootstrap_run.log' -Tail 30"
pause
