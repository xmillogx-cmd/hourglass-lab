@echo off
rem Exp7 (multi-rep) — statistical power of the gradient-interference geometry (GPU 1): all six models, 5 random corpus windows each.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 14 GiB free on GPU 1. Takes ~20-40 min (5 reps x 6 models; gemma-4-e2b is the slow one).
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp7_multisubsample.py > data\exp7_multisubsample_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp7_multisubsample_run.log
) else (
    echo Script finished OK. Results: data\exp7_multisubsample_metrics.json
)
powershell -NoProfile -Command "Get-Content 'data\exp7_multisubsample_run.log' -Tail 30"
pause
