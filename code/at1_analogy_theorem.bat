@echo off
rem AT-1 — analogy-theorem (Nikitin) perturbation test on the diverse-doc corpus: all 8 models, GPU 1.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 14 GiB free on GPU 1. Takes ~20-35 min (8 models x 8 layers x 5 deltas). OOM falls back L=512 -> 256 mid-scan.
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <repo root>\models (or $MODELS_DIR).
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_at1_analogy_theorem.py > data\at1_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\at1_run.log
) else (
    echo Script finished OK. Results: data\at1_analogy_perturbation_metrics.json + data\at1_act_*.npz
)
powershell -NoProfile -Command "Get-Content 'data\at1_run.log' -Tail 30"
pause
