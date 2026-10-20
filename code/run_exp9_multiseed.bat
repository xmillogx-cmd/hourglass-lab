@echo off
rem Exp9 (multi-seed) — KL(rel-eps) curves with 10 perturbations/layer, eps normalized on hidden_rms: gemma/qwen2.5-0.5b/tinyllama.
rem BEFORE RUNNING: close LM Studio FULLY - check the tray, it may keep a background instance holding VRAM.
rem Need >= 14 GiB free on GPU 1. Takes ~90-120 min (3 models x per-layer x 3 eps levels x 10 perturbations; gemma is the slow one).
rem PYTHON overridable via environment; default: python on PATH.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp9_multiseed.py > data\exp9_multiseed_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED - log tail below, full: data\exp9_multiseed_run.log
) else (
    echo Script finished OK. Results: data\exp9_multiseed_metrics.json
)
powershell -NoProfile -Command "Get-Content 'data\exp9_multiseed_run.log' -Tail 30"
pause
