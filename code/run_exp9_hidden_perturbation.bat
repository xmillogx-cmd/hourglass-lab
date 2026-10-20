@echo off
rem Exp9 — hidden-state perturbation / layer-sensitivity scan (bf16, GPU 1): gemma-4-e2b + qwen2.5-0.5b + tinyllama-1.1b.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 14 GiB free on GPU 1. Takes ~8-10 min. OOM now falls back L=1024 -> 512 -> 256 even mid-scan.
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp9_hidden_perturbation.py > data\exp9_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp9_run.log
) else (
    echo Script finished OK. Results: data\exp9_hidden_perturbation_metrics.json
)
powershell -NoProfile -Command "Get-Content 'data\exp9_run.log' -Tail 30"
pause
