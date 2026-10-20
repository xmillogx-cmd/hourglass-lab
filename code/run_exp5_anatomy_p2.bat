@echo off
rem Exp5 (anatomy) — p.2 extension: embedding anisotropy + centered PR profile / neck level,
rem ONLY the two new untied models: pythia-2.8b (bf16), phi-1_5 (fp32).
rem RUN FIRST of the four p.2 bats: run_exp5_neck_bootstrap_p2.bat needs the reference_neck this writes.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 10 GiB free on GPU 1 (no_grad + per-level CPU transfer). Takes ~5-10 min for two models.
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp5_anatomy.py pythia-2.8b,phi-1_5 > data\exp5_anatomy_p2_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp5_anatomy_p2_run.log
) else (
    echo Script finished OK. Results merged into: data\exp5_metrics.json
)
powershell -NoProfile -Command "Get-Content 'data\exp5_anatomy_p2_run.log' -Tail 30"
pause
