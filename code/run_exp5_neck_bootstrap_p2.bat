@echo off
rem Exp5 (bootstrap) — p.2 extension: stability of the neck-level argmin under corpus resampling,
rem ONLY pythia-2.8b + phi-1_5, N_BOOT=25 random windows each.
rem PREREQUISITE: run_exp5_anatomy_p2.bat must have finished first (reference_neck comes from data\exp5_metrics.json).
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 10 GiB free on GPU 1 (no_grad + per-level CPU transfer). Takes ~5-10 min for two models x 25 boots.
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp5_neck_bootstrap.py pythia-2.8b,phi-1_5 25 > data\exp5_neck_bootstrap_p2_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp5_neck_bootstrap_p2_run.log
) else (
    echo Script finished OK. Results merged into: data\exp5_neck_bootstrap_metrics.json
)
powershell -NoProfile -Command "Get-Content 'data\exp5_neck_bootstrap_p2_run.log' -Tail 30"
pause
