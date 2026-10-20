@echo off
rem Exp5 (bootstrap) — pythia RETRY after the GPTNeoX naming fix. The first p.2 run left a
rem DEGRADED pythia entry (reference neck=None, match_frac=n/a); this rewrites ONLY that key
rem in data\exp5_neck_bootstrap_metrics.json (incremental merge). phi-1_5 stays untouched.
rem PREREQUISITE: run_exp5_anatomy_pythia_retry.bat must have finished first
rem (reference_neck comes from data\exp5_metrics.json), N_BOOT=25 random windows.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 10 GiB free on GPU 1. Takes ~5-8 min (one model x 25 boots).
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp5_neck_bootstrap.py pythia-2.8b 25 > data\exp5_neck_bootstrap_pythia_retry.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp5_neck_bootstrap_pythia_retry.log
) else (
    echo Script finished OK. Results merged into: data\exp5_neck_bootstrap_metrics.json
)
powershell -NoProfile -Command "Get-Content 'data\exp5_neck_bootstrap_pythia_retry.log' -Tail 30"
pause
