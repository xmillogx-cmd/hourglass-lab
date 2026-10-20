@echo off
rem Exp7 (multi-rep gradient interference) — p.2 extension: 5 random corpus windows each,
rem ONLY pythia-2.8b + phi-1_5 — both UNTIED, so this runs the untied baseline protocol
rem (cos(g_emb, g_lm) + norm ratio), exactly as for tinyllama/qwen.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 14 GiB free on GPU 1 (autograd through the whole model; pythia bf16 is the heavier one).
rem Takes ~10-20 min for two models x 5 reps.
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp7_multisubsample.py pythia-2.8b,phi-1_5 5 > data\exp7_multisubsample_p2_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp7_multisubsample_p2_run.log
) else (
    echo Script finished OK. Results merged into: data\exp7_multisubsample_metrics.json
)
powershell -NoProfile -Command "Get-Content 'data\exp7_multisubsample_p2_run.log' -Tail 30"
pause
