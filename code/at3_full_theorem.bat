@echo off
rem AT-3 — full empirical verification of Nikitin's analogy theorem (all statements: U4/U5/U6,
rem A8-A11, regularity 1-6, FOL 3/4 + SWD stage inputs) on the diverse-doc corpus: all 8 models, GPU 1.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 14 GiB free on GPU 1. Takes ~35-50 min (8 models x 8 layers x {2 singles + 2 pair-scales x 2 arms} + dumps).
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <repo root>\models (or $MODELS_DIR).
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_at3_full_theorem.py > data\at3_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\at3_run.log
) else (
    echo Script finished OK. Results: data\at3_full_theorem_metrics.json + data\at3_base_logits_*.npz / at3_noisy_last_*.npz / at3_margins_*.npz
)
powershell -NoProfile -Command "Get-Content 'data\at3_run.log' -Tail 30"
pause
