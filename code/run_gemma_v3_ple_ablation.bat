@echo off
rem V3 ablation for Gemma-4-E2B: scale-stripped embedding + zeroed PLE identity table (GPU bf16).
rem BEFORE RUNNING: close LM Studio to free VRAM (need >= 14 GiB free on one card). Takes ~5 min.
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_gemma_v3_ple_ablation.py
echo.
echo Script finished (exit code %errorlevel%). Results: data\gemma_v3_ple_ablation_metrics.json
pause
