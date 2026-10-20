@echo off
rem Functional impact experiment for Gemma-4-E2B (Part B on GPU bf16 + Part A fp32 CPU).
rem BEFORE RUNNING: close LM Studio to free VRAM (need >= 14 GiB free on one card).
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_gemma_functional_impact.py
echo.
echo Script finished (exit code %errorlevel%). Results: data\gemma_functional_impact_metrics.json
pause
