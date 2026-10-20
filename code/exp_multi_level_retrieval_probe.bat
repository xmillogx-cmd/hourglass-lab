@echo off
rem Qwen2.5-0.5B multi-level needle retrieval probe (bf16, GPU): 3 levels x 3 positions x 3 reps = 27 forwards of L=2048.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 8 GiB free on GPU 1. Takes ~5-10 min (filler build + 27 forwards + alt-code generation).
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\exp_multi_level_retrieval_probe.py > data\multi_level_retrieval_probe_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\multi_level_retrieval_probe_run.log
) else (
    echo Script finished OK. Results: data\multi_level_retrieval_probe_metrics.json + data\multi_level_retrieval_probe.png
)
powershell -NoProfile -Command "Get-Content 'data\multi_level_retrieval_probe_run.log' -Tail 30"
pause
