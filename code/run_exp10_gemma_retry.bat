@echo off
rem Exp 10 — gemma-only retry. The JSON already holds 7 of 8 models; this re-runs ONLY
rem gemma-4-e2b (merge logic keeps the other seven), then recomputes the cross-model
rem analysis for all eight and writes exp10_*.png into data\. ~3-5 min.
rem BEFORE RUNNING: nvidia-smi — GPU 1 needs >= 12.5 GiB free right now. Last run saw only
rem 6.88 GiB at gemma's turn (something held ~9 GiB even with LM Studio closed).
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp10_activation_structure_8m.py gemma-4-e2b > data\exp10_gemma_retry.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp10_gemma_retry.log
) else (
    echo OK: gemma merged + cross-model analysis for all eight recomputed; PNGs in data\.
)
powershell -NoProfile -Command "Get-Content 'data\exp10_gemma_retry.log' -Tail 40"
pause
