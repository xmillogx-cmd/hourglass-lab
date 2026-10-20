@echo off
rem Qwen2.5-0.5B neck-L4 attractor-subspace perturbation test (fp32 on GPU): 1 base forward + SVD top-10 subspace + 9 arm forwards.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 8 GiB free on GPU 1 (model in fp32 ~2 GB). Takes ~2-4 min.
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\exp_attractor_subspace_perturbation.py > data\attractor_subspace_perturbation_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\attractor_subspace_perturbation_run.log
) else (
    echo Script finished OK. Results: data\attractor_subspace_perturbation_metrics.json + data\attractor_subspace_perturbation.png
)
powershell -NoProfile -Command "Get-Content 'data\attractor_subspace_perturbation_run.log' -Tail 30"
pause
