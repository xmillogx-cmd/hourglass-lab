@echo off
rem Gemma-4-E2B gradient-direction functional perturbation sweep (fp32, CPU): G_in/G_out unit directions x 5 epsilons x +/- signs.
rem CPU-only run — no GPU needed; ~18.5 GB RAM for the model in fp32. Takes ~20-40 min (one backward + 20 perturbed forwards).
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\exp_gemma_gradient_perturbation.py > data\gemma_gradient_perturbation_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\gemma_gradient_perturbation_run.log
) else (
    echo Script finished OK. Results: data\gemma_gradient_perturbation_metrics.json + data\gemma_gradient_perturbation.png
)
powershell -NoProfile -Command "Get-Content 'data\gemma_gradient_perturbation_run.log' -Tail 30"
pause
