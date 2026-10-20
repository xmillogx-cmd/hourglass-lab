@echo off
rem p.2 reviewer extension — download EleutherAI/pythia-2.8b + microsoft/phi-1_5 (CPU only, no GPU).
rem ~8.4 GB total (fp16 weights). Network+disk only; safe to run with LM Studio open.
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\download_p2_models.py > data\p2_download.log 2>&1
echo.
if errorlevel 1 (
    echo Download FAILED — log tail below, full: data\p2_download.log
) else (
    echo Download finished OK. Models: ..\models\pythia-2.8b and ..\models\phi-1_5
)
powershell -NoProfile -Command "Get-Content 'data\p2_download.log' -Tail 30"
pause
