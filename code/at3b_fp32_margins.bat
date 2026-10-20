@echo off
rem AT-3b — fp32 rerun of the AT-3 margin dumps (CPU pass): all 8 models forced to torch.float32.
rem Fixes the broken at3_margins_*.npz ((1,2)-shape bug) and resolves the dtype-vs-stability confound:
rem per-layer delta grid {1e-4, 3e-4, 7e-4, 1.3e-3}; margins + top-k dumps + flip rates only (no U6/KL).
rem CPU run — no GPU/VRAM needed; needs ~25 GB free RAM at peak (gemma fp32 weights ~20 GB).
rem Measured CPU speed (L=512 fp32): smollm 1.2 s/fwd, qwen-0.5b 2.8 s/fwd -> all 8 models take
rem ~45-60 min (33 forwards per model; no U6/KL in this pass). Checkpointed: reruns skip completed models.
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <repo root>\models (or $MODELS_DIR).
rem Gemma-4-E2B needs transformers >= 5.x - for those runs set PYTHON to your venv's python.exe.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_at3b_fp32_margins.py > data\at3b_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\at3b_run.log
) else (
    echo Script finished OK. Results: data\at3b_fp32_margins_metrics.json + data\at3b_margins_*.npz
)
powershell -NoProfile -Command "Get-Content 'data\at3b_run.log' -Tail 30"
pause
