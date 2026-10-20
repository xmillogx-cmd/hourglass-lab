@echo off
rem Exp 10c stage A — GPU activation dump for the structural-geometry analysis (stage B).
rem All 8 local models, identical input: wikitext-2 test split (the quantization corpus),
rem first 2048 tokens each. Dumps raw per-level hidden states to results/data/exp10c_act_*.npz
rem (~1.5-2 GB total) + token surface forms — stage B (CPU) then computes CKA / Procrustes / PCA.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 8 GiB free on GPU 1 (gemma-4-e2b needs >= 12.5). Takes ~20-40 min for all eight models.
rem PYTHON = python on PATH unless overridden via environment; Gemma runs need a venv with transformers>=5 (set PYTHON).
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp10c_activation_dump.py > data\exp10c_dump.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp10c_dump.log
) else (
    echo Script finished OK. Dumps in: data\exp10c_act_*.npz + exp10c_tokens_*.json
)
powershell -NoProfile -Command "Get-Content 'data\exp10c_dump.log' -Tail 40"
pause
