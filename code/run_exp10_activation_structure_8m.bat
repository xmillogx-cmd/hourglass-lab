@echo off
rem Exp 10 — cross-model activation-structure comparison on ONE common control group.
rem All 8 local models, identical input: wikitext-2 test split (the quantization corpus),
rem first 2048 tokens each. Per-level PRc / anisotropy / k95 vs depth fraction + pairwise
rem cross-model analysis. Verdict criteria are pre-declared in the plan board v0.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 8 GiB free on GPU 1 (gemma-4-e2b needs >= 12.5). Takes ~15-30 min for all eight models.
rem PYTHON = python on PATH unless overridden via environment; Gemma runs need a venv with transformers>=5 (set PYTHON).
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp10_activation_structure_8m.py > data\exp10_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp10_run.log
) else (
    echo Script finished OK. Results in: data\exp10_activation_structure_8m.json + exp10_*.png
)
powershell -NoProfile -Command "Get-Content 'data\exp10_run.log' -Tail 40"
pause
