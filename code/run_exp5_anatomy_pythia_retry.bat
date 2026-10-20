@echo off
rem Exp5 (anatomy) — pythia RETRY after the GPTNeoX naming fix (embed_in / gpt_neox stack).
rem The first p.2 run SILENTLY SKIPPED pythia here; phi-1_5 results are valid and untouched —
rem this rewrites ONLY the pythia key in data\exp5_metrics.json (incremental merge).
rem RUN FIRST of the three retry bats: bootstrap needs the reference_neck this writes.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 10 GiB free on GPU 1. Takes ~3-5 min (one model).
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp5_anatomy.py pythia-2.8b > data\exp5_anatomy_pythia_retry.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp5_anatomy_pythia_retry.log
) else (
    echo Script finished OK. Results merged into: data\exp5_metrics.json
)
powershell -NoProfile -Command "Get-Content 'data\exp5_anatomy_pythia_retry.log' -Tail 30"
pause
