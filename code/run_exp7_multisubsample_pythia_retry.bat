@echo off
rem Exp7 (multi-rep gradient interference) — pythia RETRY after the GPTNeoX naming fix.
rem The first p.2 run SILENTLY SKIPPED pythia here; phi-1_5 results are valid and untouched —
rem this rewrites ONLY the pythia key in data\exp7_multisubsample_metrics.json (incremental merge).
rem Untied baseline protocol: 5 random corpus windows, cos(g_emb, g_lm) + norm ratio per window.
rem SUCCESS CRITERION in the log tail: a "pythia-2.8b UNTIED cos(G_emb,G_lm)=..." line and NO "SKIPPED" line.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 14 GiB free on GPU 1 (autograd through the whole model). Takes ~8-15 min (one model x 5 reps).
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp7_multisubsample.py pythia-2.8b 5 > data\exp7_multisubsample_pythia_retry.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp7_multisubsample_pythia_retry.log
) else (
    echo Script finished OK. Results merged into: data\exp7_multisubsample_metrics.json
)
powershell -NoProfile -Command "Get-Content 'data\exp7_multisubsample_pythia_retry.log' -Tail 30"
pause
