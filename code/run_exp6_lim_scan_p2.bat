@echo off
rem Exp6 (Lost-in-the-Middle scan) — p.2 extension: needle-retrieval lift vs control at 5 positions,
rem ONLY pythia-2.8b + phi-1_5, L=2048 only (both max_position_embeddings = 2048), 2 reps per cell.
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 8 GiB free on GPU 1 (no_grad, last-position logits row only). Takes ~5-10 min for two models.
rem After this bat: IC confidence intervals are recomputed on CPU by the maintainer (exp6_ic_ci.py) — no action needed here.
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\run_exp6_lim_scan.py pythia-2.8b,phi-1_5 > data\exp6_lim_scan_p2_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\exp6_lim_scan_p2_run.log
) else (
    echo Script finished OK. Results merged into: data\exp6_metrics.json
)
powershell -NoProfile -Command "Get-Content 'data\exp6_lim_scan_p2_run.log' -Tail 30"
pause
