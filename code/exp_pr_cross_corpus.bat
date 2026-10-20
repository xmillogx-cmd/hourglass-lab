@echo off
rem Qwen2.5-0.5B participation-ratio profile across 5 local corpora (bf16, GPU).
rem BEFORE RUNNING: close LM Studio FULLY — check the tray, it may keep a background instance holding VRAM.
rem Need >= 8 GiB free on GPU 1. Takes ~5-10 min (5 forwards of 2048 tokens + per-layer SVD on CPU).
rem PYTHON / MODELS_DIR overridable via environment; defaults: python on PATH, models dir = <host tree>\models.
if not defined PYTHON set PYTHON=python
cd /d "%~dp0.."
"%PYTHON%" code\exp_pr_cross_corpus.py > data\pr_cross_corpus_run.log 2>&1
echo.
if errorlevel 1 (
    echo Script FAILED — log tail below, full: data\pr_cross_corpus_run.log
) else (
    echo Script finished OK. Results: data\pr_cross_corpus_metrics.json + data\pr_cross_corpus.png
)
powershell -NoProfile -Command "Get-Content 'data\pr_cross_corpus_run.log' -Tail 30"
pause
