@echo off
cd /d "%~dp0.."
set USE_CPP_RENDER=1
python scripts/visualize_marl.py
