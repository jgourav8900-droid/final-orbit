@echo off
title Orbit Overload - Full-Stack Localhost Server (Port 8000)
cd /d "%~dp0"
echo Starting Orbit Overload (FastAPI + Three.js + NASA + Gemini) on Port 8000...
python run_server.py
pause
