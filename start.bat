@echo off
rem Windows: double-click to build (first time only, a few minutes) and start the app.
cd /d "%~dp0"
docker compose up --build -d || (echo. & echo Docker is not running. Start Docker Desktop and try again. & pause & exit /b 1)
timeout /t 4 >nul
start http://localhost:8080
echo Video to 3D is running at http://localhost:8080  (stop it with: docker compose down)
pause
