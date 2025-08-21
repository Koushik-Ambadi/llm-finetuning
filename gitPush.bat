@echo off
REM === Navigate to project directory (folder where this script is placed) ===
cd /d %~dp0

REM === Stage all changes ===
git add .

REM === Ask user for commit message ===
set /p msg=Enter commit message: 

REM === Commit with user message ===
git commit -m "%msg%"

REM === Push to main branch ===
git push origin main

pause
