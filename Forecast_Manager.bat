@echo off
REM ======================================================================
REM  Forecast Manager — double-clic pour lancer
REM  Démarre le serveur local et ouvre automatiquement le navigateur
REM ======================================================================
cd /d "%~dp0"
title Forecast Manager (ne pas fermer cette fenetre)
echo.
echo  ============================================================
echo   Forecast Manager - Programmation triennale
echo  ============================================================
echo.
echo  Le navigateur va s'ouvrir automatiquement sur :
echo     http://127.0.0.1:5000
echo.
echo  NE FERMEZ PAS cette fenetre tant que vous utilisez l'application.
echo  Pour quitter : fermez d'abord le navigateur, puis cette fenetre.
echo.
".venv\Scripts\python.exe" "v2\code\manager_app.py"
pause
