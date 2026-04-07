@echo off
REM Machine A launcher (Windows) — RTX 4090 (24GB)
REM GAT + TAGCN, 4 escenarios, 3 balanceos = 72 configs
REM
REM Corre setup_windows.ps1 PRIMERO si es la primera vez.
REM
REM Preferencia de ejecucion:
REM   1. WSL2  →  scripts completos con watchdog, lock file, auto-restart
REM   2. Python nativo Windows  →  pipeline basico con --resume

setlocal enabledelayedexpansion

set MACHINE=A
set CONFIG=configs/experiment_machineA.yaml
set LOG_DIR=logs
set LOG_FILE=%LOG_DIR%\machineA_windows.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.
echo ================================================================
echo   XAI-GNN Pipeline ^| Machine %MACHINE% ^| Windows Launcher
echo   %DATE% %TIME%
echo ================================================================

REM ── Verificar CUDA antes de lanzar ──────────────────────────────────────────
echo.
echo [INFO] Verificando GPU y CUDA...
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK]   nvidia-smi encontrado:
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader 2>&1
) else (
    echo [WARN] nvidia-smi no encontrado. Verifica drivers NVIDIA.
    echo        Corre setup_windows.ps1 para diagnostico completo.
)

REM Verificar torch CUDA
set VENV_PYTHON=.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" set VENV_PYTHON=python

"%VENV_PYTHON%" -c "import torch; ok=torch.cuda.is_available(); print('[OK]   torch CUDA disponible: ' + str(ok) + '  |  ' + str(torch.version.cuda) if ok else '[WARN] torch sin CUDA — se usara CPU')" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] torch no instalado. Corre: powershell -File scripts\setup_windows.ps1
    pause
    exit /b 1
)

echo.
echo Configs a correr: 72 (GAT+TAGCN x 4 escenarios x 3 balanceos x 3 explainers^)
echo.

REM ── Confirmacion ─────────────────────────────────────────────────────────────
set /p CONFIRM="Lanzar pipeline? [Y/n]: "
if /i "%CONFIRM%"=="n" ( echo Cancelado. & exit /b 0 )

REM ── Intentar WSL2 primero ────────────────────────────────────────────────────
echo.
wsl --status >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] WSL2 detectado — lanzando script bash completo (watchdog + lock + restart^)
    echo [INFO] Log: %LOG_FILE%
    wsl bash scripts/run_machineA.sh 2>&1 | tee "%LOG_FILE%"
    goto :done
)

REM ── Fallback: Python nativo ──────────────────────────────────────────────────
echo [WARN] WSL2 no disponible. Corriendo pipeline nativo (sin watchdog ni lock file^).
echo [WARN] Si quieres las funciones de robustez completas, instala WSL2:
echo         wsl --install   (en PowerShell como Admin^)
echo.

set MAX_RETRIES=3
set RETRY=0

:retry_loop
set /a ATTEMPT=RETRY+1
echo [INFO] Intento %ATTEMPT%/%MAX_RETRIES%...

set PYTHONPATH=.
"%VENV_PYTHON%" scripts\run_full_pipeline.py ^
    --config %CONFIG% ^
    --resume ^
    --device auto 2>&1 | tee -a "%LOG_FILE%"

set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% EQU 0 goto :done

set /a RETRY=RETRY+1
if %RETRY% LSS %MAX_RETRIES% (
    echo [INFO] Exit %EXIT_CODE% — reintentando en 30s (intento %RETRY%/%MAX_RETRIES%^)...
    timeout /t 30 /nobreak >nul
    goto :retry_loop
)

echo [ERR] Max reintentos alcanzados. Revisa %LOG_FILE%

:done
echo.
echo ================================================================
echo   Pipeline Machine %MACHINE% finalizado ^| %DATE% %TIME%
echo ================================================================
endlocal
exit /b %EXIT_CODE%
