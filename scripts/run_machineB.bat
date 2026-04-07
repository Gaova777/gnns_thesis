@echo off
REM Machine B launcher (Windows) — RTX 4060 (8GB)
REM GCN + GraphSAGE, escenarios 1:1 y 1:10 = 36 configs
REM
REM Corre setup_windows.ps1 PRIMERO si es la primera vez.

setlocal enabledelayedexpansion

set MACHINE=B
set CONFIG=configs/experiment_machineB.yaml
set LOG_DIR=logs
set LOG_FILE=%LOG_DIR%\machineB_windows.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.
echo ================================================================
echo   XAI-GNN Pipeline ^| Machine %MACHINE% ^| Windows Launcher
echo   %DATE% %TIME%
echo ================================================================

REM ── Verificar CUDA ───────────────────────────────────────────────────────────
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

set VENV_PYTHON=.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" set VENV_PYTHON=python

"%VENV_PYTHON%" -c "import torch; ok=torch.cuda.is_available(); print('[OK]   torch CUDA disponible: ' + str(ok) + '  |  ' + str(torch.version.cuda) if ok else '[WARN] torch sin CUDA — se usara CPU')" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] torch no instalado. Corre: powershell -File scripts\setup_windows.ps1
    pause
    exit /b 1
)

echo.
echo Configs a correr: 36 (GCN+SAGE x escenarios 1:1,1:10 x 3 balanceos x 3 explainers^)
echo.

set /p CONFIRM="Lanzar pipeline? [Y/n]: "
if /i "%CONFIRM%"=="n" ( echo Cancelado. & exit /b 0 )

REM ── Intentar WSL2 primero ────────────────────────────────────────────────────
echo.
wsl --status >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] WSL2 detectado — lanzando script bash completo
    wsl bash scripts/run_machineB.sh 2>&1 | tee "%LOG_FILE%"
    goto :done
)

REM ── Fallback: Python nativo ──────────────────────────────────────────────────
echo [WARN] WSL2 no disponible. Pipeline nativo sin watchdog.
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
    echo [INFO] Exit %EXIT_CODE% — reintentando en 30s...
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
