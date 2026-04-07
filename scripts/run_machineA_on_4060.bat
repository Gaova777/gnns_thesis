@echo off
REM Machine A configs en RTX 4060 (Windows) — GAT + TAGCN, 72 configs
REM Detecta VRAM y ajusta GNNShap automaticamente.
REM Resultados en results_machineA_4060\ (separado de results_machineA\)
REM
REM Corre setup_windows.ps1 PRIMERO si es la primera vez.

setlocal enabledelayedexpansion

set MACHINE=A-4060
set BASE_CONFIG=configs/experiment_machineA.yaml
set LOG_DIR=logs
set LOG_FILE=%LOG_DIR%\machineA_4060_windows.log
set OVERRIDE_CONFIG=%TEMP%\experiment_machineA_4060_override.yaml

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.
echo ================================================================
echo   XAI-GNN Pipeline ^| Machine A en 4060 ^| Windows Launcher
echo   %DATE% %TIME%
echo ================================================================

REM ── Verificar CUDA y VRAM ────────────────────────────────────────────────────
echo.
echo [INFO] Verificando GPU y VRAM...

set VENV_PYTHON=.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" set VENV_PYTHON=python

set SHAP_SAMPLES=25
for /f "tokens=*" %%i in ('"%VENV_PYTHON%" -c "import torch, sys; p=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; vram=round(p.total_memory/1024**3,1) if p else 0; print(p.name if p else 'CPU'); print(vram); samples=50 if vram>=20 else (25 if vram>=6 else 0); print(samples)" 2^>^&1') do (
    set OUTPUT=%%i
)

REM Detectar VRAM con Python y ajustar SHAP_SAMPLES
"%VENV_PYTHON%" -c ^
"import torch, sys; ^
p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; ^
vram = round(p.total_memory/1024**3, 1) if p else 0; ^
print('[OK]   GPU: ' + (p.name if p else 'CPU')); ^
print('[OK]   VRAM: ' + str(vram) + ' GB'); ^
samples = 50 if vram >= 20 else (25 if vram >= 6 else 0); ^
print('SHAP_SAMPLES=' + str(samples)); ^
sys.exit(0 if samples > 0 else 1)" 2>&1 > %TEMP%\gpu_check.txt
type %TEMP%\gpu_check.txt

REM Leer SHAP_SAMPLES del output
for /f "tokens=2 delims==" %%s in ('findstr "SHAP_SAMPLES" %TEMP%\gpu_check.txt') do (
    set SHAP_SAMPLES=%%s
)

if "%SHAP_SAMPLES%"=="0" (
    echo [ERR] VRAM insuficiente para GAT/TAGCN. Se necesitan al menos 6GB.
    echo       Usa run_machineC.bat en su lugar (GCN+SAGE, mas liviano^).
    pause
    exit /b 1
)

echo [INFO] GNNShap samples: %SHAP_SAMPLES% (auto-seleccionado segun VRAM^)
echo.
echo Configs a correr: 72 (GAT+TAGCN x 4 escenarios x 3 balanceos x 3 explainers^)
echo Resultados en: results_machineA_4060\
echo.

set /p CONFIRM="Lanzar pipeline? [Y/n]: "
if /i "%CONFIRM%"=="n" ( echo Cancelado. & exit /b 0 )

REM ── Generar YAML con override ─────────────────────────────────────────────────
"%VENV_PYTHON%" -c ^
"import yaml; ^
f=open('%BASE_CONFIG%'); cfg=yaml.safe_load(f); f.close(); ^
[m.update({'num_samples': %SHAP_SAMPLES%}) for m in cfg['explainability']['methods'] if m['name']=='GNNShap']; ^
cfg['tracking']['results_dir']='./results_machineA_4060'; ^
cfg['tracking']['experiment_name']='xai-gnn-stability-A-4060'; ^
open('%OVERRIDE_CONFIG%','w').write(yaml.dump(cfg, default_flow_style=False))" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERR] No se pudo generar el YAML de override.
    pause
    exit /b 1
)
echo [INFO] YAML override generado: %OVERRIDE_CONFIG%

REM ── Intentar WSL2 primero ────────────────────────────────────────────────────
echo.
wsl --status >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] WSL2 detectado — lanzando script bash completo
    REM Convertir ruta Windows a ruta WSL para el override
    for /f "tokens=*" %%p in ('wsl wslpath "%OVERRIDE_CONFIG%"') do set WSL_OVERRIDE=%%p
    wsl bash -c "BASE_CONFIG=%BASE_CONFIG% bash scripts/run_machineA_on_4060.sh" 2>&1 | tee "%LOG_FILE%"
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
    --config "%OVERRIDE_CONFIG%" ^
    --resume ^
    --inter-config-pause 5 ^
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
del "%OVERRIDE_CONFIG%" 2>nul
echo.
echo ================================================================
echo   Pipeline Machine %MACHINE% finalizado ^| %DATE% %TIME%
echo ================================================================
endlocal
exit /b %EXIT_CODE%
