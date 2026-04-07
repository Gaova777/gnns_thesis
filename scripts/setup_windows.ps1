# setup_windows.ps1 — Verificacion y setup de CUDA + dependencias
# Corre este script UNA VEZ en cada maquina Windows antes de lanzar el pipeline.
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#
# Que hace:
#   1. Verifica Python 3.12+
#   2. Verifica / instala uv
#   3. Detecta GPU NVIDIA y drivers
#   4. Verifica si torch tiene soporte CUDA
#   5. Si no: pregunta instalacion manual o automatica
#   6. Verifica WSL2 (recomendado para los scripts .sh)
#   7. Imprime resumen final

param(
    [switch]$SkipConfirm  # Para CI/automatizacion: no pedir confirmacion
)

$ErrorActionPreference = "Stop"

function Write-Header($text) {
    Write-Host ""
    Write-Host ("=" * 68) -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host ("=" * 68) -ForegroundColor Cyan
}

function Write-OK($text)   { Write-Host "  [OK]   $text" -ForegroundColor Green }
function Write-WARN($text) { Write-Host "  [WARN] $text" -ForegroundColor Yellow }
function Write-ERR($text)  { Write-Host "  [ERR]  $text" -ForegroundColor Red }
function Write-INFO($text) { Write-Host "  [INFO] $text" -ForegroundColor White }

$ts = { Get-Date -Format "yyyy-MM-dd HH:mm:ss" }

Write-Header "XAI-GNN Pipeline — Setup Windows"
Write-INFO "$(& $ts)"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Python 3.12+
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "1/6  Python"

$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -eq 3 -and $minor -ge 12) {
                $pythonCmd = $cmd
                Write-OK "$ver  →  $cmd"
                break
            } else {
                Write-WARN "$ver encontrado pero se necesita 3.12+. Intentando otro..."
            }
        }
    } catch { }
}

if (-not $pythonCmd) {
    Write-ERR "Python 3.12+ no encontrado."
    Write-INFO "Descarga desde: https://www.python.org/downloads/"
    Write-INFO "Asegurate de marcar 'Add Python to PATH' durante la instalacion."
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. uv
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "2/6  uv (gestor de paquetes)"

$uvAvailable = $false
try {
    $uvVer = uv --version 2>&1
    Write-OK "uv $uvVer"
    $uvAvailable = $true
} catch {
    Write-WARN "uv no encontrado. Instalando..."
    try {
        # Instalador oficial de uv
        Invoke-RestMethod "https://astral.sh/uv/install.ps1" | Invoke-Expression
        # Recargar PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" + $env:PATH
        $uvVer = uv --version 2>&1
        Write-OK "uv instalado: $uvVer"
        $uvAvailable = $true
    } catch {
        Write-ERR "No se pudo instalar uv automaticamente."
        Write-INFO "Instala manualmente: https://docs.astral.sh/uv/getting-started/installation/"
    }
}

# Instalar dependencias del proyecto
if ($uvAvailable) {
    Write-INFO "Instalando dependencias del proyecto (uv sync)..."
    try {
        uv sync 2>&1 | ForEach-Object { Write-INFO $_ }
        Write-OK "Dependencias instaladas"
    } catch {
        Write-WARN "uv sync fallo. Intentando pip install..."
        & $pythonCmd -m pip install -r requirements.txt 2>&1 | Out-Null
    }
}

# Determinar python del venv
$venvPython = if (Test-Path ".venv\Scripts\python.exe") {
    ".venv\Scripts\python.exe"
} elseif (Test-Path ".venv\bin\python") {
    ".venv\bin\python"
} else {
    $pythonCmd
}
Write-INFO "Python del entorno: $venvPython"

# ─────────────────────────────────────────────────────────────────────────────
# 3. GPU NVIDIA — drivers
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "3/6  GPU NVIDIA"

$gpuDetected = $false
$vramGB = 0
try {
    $nvOut = nvidia-smi --query-gpu=name,memory.total,driver_version `
                        --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        $parts = $nvOut -split ","
        $gpuName = $parts[0].Trim()
        $vramMB  = [int]($parts[1].Trim() -replace " MiB","")
        $vramGB  = [math]::Round($vramMB / 1024, 1)
        $driver  = $parts[2].Trim()
        Write-OK "GPU: $gpuName"
        Write-OK "VRAM: ${vramGB} GB"
        Write-OK "Driver: $driver"
        $gpuDetected = $true
    }
} catch { }

if (-not $gpuDetected) {
    Write-WARN "nvidia-smi no encontrado o no responde."
    Write-INFO "Asegurate de tener drivers NVIDIA instalados:"
    Write-INFO "  https://www.nvidia.com/Download/index.aspx"
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. torch + CUDA
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "4/6  PyTorch CUDA"

$torchOk = $false
$cudaAvail = $false

try {
    $torchCheck = & $venvPython -c @"
import sys
try:
    import torch
    print('TORCH_VER=' + torch.__version__)
    print('CUDA_AVAIL=' + str(torch.cuda.is_available()))
    print('CUDA_VER=' + str(torch.version.cuda))
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print('GPU_NAME=' + p.name)
        print('VRAM_GB=' + str(round(p.total_memory/1024**3, 1)))
except ImportError:
    print('TORCH_VER=NOT_INSTALLED')
    print('CUDA_AVAIL=False')
"@ 2>&1

    $torchVer   = ($torchCheck | Where-Object { $_ -match "TORCH_VER=" }) -replace "TORCH_VER=",""
    $cudaAvail  = ($torchCheck | Where-Object { $_ -match "CUDA_AVAIL=True" }) -ne $null
    $cudaVer    = ($torchCheck | Where-Object { $_ -match "CUDA_VER=" }) -replace "CUDA_VER=",""
    $torchGpu   = ($torchCheck | Where-Object { $_ -match "GPU_NAME=" }) -replace "GPU_NAME=",""

    if ($torchVer -eq "NOT_INSTALLED") {
        Write-WARN "torch no instalado."
    } elseif ($cudaAvail) {
        Write-OK "torch $torchVer  |  CUDA $cudaVer"
        Write-OK "GPU en uso: $torchGpu"
        $torchOk = $true
    } else {
        Write-WARN "torch $torchVer instalado pero CUDA no disponible."
        Write-INFO "  (torch reporta: cuda_available=False)"
    }
} catch {
    Write-WARN "No se pudo verificar torch: $_"
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Instalacion de torch con CUDA (si es necesario)
# ─────────────────────────────────────────────────────────────────────────────
if (-not $torchOk -and $gpuDetected) {
    Write-Header "5/6  Instalacion de CUDA para PyTorch"
    Write-INFO "Se detecto GPU NVIDIA pero torch no tiene soporte CUDA activo."
    Write-INFO ""
    Write-INFO "Opciones:"
    Write-INFO "  [A] Automatica — instalar torch con CUDA via PyPI (recomendado)"
    Write-INFO "      El wheel de PyPI incluye las librerias CUDA necesarias."
    Write-INFO "      No se requiere instalar CUDA Toolkit por separado."
    Write-INFO ""
    Write-INFO "  [M] Manual — instrucciones para instalar manualmente"
    Write-INFO "      Util si la red bloquea PyPI o necesitas una version especifica."
    Write-INFO ""

    if ($SkipConfirm) {
        $choice = "A"
    } else {
        $choice = Read-Host "  Elige [A/M]"
    }

    if ($choice.ToUpper() -eq "A") {
        Write-INFO ""
        Write-INFO "Instalando torch con CUDA desde PyPI..."
        Write-INFO "(Descarga ~2.5 GB — puede tardar varios minutos)"
        Write-INFO ""

        try {
            if ($uvAvailable) {
                uv pip install "torch>=2.6.0" --index-url "https://pypi.org/simple/"
            } else {
                & $pythonCmd -m pip install "torch>=2.6.0" `
                    --index-url "https://pypi.org/simple/"
            }

            # Re-verificar
            $reCheck = & $venvPython -c @"
import torch
print('CUDA_AVAIL=' + str(torch.cuda.is_available()))
print('CUDA_VER=' + str(torch.version.cuda))
print('TORCH_VER=' + torch.__version__)
"@ 2>&1
            $cudaAfter = ($reCheck | Where-Object { $_ -match "CUDA_AVAIL=True" }) -ne $null
            $torchVerAfter = ($reCheck | Where-Object { $_ -match "TORCH_VER=" }) -replace "TORCH_VER=",""
            $cudaVerAfter  = ($reCheck | Where-Object { $_ -match "CUDA_VER=" }) -replace "CUDA_VER=",""

            if ($cudaAfter) {
                Write-OK "torch $torchVerAfter con CUDA $cudaVerAfter instalado correctamente"
                $torchOk = $true
            } else {
                Write-ERR "torch instalado pero CUDA sigue sin estar disponible."
                Write-INFO "Posibles causas:"
                Write-INFO "  - Drivers NVIDIA desactualizados (actualiza a la ultima version)"
                Write-INFO "  - GPU no soportada por torch (necesita Compute Capability >= 3.7)"
                Write-INFO "  - Ejecuta: nvidia-smi  y verifica que el driver funciona"
            }
        } catch {
            Write-ERR "Error durante la instalacion: $_"
        }

    } else {
        # Manual instructions
        Write-Header "Instrucciones de instalacion manual"
        Write-INFO ""
        Write-INFO "PASO 1 — Verifica que tienes drivers NVIDIA actualizados:"
        Write-INFO "  https://www.nvidia.com/Download/index.aspx"
        Write-INFO "  (Game Ready Driver o Studio Driver — incluyen CUDA runtime)"
        Write-INFO ""
        Write-INFO "PASO 2 — Instala torch con CUDA desde PyPI:"
        Write-INFO "  Si usas uv (recomendado):"
        Write-INFO "    uv pip install `"torch>=2.6.0`""
        Write-INFO ""
        Write-INFO "  Si usas pip:"
        Write-INFO "    pip install torch --index-url https://pypi.org/simple/"
        Write-INFO ""
        Write-INFO "  Si necesitas version especifica de CUDA (e.g. cu121):"
        Write-INFO "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
        Write-INFO "    (requiere acceso a download.pytorch.org)"
        Write-INFO ""
        Write-INFO "PASO 3 — Verifica:"
        Write-INFO "    python -c `"import torch; print(torch.cuda.is_available(), torch.version.cuda)`""
        Write-INFO ""
        Write-INFO "Vuelve a ejecutar este script despues de instalar."
    }
} elseif (-not $gpuDetected) {
    Write-Header "5/6  GPU no detectada"
    Write-WARN "No se encontro GPU NVIDIA. El pipeline correra en CPU."
    Write-WARN "En CPU, PGExplainer tarda ~24 min/nodo y GNNShap ~8 min/nodo."
    Write-INFO "Si tienes GPU, verifica que los drivers NVIDIA esten instalados."
} else {
    Write-Header "5/6  torch CUDA"
    Write-OK "torch ya tiene soporte CUDA. No se requiere accion."
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. WSL2
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "6/6  WSL2 (recomendado para correr los scripts .sh)"

$wslAvail = $false
try {
    $wslOut = wsl --status 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-OK "WSL2 disponible."
        # Check for a Linux distro
        $wslList = wsl --list --verbose 2>&1
        Write-OK $wslList
        $wslAvail = $true
    }
} catch { }

if (-not $wslAvail) {
    Write-WARN "WSL2 no detectado."
    Write-INFO ""
    Write-INFO "WSL2 es ALTAMENTE recomendado para este proyecto porque:"
    Write-INFO "  - Los scripts .sh (watchdog, lock file, restart) corren nativamente"
    Write-INFO "  - Mejor compatibilidad con PyTorch CUDA en Windows 11"
    Write-INFO "  - El dataset Elliptic y los modelos se comportan igual que en Linux"
    Write-INFO ""
    Write-INFO "Para instalar WSL2:"
    Write-INFO "  1. Abre PowerShell como Administrador"
    Write-INFO "  2. Ejecuta: wsl --install"
    Write-INFO "  3. Reinicia el equipo"
    Write-INFO "  4. Abre Ubuntu desde el menu Inicio y configura usuario"
    Write-INFO ""
    Write-INFO "Alternativa sin WSL2: usa los archivos .bat incluidos en scripts\"
    Write-INFO "  (ejecutan el pipeline nativo pero SIN watchdog ni lock file)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Resumen final
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "RESUMEN"

$items = @(
    @{ ok = ($null -ne $pythonCmd); label = "Python 3.12+" },
    @{ ok = $uvAvailable;           label = "uv" },
    @{ ok = $gpuDetected;           label = "GPU NVIDIA detectada" },
    @{ ok = $torchOk;               label = "torch con CUDA" },
    @{ ok = $wslAvail;              label = "WSL2 (recomendado)" }
)

foreach ($item in $items) {
    if ($item.ok) {
        Write-OK $item.label
    } else {
        Write-WARN $item.label
    }
}

Write-Host ""
if ($torchOk) {
    Write-Host "  Setup completo. Proximos pasos:" -ForegroundColor Green
    if ($wslAvail) {
        Write-Host "    WSL2:    bash scripts/run_machineX.sh" -ForegroundColor Green
    }
    Write-Host "    Windows: scripts\run_machineX.bat" -ForegroundColor Green
} else {
    Write-Host "  Revisa los warnings anteriores antes de lanzar el pipeline." -ForegroundColor Yellow
}
Write-Host ""
