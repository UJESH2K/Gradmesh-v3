[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Action = "help",
    [Parameter(Position = 1)]
    [string]$Target = "",
    [Parameter(Position = 2)]
    [string]$Value = ""
)

$ErrorActionPreference = "Stop"
$RootDir = $PSScriptRoot
$RuntimeDir = if ($env:GRADMESH_RUNTIME_DIR) { $env:GRADMESH_RUNTIME_DIR } else { Join-Path $RootDir ".gradmesh" }
$PidDir = Join-Path $RuntimeDir "pids"
$LogDir = Join-Path $RuntimeDir "logs"
$CoordinatorHost = if ($env:GRADMESH_HOST) { $env:GRADMESH_HOST } else { "0.0.0.0" }
$CoordinatorPort = if ($env:GRADMESH_PORT) { [int]$env:GRADMESH_PORT } else { 8000 }
$CoordinatorUrl = if ($env:GRADMESH_SERVER_URL) { $env:GRADMESH_SERVER_URL.TrimEnd("/") } else { "http://127.0.0.1:$CoordinatorPort" }
$LocalWorkerCount = if ($env:GRADMESH_WORKER_COUNT) { [int]$env:GRADMESH_WORKER_COUNT } else { 1 }
$WorkerBackend = if ($env:GRADMESH_WORKER_BACKEND) { $env:GRADMESH_WORKER_BACKEND } else { "auto" }
$MaxBatchSize = if ($env:GRADMESH_MAX_BATCH_SIZE) { [int]$env:GRADMESH_MAX_BATCH_SIZE } else { 2 }
$HeartbeatSeconds = if ($env:GRADMESH_HEARTBEAT_SECONDS) { $env:GRADMESH_HEARTBEAT_SECONDS } else { "5" }
$PollSeconds = if ($env:GRADMESH_POLL_SECONDS) { $env:GRADMESH_POLL_SECONDS } else { "1.5" }

New-Item -ItemType Directory -Force -Path $PidDir, $LogDir | Out-Null

function Find-Python {
    $candidates = [System.Collections.Generic.List[string]]::new()
    if ($env:GRADMESH_PYTHON) { $candidates.Add($env:GRADMESH_PYTHON) }
    if ($env:VIRTUAL_ENV) { $candidates.Add((Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")) }
    $candidates.Add((Join-Path $RootDir ".venv\Scripts\python.exe"))
    $candidates.Add((Join-Path (Split-Path $RootDir -Parent) ".venv\Scripts\python.exe"))
    $pathPython = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pathPython -and $pathPython.Source -notlike "*WindowsApps*") { $candidates.Add($pathPython.Source) }

    foreach ($candidate in $candidates) {
        if (-not $candidate -or -not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        try {
            & $candidate -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { return (Resolve-Path -LiteralPath $candidate).Path }
        } catch {}
    }
    throw "A working Python was not found. Activate .venv or set GRADMESH_PYTHON to python.exe."
}

function ConvertTo-SafeName([string]$Name) {
    return ($Name -replace '[^A-Za-z0-9_.-]', '-')
}

function Get-ManagedProcess([string]$PidFile) {
    if (-not (Test-Path -LiteralPath $PidFile -PathType Leaf)) { return $null }
    $processId = Get-Content -LiteralPath $PidFile -TotalCount 1
    if ($processId -notmatch '^\d+$') {
        Remove-Item -LiteralPath $PidFile -Force
        return $null
    }
    $process = Get-Process -Id ([int]$processId) -ErrorAction SilentlyContinue
    if (-not $process) { Remove-Item -LiteralPath $PidFile -Force }
    return $process
}

function Start-Coordinator {
    $pythonExe = Find-Python
    $pidFile = Join-Path $PidDir "server.pid"
    $existing = Get-ManagedProcess $pidFile
    if ($existing) { Write-Host "coordinator already running (pid $($existing.Id))"; return }
    $stdout = Join-Path $LogDir "server.out.log"
    $stderr = Join-Path $LogDir "server.err.log"
    $process = Start-Process -FilePath $pythonExe -ArgumentList @(
        "-m", "uvicorn", "server:app", "--host", $CoordinatorHost, "--port", "$CoordinatorPort"
    ) -WorkingDirectory $RootDir -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii
    Start-Sleep -Seconds 1
    if (-not (Get-ManagedProcess $pidFile)) { throw "Coordinator failed to start. Inspect $stderr" }
    Write-Host "coordinator started (pid $($process.Id), $CoordinatorUrl/dashboard)"
}

function Wait-Coordinator {
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "$CoordinatorUrl/" -TimeoutSec 1 | Out-Null
            return
        } catch { Start-Sleep -Seconds 1 }
    }
    throw "Coordinator did not become reachable at $CoordinatorUrl within 30 seconds."
}

function Start-Worker([string]$Name) {
    $pythonExe = Find-Python
    if (-not $Name) { throw "Worker name is required." }
    $key = ConvertTo-SafeName $Name
    $pidFile = Join-Path $PidDir "worker-$key.pid"
    $existing = Get-ManagedProcess $pidFile
    if ($existing) { Write-Host "worker '$Name' already running (pid $($existing.Id))"; return }
    Wait-Coordinator
    $stdout = Join-Path $LogDir "worker-$key.out.log"
    $stderr = Join-Path $LogDir "worker-$key.err.log"
    $process = Start-Process -FilePath $pythonExe -ArgumentList @(
        "worker.py", "--server-url", $CoordinatorUrl, "--name", $Name,
        "--node-id", "local-$key", "--backend", $WorkerBackend,
        "--max-batch-size", "$MaxBatchSize", "--heartbeat-seconds", "$HeartbeatSeconds",
        "--poll-seconds", "$PollSeconds"
    ) -WorkingDirectory $RootDir -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii
    Start-Sleep -Seconds 1
    if (-not (Get-ManagedProcess $pidFile)) { throw "Worker '$Name' failed to start. Inspect $stderr" }
    Write-Host "worker '$Name' started (pid $($process.Id), backend=$WorkerBackend)"
}

function Start-Workers([int]$Count) {
    if ($Count -lt 1) { throw "Worker count must be a positive integer." }
    for ($index = 1; $index -le $Count; $index++) { Start-Worker "local-worker-$index" }
}

function Stop-ManagedProcess([string]$Label, [string]$PidFile) {
    $process = Get-ManagedProcess $PidFile
    if (-not $process) { Write-Host "$Label is not running"; return }
    Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 500
    }
    if (Get-Process -Id $process.Id -ErrorAction SilentlyContinue) {
        Write-Warning "$Label did not stop after 10 seconds; forcing pid $($process.Id)"
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host "$Label stopped"
}

function Stop-Worker([string]$Name) {
    $key = ConvertTo-SafeName $Name
    Stop-ManagedProcess "worker '$Name'" (Join-Path $PidDir "worker-$key.pid")
}

function Stop-Workers {
    $pidFiles = @(Get-ChildItem -LiteralPath $PidDir -Filter "worker-*.pid" -File -ErrorAction SilentlyContinue)
    if ($pidFiles.Count -eq 0) { Write-Host "no managed workers are running"; return }
    foreach ($pidFile in $pidFiles) { Stop-ManagedProcess $pidFile.BaseName $pidFile.FullName }
}

function Show-StatusLine([string]$Label, [string]$PidFile) {
    $process = Get-ManagedProcess $PidFile
    if ($process) { Write-Host "RUNNING  $Label  pid=$($process.Id)" } else { Write-Host "STOPPED  $Label" }
}

function Show-Status {
    Show-StatusLine "coordinator" (Join-Path $PidDir "server.pid")
    $pidFiles = @(Get-ChildItem -LiteralPath $PidDir -Filter "worker-*.pid" -File -ErrorAction SilentlyContinue)
    if ($pidFiles.Count -eq 0) { Write-Host "STOPPED  workers (none managed)"; return }
    foreach ($pidFile in $pidFiles) { Show-StatusLine $pidFile.BaseName $pidFile.FullName }
}

function Show-Logs([string]$Name) {
    $prefix = if (-not $Name -or $Name -eq "server") { "server" } else { "worker-$(ConvertTo-SafeName $Name)" }
    $files = @("$prefix.out.log", "$prefix.err.log")
    foreach ($file in $files) {
        $path = Join-Path $LogDir $file
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            Write-Host "==> $path <=="
            Get-Content -LiteralPath $path -Tail 100
        }
    }
}

function Show-Usage {
    @"
Usage:
  .\run.ps1 start
  .\run.ps1 start server
  .\run.ps1 start workers [COUNT]
  .\run.ps1 start worker NAME
  .\run.ps1 stop
  .\run.ps1 stop server|workers
  .\run.ps1 stop worker NAME
  .\run.ps1 restart
  .\run.ps1 status
  .\run.ps1 logs [server|WORKER_NAME]
"@ | Write-Host
}

switch ($Action.ToLowerInvariant()) {
    "start" {
        switch ($Target.ToLowerInvariant()) {
            "" { Start-Coordinator; Start-Workers $LocalWorkerCount }
            "server" { Start-Coordinator }
            "workers" { Start-Workers $(if ($Value) { [int]$Value } else { $LocalWorkerCount }) }
            "worker" { Start-Worker $Value }
            default { Show-Usage; exit 2 }
        }
    }
    "stop" {
        switch ($Target.ToLowerInvariant()) {
            "" { Stop-Workers; Stop-ManagedProcess "coordinator" (Join-Path $PidDir "server.pid") }
            "server" { Stop-ManagedProcess "coordinator" (Join-Path $PidDir "server.pid") }
            "workers" { Stop-Workers }
            "worker" { Stop-Worker $Value }
            default { Show-Usage; exit 2 }
        }
    }
    "restart" { Stop-Workers; Stop-ManagedProcess "coordinator" (Join-Path $PidDir "server.pid"); Start-Coordinator; Start-Workers $LocalWorkerCount }
    "status" { Show-Status }
    "logs" { Show-Logs $Target }
    { $_ -in @("help", "-h", "--help") } { Show-Usage }
    default { Show-Usage; exit 2 }
}
