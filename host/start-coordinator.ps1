$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "Starting GradMesh coordinator on all network interfaces at port 8000..."
Write-Host "Other laptops should use: http://$env:COMPUTERNAME:8000"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
