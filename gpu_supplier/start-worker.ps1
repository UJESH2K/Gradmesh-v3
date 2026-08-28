param(
    [Parameter(Mandatory = $true)]
    [string]$HostIp,
    [string]$Name = "gpu-supplier",
    [int]$MaxBatchSize = 2
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
python (Join-Path $PSScriptRoot "worker.py") --server-url "http://$HostIp`:8000" --name $Name --max-batch-size $MaxBatchSize
