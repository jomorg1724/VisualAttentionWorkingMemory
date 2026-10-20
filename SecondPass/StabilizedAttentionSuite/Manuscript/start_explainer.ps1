$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$listener = Get-NetTCPConnection -State Listen -LocalPort 8877 -ErrorAction SilentlyContinue
if (-not $listener) {
    $pythonCommand = (Get-Command python.exe -ErrorAction Stop).Source
    $serverArgs = @('-m', 'http.server', '8877', '--bind', '127.0.0.1', '--directory', "`"$projectRoot`"")
    $server = Start-Process -FilePath $pythonCommand -ArgumentList $serverArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
    Write-Output "Started local explainer server (PID $($server.Id))."
}
Write-Output 'http://127.0.0.1:8877/SecondPass/StabilizedAttentionSuite/Manuscript/index.html'
