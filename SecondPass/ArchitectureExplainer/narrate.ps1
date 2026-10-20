$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$movieRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$movieScript = Get-Content -LiteralPath (Join-Path $movieRoot 'script.json') -Raw | ConvertFrom-Json
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.SelectVoice('Microsoft Zira Desktop')
$voice.Rate = 1
New-Item -ItemType Directory -Path (Join-Path $movieRoot 'voice') -Force | Out-Null
for ($scene=0; $scene -lt $movieScript.Count; $scene++) {
    for ($sentence=0; $sentence -lt $movieScript[$scene].sentences.Count; $sentence++) {
        $voice.SetOutputToWaveFile((Join-Path $movieRoot ('voice/{0:D2}-{1:D2}.wav' -f $scene,$sentence)))
        $voice.Speak([string]$movieScript[$scene].sentences[$sentence])
        $voice.SetOutputToNull()
    }
}
$voice.Dispose()
Write-Output 'Narration generated.'
