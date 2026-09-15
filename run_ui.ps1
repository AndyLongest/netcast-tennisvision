$python = "$PSScriptRoot\.venv\Scripts\python.exe"
$cloudProvider = [Environment]::GetEnvironmentVariable("TENNISVISION_CLOUD_PROVIDER", "User")
$cloudUrl = [Environment]::GetEnvironmentVariable("TENNISVISION_CLOUD_URL", "User")
$cloudToken = [Environment]::GetEnvironmentVariable("TENNISVISION_CLOUD_TOKEN", "User")
$ppioApiKey = [Environment]::GetEnvironmentVariable("PPIO_API_KEY", "User")
if ($cloudProvider) {
    $env:TENNISVISION_CLOUD_PROVIDER = $cloudProvider
    $env:TENNISVISION_CLOUD_TOKEN = $cloudToken
    $env:PPIO_API_KEY = $ppioApiKey
}
if ($cloudUrl) {
    $env:TENNISVISION_CLOUD_URL = $cloudUrl
    $env:TENNISVISION_CLOUD_TOKEN = $cloudToken
}

function Test-NetcastTennisVisionServer([int]$CandidatePort) {
    try {
        $candidateBaseUrl = "http://127.0.0.1:$CandidatePort"
        $statusResponse = Invoke-WebRequest -UseBasicParsing "$candidateBaseUrl/api/status" -TimeoutSec 1
        $pageResponse = Invoke-WebRequest -UseBasicParsing "$candidateBaseUrl/web/" -TimeoutSec 1
        $targetHeader = @($statusResponse.Headers["X-Netcast-Execution-Target"]) -join ","
        $expectsCloud = [bool]$cloudUrl -or [bool]$cloudProvider
        $isCloudRelay = $targetHeader -match "cloud"
        return $statusResponse.StatusCode -eq 200 -and $pageResponse.StatusCode -eq 200 `
            -and $isCloudRelay -eq $expectsCloud
    }
    catch {
        return $false
    }
}

$port = $null
foreach ($candidatePort in 4173..4183) {
    if (Test-NetcastTennisVisionServer $candidatePort) {
        $port = $candidatePort
        break
    }
}

if ($null -eq $port) {
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Netcast TennisVision is not installed. Run .\setup.ps1 first."
    }
    foreach ($candidatePort in 4173..4183) {
        $listener = $null
        try {
            $listener = [System.Net.Sockets.TcpListener]::new(
                [System.Net.IPAddress]::Loopback, $candidatePort)
            $listener.Start()
            $port = $candidatePort
            break
        }
        catch {
            continue
        }
        finally {
            if ($null -ne $listener) { $listener.Stop() }
        }
    }
    if ($null -eq $port) {
        throw "Ports 4173-4183 are unavailable. Netcast TennisVision cannot start."
    }
    $server = Start-Process -FilePath $python `
        -ArgumentList @("-m", "netcast_tennisvision", "--port", "$port") `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 200
        if ($server.HasExited) {
            throw "The Netcast TennisVision service exited during startup."
        }
        if (Test-NetcastTennisVisionServer $port) {
            $ready = $true
            break
        }
    }
    if (-not $ready) {
        throw "The Netcast TennisVision service did not become ready in time."
    }
}

$baseUrl = "http://127.0.0.1:$port"
$appUrl = "$baseUrl/web/"
Write-Host "Netcast TennisVision is ready: $appUrl" -ForegroundColor Green
Start-Process $appUrl
