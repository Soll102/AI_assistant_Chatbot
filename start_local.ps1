# Start the RAG chatbot locally: backend (FastAPI) + frontend (Vite) + browser.
#
# Usage (double-click also works - right-click > Run with PowerShell - or):
#   powershell -ExecutionPolicy Bypass -File .\start_local.ps1
#
#   Backend:  http://127.0.0.1:8000  (docs: /docs)
#   Frontend: http://127.0.0.1:3000

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$BackendDir = Join-Path $Root "backend"
$FrontendDir = Join-Path $Root "frontend"
$VenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Backend venv not found: $VenvPython" -ForegroundColor Red
    Write-Host "Create it first:"
    Write-Host "  cd backend; python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path -LiteralPath (Join-Path $FrontendDir "node_modules"))) {
    Write-Host "Frontend dependencies missing - installing (npm install)..." -ForegroundColor Yellow
    Push-Location $FrontendDir
    try { npm install } finally { Pop-Location }
}
if (-not (Test-Path -LiteralPath (Join-Path $BackendDir ".env"))) {
    Write-Host "WARNING: backend/.env not found - the backend will run but LLM answers need OPENROUTER_API_KEY." -ForegroundColor Yellow
}

$env:PYTHONIOENCODING = "utf-8"

Write-Host "Starting backend (port 8000)..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$BackendDir'; `$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
) -WorkingDirectory $BackendDir

Write-Host "Starting frontend (port 3000)..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$FrontendDir'; npm run dev"
) -WorkingDirectory $FrontendDir

# Wait for the backend health check, then open the UI.
$BackendUrl = "http://127.0.0.1:8000/health"
$FrontendUrl = "http://127.0.0.1:3000"
for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri $BackendUrl -TimeoutSec 2 -UseBasicParsing
        if ($response.StatusCode -eq 200) { break }
    } catch { Start-Sleep -Seconds 1 }
}

Write-Host "Opening $FrontendUrl ..." -ForegroundColor Green
Start-Process $FrontendUrl
Write-Host "Done. Close the two new terminal windows to stop the app." -ForegroundColor Green
