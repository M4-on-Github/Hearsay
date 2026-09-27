.\.venv\Scripts\python.exe main.py
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe predict.py
} else {
    Write-Host "Training failed, skipping prediction."
}
