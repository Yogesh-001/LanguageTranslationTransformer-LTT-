@echo off
echo Starting Transformer Translation Web App...

REM Set PyTorch memory optimization flags
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Set Flask environment
set FLASK_ENV=development

REM Free up memory before starting
echo Clearing GPU memory if available...
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

echo Starting application on http://localhost:5000
python app.py
