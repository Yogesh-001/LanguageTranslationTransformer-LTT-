@echo off
echo Starting Transformer Translation Web App in CPU-only mode...

REM Force CPU usage
set FORCE_CPU=True

REM Set Flask environment
set FLASK_ENV=development

echo Starting application on http://localhost:5000 (CPU-only mode)
python app.py
