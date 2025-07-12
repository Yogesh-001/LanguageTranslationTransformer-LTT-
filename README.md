# Transformer Translation Model

This is a transformer-based neural machine translation model built with PyTorch.

## Local Development

### Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask application:
   ```
   python app.py
   ```
   Or on Windows, use the provided batch files:
   
   For GPU acceleration (with fallback to CPU):
   ```
   run_app.bat
   ```
   
   For CPU-only mode (to avoid CUDA memory issues):
   ```
   run_app_cpu.bat
   ```

3. Open your browser and navigate to `http://localhost:5000` to use the web interface.

## Memory Management

If you encounter CUDA memory errors like:

```
Error: CUDA out of memory. Tried to allocate X MiB...
```

You have several options:

1. **Use the CPU-only mode**: 
   - Run with `run_app_cpu.bat` instead of `run_app.bat`
   - Check the "Force CPU usage" option in the web interface

2. **Optimize GPU memory**:
   - The application already includes settings to optimize CUDA memory allocation
   - Try to clear your GPU memory before running (other applications might be using it)
   - Reduce batch size if applicable

3. **For deployment**:
   - Use the CPU-only Docker image when GPU memory is limited
   - Consider using a cloud provider with more GPU memory

## Deployment Options

### Docker Deployment

1. Build the Docker image:

   With GPU support (if available):
   ```
   docker build -t transformer-translation-app .
   ```

   CPU-only version (more stable but slower):
   ```
   docker build -f Dockerfile.cpu -t transformer-translation-app-cpu .
   ```

2. Run the Docker container:

   With GPU support (requires nvidia-docker):
   ```
   docker run --gpus all -p 5000:5000 transformer-translation-app
   ```

   CPU-only version:
   ```
   docker run -p 5000:5000 transformer-translation-app-cpu
   ```

3. Access the application at `http://localhost:5000`