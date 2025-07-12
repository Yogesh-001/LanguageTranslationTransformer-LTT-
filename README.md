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

### Cloud Deployment

#### Deploying to Heroku

1. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)

2. Login to Heroku:
   ```
   heroku login
   ```

3. Create a new Heroku app:
   ```
   heroku create your-app-name
   ```

4. Push to Heroku:
   ```
   git push heroku main
   ```

#### Deploying to Azure Web App

1. Install the [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)

2. Login to Azure:
   ```
   az login
   ```

3. Create a resource group:
   ```
   az group create --name TranslatorResourceGroup --location eastus
   ```

4. Create an App Service plan:
   ```
   az appservice plan create --name TranslatorAppServicePlan --resource-group TranslatorResourceGroup --sku B1 --is-linux
   ```

5. Create a Web App:
   ```
   az webapp create --resource-group TranslatorResourceGroup --plan TranslatorAppServicePlan --name your-app-name --runtime "PYTHON|3.9"
   ```

6. Deploy using local Git:
   ```
   az webapp deployment source config-local-git --name your-app-name --resource-group TranslatorResourceGroup
   ```

7. Push your code to the Azure Git repository:
   ```
   git remote add azure <url-from-previous-command>
   git push azure main
   ```

## Usage

The web interface allows users to input text in the source language and get translations in the target language. The application can also be used programmatically via the API endpoint:

```
POST /translate
Content-Type: application/json

{
    "source_text": "Your text to translate"
}
```

Response:
```json
{
    "translated_text": "Translated text"
}
```

### Command Line Usage

You can also use the model directly from the command line:

```
python translate.py "Your text to translate"
```

## Deployment Options

### Docker Deployment

1. Build the Docker image:
   ```
   docker build -t transformer-translation .
   ```

2. Run the Docker container:
   ```
   docker run -p 5000:5000 transformer-translation
   ```

3. Access the application at `http://localhost:5000`

### Cloud Deployment

#### Deploying to Azure App Service

1. Create an Azure App Service:
   ```
   az group create --name TranslationApp --location eastus
   az appservice plan create --name TranslationAppPlan --resource-group TranslationApp --sku B1 --is-linux
   az webapp create --resource-group TranslationApp --plan TranslationAppPlan --name your-app-name --runtime "PYTHON|3.9"
   ```

2. Deploy your code:
   ```
   az webapp deploy --resource-group TranslationApp --name your-app-name --src-path .
   ```

#### Deploying to Heroku

1. Create a `Procfile`:
   ```
   echo "web: gunicorn app:app" > Procfile
   ```

2. Deploy to Heroku:
   ```
   heroku create your-app-name
   git add .
   git commit -m "Prepare for Heroku deployment"
   git push heroku master
   ```

## API Usage

You can use the translation API with a POST request:

```
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Your text to translate"}'
```

The API will return a JSON response with the translated text:

```json
{
  "translation": "Translated text"
}
```
