from flask import Flask, request, render_template, jsonify
import torch
from translate import translate
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Check available memory and device at startup
device_info = {}
if torch.cuda.is_available():
    device_info['device'] = 'CUDA'
    device_info['device_name'] = torch.cuda.get_device_name(0)
    device_info['memory_allocated'] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
    device_info['memory_reserved'] = f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
    device_info['max_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    logger.info(f"CUDA device: {device_info['device_name']}")
    logger.info(f"Memory: {device_info['memory_allocated']} allocated, {device_info['memory_reserved']} reserved, {device_info['max_memory']} total")
    
    # Set memory optimization environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
else:
    device_info['device'] = 'CPU'
    logger.info("CUDA not available, using CPU")

@app.route('/')
def home():
    return render_template('index.html', device_info=device_info)

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    source_text = data.get('source_text', '')
    force_cpu = data.get('force_cpu', False)
    max_length = data.get('max_length', 50)  # Default max length
    
    if not source_text:
        return jsonify({'error': 'No text provided for translation'}), 400
    
    try:
        # Check if we need to clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Memory before translation: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")
        
        # Pass all parameters to the translate function
        translated_text = translate(source_text, force_cpu=force_cpu, max_length=max_length)
        
        # Log memory usage after translation
        if torch.cuda.is_available():
            logger.info(f"Memory after translation: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")
        
        return jsonify({
            'translated_text': translated_text,
            'device_used': 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
        })
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            logger.warning(f"CUDA out of memory error: {str(e)}")
            # Try again with CPU
            try:
                logger.info("Retrying with CPU")
                translated_text = translate(source_text, force_cpu=True)
                return jsonify({
                    'translated_text': translated_text,
                    'device_used': 'cpu',
                    'warning': 'GPU memory was exhausted, fell back to CPU'
                })
            except Exception as cpu_error:
                logger.error(f"CPU fallback also failed: {str(cpu_error)}")
                return jsonify({'error': f"Translation failed on both GPU and CPU: {str(cpu_error)}"}), 500
        else:
            logger.error(f"Runtime error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check if running in production or development
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
