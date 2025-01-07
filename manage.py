#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import joblib
import logging
from sklearn import __version__ as sklearn_version

# Configure logging for better debugging information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """Function to load the machine learning model using joblib."""
    model_path = os.path.join(os.path.dirname(__file__), 'path_to_your_model.pkl')
    
    logger.info(f"Attempting to load model from {model_path}")
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully! Using scikit-learn version: {sklearn_version}")
        return model
    
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise Exception(f"Error loading model: {str(e)}")


def main():
    """Main function to run Django and handle model-related functionality."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')

    # Load the model (for example purposes, you can load it wherever required)
    try:
        model = load_model()
        logger.info("Model loaded successfully!")
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()