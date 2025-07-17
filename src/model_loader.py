import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path

@st.cache_resource
def load_model_and_data():
    """
    Load the trained model, processed data, and metadata
    Returns: model, dataframe, encoders, content_representations
    """
    try:
        # Define paths
        base_path = Path(__file__).parent.parent
        models_path = base_path / "models"
        data_path = base_path / "data"
        
        # Load model
        model_path = models_path / "netflix_hybrid_recommender.h5"
        model = tf.keras.models.load_model(str(model_path))
        
        # Load processed data
        data_file = data_path / "processed_netflix_hybrid.csv"
        df = pd.read_csv(data_file)
        
        # Load metadata and encoders
        metadata_path = models_path / "hybrid_model_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load precomputed representations
        repr_path = data_path / "content_representations.npy"
        representations = np.load(repr_path)
        
        # Extract encoders
        encoders = {
            'language_encoder': metadata['language_encoder'],
            'content_type_encoder': metadata['content_type_encoder'],
            'scaler': metadata['scaler'],
            'feature_scaler': metadata['feature_scaler']
        }
        
        return model, df, encoders, representations
        
    except Exception as e:
        st.error(f"Error loading model and data: {e}")
        return None, None, None, None

def get_model_info():
    """Get model metadata information"""
    try:
        base_path = Path(__file__).parent.parent
        metadata_path = base_path / "models" / "hybrid_model_metadata.pkl"
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return metadata
    except Exception as e:
        st.error(f"Error loading model info: {e}")
        return None