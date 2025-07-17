# Recommendation System Using TensorFlow

An intelligent recommendation system built with Machine Learning and deployed using Streamlit and Docker.

## Features

- **Hybrid Recommendation Engine**: Combines content-based and collaborative filtering
- **Interactive Web Interface**: Built with Streamlit (For Demo Purposes)
- **Real-time Analytics**: Visualization of recommendation scores and patterns
- **Multiple Recommendation Methods**: Hybrid, Content-based, and Collaborative filtering
- **Dockerized**: Easy deployment and scaling

## Technology Stack

- **Backend**: Python, TensorFlow , Scikit-learn, Pandas, NumPy
- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Containerization**: Docker
- **ML Models**: Content-based filtering, Collaborative filtering , Hybrid(Both Combined)

## Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Used Google Colab T4 Gpu For Model Training


## Project Structure

```
├── src/
│   ├── model_loader.py          # Model loading utilities
│   ├── recommendation_engine.py  # Core recommendation logic
│   └── utils.py                 # Helper functions
├── pages/
│   ├── Home.py                  # Main application page
│   ├── 1_Recommendations.py     # Recommendation interface
│   └── 2_Analytics.py           # Analytics dashboard
├── notebooks/                   # Jupyter notebooks
├── Dockerfile                   # Docker configuration
├── requirements.txt             # Python dependencies
└── README.md                    
```

## Recommendation Models Used : 

- **Hybrid**: Combines content-based and collaborative filtering for best results
- **Content-Based**: Recommends based on content similarity 
- **Collaborative**: Recommends based on user behavior patterns

