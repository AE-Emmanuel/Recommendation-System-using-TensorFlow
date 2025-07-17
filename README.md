# 🎬 Netflix AI Recommendation System

An intelligent recommendation system built with Machine Learning and deployed using Streamlit and Docker.

## 🚀 Features

- **Hybrid Recommendation Engine**: Combines content-based and collaborative filtering
- **Interactive Web Interface**: Built with Streamlit
- **Real-time Analytics**: Visualization of recommendation scores and patterns
- **Multiple Recommendation Methods**: Hybrid, Content-based, and Collaborative filtering
- **Dockerized**: Easy deployment and scaling

## 🛠️ Technology Stack

- **Backend**: Python, Scikit-learn, Pandas, NumPy
- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Containerization**: Docker
- **ML Models**: Content-based filtering, Collaborative filtering

## 📋 Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- 4GB+ RAM (for model loading)

## 🔧 Installation & Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd netflix-recommendation-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download data and models**
   ```bash
   # Add instructions for downloading your data files
   # Example:
   # wget <data-url> -O data/netflix_content.csv
   # wget <model-url> -O models/hybrid_model_metadata.pkl
   ```

5. **Run the application**
   ```bash
   streamlit run src/Home.py
   ```

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t netflix-ai-app .
   ```

2. **Run the container**
   ```bash
   docker run -d \
     --name netflix-app \
     -p 8501:8501 \
     -v "$(pwd)/data:/app/data" \
     -v "$(pwd)/models:/app/models" \
     netflix-ai-app
   ```

3. **Access the application**
   ```
   http://localhost:8501
   ```

## 📁 Project Structure

```
├── src/
│   ├── model_loader.py          # Model loading utilities
│   ├── recommendation_engine.py  # Core recommendation logic
│   └── utils.py                 # Helper functions
├── pages/
│   ├── Home.py                  # Main application page
│   ├── 1_Recommendations.py     # Recommendation interface
│   └── 2_Analytics.py           # Analytics dashboard
├── data/                        # Data files (not in git)
├── models/                      # Trained models (not in git)
├── notebooks/                   # Jupyter notebooks
├── Dockerfile                   # Docker configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎯 How to Use

1. **Search Content**: Use the search dropdown to find Netflix titles
2. **Get Recommendations**: Select a title to get AI-powered recommendations
3. **Customize Settings**: Adjust recommendation method, filters, and number of results
4. **View Analytics**: Explore recommendation scores and patterns
5. **Compare Methods**: See how different algorithms perform

## 🔄 Recommendation Methods

- **🔥 Hybrid**: Combines content-based and collaborative filtering for best results
- **📚 Content-Based**: Recommends based on content similarity (genres, cast, etc.)
- **🤝 Collaborative**: Recommends based on user behavior patterns

## 🚀 Cloud Deployment

### AWS ECS Deployment

1. **Push to ECR**
   ```bash
   # Build and tag for ECR
   docker build -t netflix-ai-app .
   docker tag netflix-ai-app:latest <aws-account-id>.dkr.ecr.<region>.amazonaws.com/netflix-ai-app:latest
   
   # Push to ECR
   docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/netflix-ai-app:latest
   ```

2. **Deploy to ECS**
   - Create ECS cluster
   - Define task definition
   - Create service with load balancer

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Configure environment variables
4. Deploy automatically

## 📊 Performance

- **Model Loading**: ~30 seconds (cached after first load)
- **Recommendation Generation**: <2 seconds
- **Memory Usage**: ~2GB RAM
- **Concurrent Users**: 50+ (depending on server specs)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For support and questions:
- Create an issue in this repository
- Contact: [your-email@example.com]

## 🔮 Future Enhancements

- [ ] Real-time model updates
- [ ] User authentication and personalization
- [ ] A/B testing framework
- [ ] Mobile-responsive design
- [ ] API endpoints for external integrations