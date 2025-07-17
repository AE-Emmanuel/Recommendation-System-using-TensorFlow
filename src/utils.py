import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

def format_large_number(num):
    """Format large numbers for display"""
    if num >= 1000000000:
        return f"{num/1000000000:.1f}B"
    elif num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return str(int(num))

def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Create a styled metric card"""
    delta_html = ""
    if delta:
        color = {"normal": "#666", "inverse": "#E50914", "off": "#666"}[delta_color]
        delta_html = f'<div style="color: {color}; font-size: 0.8rem;">{delta}</div>'
    
    return f"""
    <div style="
        background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
        border: 1px solid #E50914;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    ">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #E50914;">{value}</div>
        {delta_html}
    </div
    """

def get_color_scale(method: str) -> List[str]:
    """Get color scale based on recommendation method"""
    scales = {
        'content': ['#E50914', '#F40612', '#FF4136'],
        'collaborative': ['#00D4AA', '#00B89C', '#009B8E'],
        'hybrid': ['#FFB000', '#FF9500', '#FF7A00']
    }
    return scales.get(method, ['#E50914', '#F40612'])

def calculate_recommendation_diversity(recommendations: pd.DataFrame) -> Dict[str, float]:
    """Calculate diversity metrics for recommendations"""
    if recommendations.empty:
        return {}
    
    metrics = {}
    
    # Content type diversity
    unique_types = len(recommendations['Content Type'].unique())
    metrics['content_diversity'] = unique_types / len(recommendations)
    
    # Language diversity
    unique_languages = len(recommendations['Language Indicator'].unique())
    metrics['language_diversity'] = unique_languages / len(recommendations)
    
    # Popularity distribution
    metrics['avg_popularity'] = recommendations['Popularity Score'].mean()
    metrics['popularity_std'] = recommendations['Popularity Score'].std()
    
    return metrics

def create_comparison_chart(data: Dict[str, List], title: str) -> go.Figure:
    """Create a comparison chart for different recommendation methods"""
    fig = go.Figure()
    
    methods = list(data.keys())
    colors = ['#E50914', '#00D4AA', '#FFB000']
    
    for i, (method, values) in enumerate(data.items()):
        fig.add_trace(go.Bar(
            name=method,
            x=list(range(len(values))),
            y=values,
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title=title,
        barmode='group',
        height=400
    )
    
    return fig

def validate_user_input(input_text: str, min_length: int = 2) -> Tuple[bool, str]:
    """Validate user input for search and recommendations"""
    if not input_text:
        return False, "Please enter a search term"
    
    if len(input_text.strip()) < min_length:
        return False, f"Search term must be at least {min_length} characters"
    
    # Check for potentially harmful input
    harmful_patterns = ['<script', 'javascript:', 'onload=', 'onerror=']
    if any(pattern in input_text.lower() for pattern in harmful_patterns):
        return False, "Invalid characters in search term"
    
    return True, ""

def export_recommendations_to_csv(recommendations: pd.DataFrame, filename: str = None):
    """Export recommendations to CSV"""
    if filename is None:
        filename = f"netflix_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    csv = recommendations.to_csv(index=False)
    return csv, filename

def get_recommendation_summary(recommendations: pd.DataFrame) -> str:
    """Generate a text summary of recommendations"""
    if recommendations.empty:
        return "No recommendations available."
    
    total = len(recommendations)
    content_types = recommendations['Content Type'].value_counts()
    languages = recommendations['Language Indicator'].value_counts()
    avg_score = recommendations.get('Hybrid Score', recommendations.get('Similarity Score', pd.Series([0]))).mean()
    
    summary = f"""
    📊 **Recommendation Summary:**
    - Total recommendations: {total}
    - Content types: {', '.join([f"{k} ({v})" for k, v in content_types.items()])}
    - Languages: {', '.join([f"{k} ({v})" for k, v in languages.items()])}
    - Average score: {avg_score:.3f}
    """
    
    return summary

def cache_key_generator(*args, **kwargs) -> str:
    """Generate cache key for Streamlit caching"""
    import hashlib
    key_string = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_string.encode()).hexdigest()

class SessionState:
    """Manage session state for Streamlit app"""
    
    @staticmethod
    def init_state():
        """Initialize session state variables"""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        if 'recommendation_history' not in st.session_state:
            st.session_state.recommendation_history = []
        
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'preferred_method': 'hybrid',
                'preferred_language': 'All',
                'preferred_type': 'All'
            }
    
    @staticmethod
    def add_search(query: str):
        """Add search query to history"""
        if query and query not in st.session_state.search_history:
            st.session_state.search_history.insert(0, query)
            # Keep only last 10 searches
            st.session_state.search_history = st.session_state.search_history[:10]
    
    @staticmethod
    def add_recommendation(title: str, method: str):
        """Add recommendation request to history"""
        entry = {'title': title, 'method': method, 'timestamp': pd.Timestamp.now()}
        st.session_state.recommendation_history.insert(0, entry)
        # Keep only last 20 recommendations
        st.session_state.recommendation_history = st.session_state.recommendation_history[:20]

def create_download_button(data: str, filename: str, label: str):
    """Create a download button for data"""
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime='text/csv'
    )