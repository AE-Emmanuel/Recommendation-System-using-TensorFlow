import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_loader import load_model_and_data, get_model_info
from recommendation_engine import NetflixRecommendationEngine

# Page config
st.set_page_config(
    page_title="Analytics Dashboard - Netflix AI",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def get_system():
    """Load and cache system components"""
    model, df, encoders, representations = load_model_and_data()
    if model is not None:
        engine = NetflixRecommendationEngine(model, df, encoders, representations)
        return engine, df, model
    return None, None, None

def create_content_distribution_chart(df):
    """Create content type distribution chart"""
    content_counts = df['Content Type'].value_counts()
    
    fig = px.pie(
        values=content_counts.values,
        names=content_counts.index,
        title="📺 Content Type Distribution",
        color_discrete_sequence=['#E50914', '#00D4AA']
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def create_language_distribution_chart(df):
    """Create language distribution chart"""
    lang_counts = df['Language Indicator'].value_counts()
    
    fig = px.bar(
        x=lang_counts.index,
        y=lang_counts.values,
        title="🌍 Content by Language",
        color=lang_counts.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        xaxis_title="Language",
        yaxis_title="Number of Titles",
        height=400,
        showlegend=False
    )
    
    return fig

def create_popularity_distribution(df):
    """Create popularity score distribution"""
    fig = px.histogram(
        df,
        x='Popularity Score',
        nbins=30,
        title="⭐ Popularity Score Distribution",
        color_discrete_sequence=['#E50914']
    )
    
    fig.update_layout(
        xaxis_title="Popularity Score",
        yaxis_title="Number of Titles",
        height=400
    )
    
    return fig

def create_content_age_analysis(df):
    """Create content age analysis"""
    age_counts = df['Content Age'].value_counts().sort_index()
    
    fig = px.line(
        x=age_counts.index,
        y=age_counts.values,
        title="📅 Content Age Distribution",
        markers=True
    )
    
    fig.update_traces(line_color='#E50914', marker_color='#E50914')
    fig.update_layout(
        xaxis_title="Content Age (Years)",
        yaxis_title="Number of Titles",
        height=400
    )
    
    return fig

def create_hours_vs_popularity_scatter(df):
    """Create scatter plot of hours viewed vs popularity"""
    # Sample data for performance
    sample_df = df.sample(min(1000, len(df)))
    
    fig = px.scatter(
        sample_df,
        x='Hours Viewed Clean',
        y='Popularity Score',
        color='Content Type',
        title="👀 Hours Viewed vs Popularity Score",
        hover_data=['Title', 'Language Indicator'],
        color_discrete_sequence=['#E50914', '#00D4AA']
    )
    
    fig.update_layout(height=400)
    
    return fig

def display_model_performance():
    """Display model performance metrics"""
    model_info = get_model_info()
    
    if model_info:
        st.markdown("## 🤖 Model Architecture & Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Model Features",
                f"{model_info.get('num_titles', 'N/A'):,}",
                "Content Items"
            )
        
        with col2:
            st.metric(
                "🌍 Languages",
                model_info.get('num_languages', 'N/A'),
                "Supported"
            )
        
        with col3:
            st.metric(
                "📺 Content Types",
                model_info.get('num_types', 'N/A'),
                "Categories"
            )
        
        with col4:
            st.metric(
                "🎭 Seasons",
                model_info.get('num_seasons', 'N/A'),
                "Seasonal Patterns"
            )
        
        # Model details
        with st.expander("🔍 Model Technical Details"):
            st.markdown(f"""
            **Architecture**: Hybrid Neural Network with Embeddings
            
            **Features**:
            - Title Embeddings: 32 dimensions
            - Language Embeddings: 8 dimensions  
            - Content Type Embeddings: 4 dimensions
            - Season Embeddings: 2 dimensions
            - Numerical Features: {len(model_info.get('feature_columns', []))} features
            
            **Training**:
            - Regularization: L2 + Dropout
            - Optimizer: Adam with adaptive learning rate
            - Loss Function: Mean Squared Error
            - Early Stopping: Validation-based
            
            **Recommendation Methods**:
            - Content-Based: Cosine similarity on learned embeddings
            - Collaborative: Popularity and behavioral patterns
            - Hybrid: Weighted combination (60% content + 40% collaborative)
            """)

def main():
    st.title("📊 Netflix Dataset Analytics Dashboard")
    st.markdown("Deep insights into your recommendation system's data and performance")
    
    # Load system
    engine, df, model = get_system()
    
    if engine is None:
        st.error("❌ Failed to load analytics data. Please check your setup.")
        return
    
    # Get analytics data
    analytics_data = engine.get_analytics_data()
    
    # Overview metrics
    st.markdown("## 🎯 Dataset Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📺 Total Content",
            f"{analytics_data['total_content']:,}",
            "Items"
        )
    
    with col2:
        movies_count = analytics_data['content_by_type'].get('Movie', 0)
        st.metric(
            "🎬 Movies",
            f"{movies_count:,}",
            f"{movies_count/analytics_data['total_content']*100:.1f}%"
        )
    
    with col3:
        shows_count = analytics_data['content_by_type'].get('Show', 0)
        st.metric(
            "📺 Shows",
            f"{shows_count:,}",
            f"{shows_count/analytics_data['total_content']*100:.1f}%"
        )
    
    with col4:
        st.metric(
            "🌍 Languages",
            len(analytics_data['content_by_language']),
            "Supported"
        )
    
    with col5:
        avg_popularity = analytics_data['popularity_distribution']['mean']
        st.metric(
            "⭐ Avg Popularity",
            f"{avg_popularity:.3f}",
            "Score"
        )
    
    # Charts in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Content Analysis", "🌍 Global Insights", "⭐ Popularity Metrics", "🎯 Top Performers"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            content_chart = create_content_distribution_chart(df)
            st.plotly_chart(content_chart, use_container_width=True)
        
        with col2:
            age_chart = create_content_age_analysis(df)
            st.plotly_chart(age_chart, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            language_chart = create_language_distribution_chart(df)
            st.plotly_chart(language_chart, use_container_width=True)
        
        with col2:
            # Language-based statistics
            st.markdown("### 🌍 Language Statistics")
            
            for lang, count in analytics_data['content_by_language'].head(6).items():
                percentage = (count / analytics_data['total_content']) * 100
                st.progress(percentage / 100, text=f"{lang}: {count:,} titles ({percentage:.1f}%)")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            popularity_chart = create_popularity_distribution(df)
            st.plotly_chart(popularity_chart, use_container_width=True)
        
        with col2:
            scatter_chart = create_hours_vs_popularity_scatter(df)
            st.plotly_chart(scatter_chart, use_container_width=True)
        
        # Popularity statistics
        st.markdown("### ⭐ Popularity Statistics")
        pop_stats = analytics_data['popularity_distribution']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Maximum", f"{pop_stats['max']:.3f}")
        with col2:
            st.metric("Average", f"{pop_stats['mean']:.3f}")
        with col3:
            st.metric("Median", f"{pop_stats['50%']:.3f}")
        with col4:
            st.metric("Std Dev", f"{pop_stats['std']:.3f}")
    
    with tab4:
        st.markdown("### 🏆 Top Performing Content")
        
        # Display top content table
        top_content = analytics_data['top_content']
        
        # Create enhanced table with formatting
        st.dataframe(
            top_content.style.format({
                'Popularity Score': '{:.3f}'
            }).background_gradient(subset=['Popularity Score'], cmap='Reds'),
            use_container_width=True
        )
        
        # Top performers by category
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎬 Top Movies")
            top_movies = df[df['Content Type'] == 'Movie'].nlargest(5, 'Popularity Score')
            for idx, (_, movie) in enumerate(top_movies.iterrows(), 1):
                st.write(f"{idx}. **{movie['Title']}** - {movie['Popularity Score']:.3f}")
        
        with col2:
            st.markdown("#### 📺 Top Shows")
            top_shows = df[df['Content Type'] == 'Show'].nlargest(5, 'Popularity Score')
            for idx, (_, show) in enumerate(top_shows.iterrows(), 1):
                st.write(f"{idx}. **{show['Title']}** - {show['Popularity Score']:.3f}")
    
    # Model performance section
    st.markdown("---")
    display_model_performance()
    
    # Additional insights
    st.markdown("## 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📈 Content Trends
        - **English content dominates** the platform
        - **Shows vs Movies** distribution shows content strategy
        - **Recent content** has higher popularity scores
        - **Global availability** correlates with higher viewership
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI Model Insights
        - **Hybrid approach** provides balanced recommendations
        - **Cross-language recommendations** enable global discovery
        - **Embedding similarity** captures content relationships
        - **Regularization** prevents overfitting for better generalization
        """)

if __name__ == "__main__":
    main()