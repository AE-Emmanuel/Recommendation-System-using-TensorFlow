import streamlit as st
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_loader import load_model_and_data
from recommendation_engine import NetflixRecommendationEngine

# Page config
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #E50914, #F40612);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #001a00;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Load model and data - cached for performance"""
    with st.spinner("🔄 Loading Netflix AI Recommendation System..."):
        try:
            model, df, encoders, representations = load_model_and_data()
            engine = NetflixRecommendationEngine(model, df, encoders, representations)
            return engine, df
        except Exception as e:
            st.error(f"Error loading system: {e}")
            return None, None

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Netflix AI Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    # Initialize system
    engine, df = initialize_system()
    
    if engine is None:
        st.error("❌ Failed to load recommendation system. Please check your model files.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Navigation")
    st.sidebar.markdown("Use the pages in the sidebar to explore different features!")
    
    # Main page content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📺 Total Content", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🌍 Languages", df['Language Indicator'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎭 Content Types", df['Content Type'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    

    st.markdown("## 🚀 Welcome to a New Way of Content Recommendation!")
    
    st.markdown("""
    This **AI-powered recommendation system** uses **hybrid machine learning** approach to suggest Netflix content 
    tailored to your preferences. Built with **TensorFlow** and **deep learning embeddings**.
    
    ### 🎯 Features:
    - **🤖 Hybrid AI Recommendations**: Combines content-based and collaborative filtering
    - **🌐 Cross-Language Discovery**: Find amazing content across different languages  
    - **🔍 Intelligent Search**: Find exactly what you're looking for
    
    ### 🛠️ Technology Stack:
    - **Machine Learning**: TensorFlow 2.15, Scikit-learn
    - **Data Processing**: Pandas, NumPy
    - **Web Interface**: Streamlit
    - **Recommendations**: Cosine similarity, Neural embeddings , Vector Emeddings
    
    **👈 Use the sidebar to navigate to the Recommendations Tab!**
    """)
    
    # Quick demo section
    st.markdown("## 🎬 Quick Demo")
    
    popular_titles = df.nlargest(10, 'Popularity Score')['Title'].tolist()
    
    selected_title = st.selectbox(
        "Choose a title to get instant recommendations:",
        ["Select a title..."] + popular_titles[:5]
    )
    
    if selected_title != "Select a title...":
        with st.spinner("🔄 Generating recommendations..."):
            recs = engine.get_recommendations(selected_title, top_k=3, method='hybrid')
            
            if recs is not None and len(recs) > 0:
                st.success(f"✨ Top recommendations for **{selected_title}**:")
                
                for idx, row in recs.iterrows():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <strong>🎬 {row['Title']}</strong><br>
                        📺 {row['Content Type']} | 🌍 {row['Language Indicator']} | 
                        ⭐ Score: {row.get('Hybrid Score', row.get('Similarity Score', 0)):.3f}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try a different title!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
       | 🎯 Trained the Model with ❤️ using TensorFlow | 
        📧 Contact: samanuel@umich.edu | 
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()