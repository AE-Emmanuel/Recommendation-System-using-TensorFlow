import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_loader import load_model_and_data
from recommendation_engine import NetflixRecommendationEngine

# Page config
st.set_page_config(
    page_title="AI Recommendations - Netflix System",
    page_icon="🎬",
    layout="wide"
)

# Simple CSS for containers only
st.markdown("""
<style>
    .search-container {
        margin: 2rem auto;
        max-width: 800px;
        text-align: center;
    }
    .settings-container {
        background: #f8f9fa;
        border: 2px solid #E50914;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .method-indicator {
        display: inline-block;
        background: #E50914;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_recommendation_engine():
    """Load and cache the recommendation engine"""
    model, df, encoders, representations = load_model_and_data()
    if model is not None:
        return NetflixRecommendationEngine(model, df, encoders, representations), df
    return None, None

def display_content_info(content_info):
    """Display content info using Streamlit native components"""
    
    # Main title with background
    st.success(f"🎯 **Now Showing Recommendations For:** {content_info['Title']}")
    
    # Content details in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📺 Type", content_info['Content Type'])
    
    with col2:
        st.metric("🌍 Language", content_info['Language Indicator'])
    
    with col3:
        st.metric("👀 Hours Viewed", f"{content_info['Hours Viewed Clean']:,}")
    
    with col4:
        st.metric("⭐ Popularity", f"{content_info['Popularity Score']:.3f}")
    
    with col5:
        st.metric("📅 Age", f"{content_info['Content Age']} years")
    
    st.divider()

def display_recommendation_card(rec, index):
    """Display a single recommendation using Streamlit native components"""
    method = rec.get('Method', 'Unknown')
    score_column = 'Hybrid Score' if 'Hybrid Score' in rec else 'Similarity Score'
    score = rec.get(score_column, 0)
    
    # Use Streamlit's container for clean layout
    with st.container():
        # Header row with title and score
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"#{index + 1} {rec['Title']}")
        
        with col2:
            st.metric("Score", f"{score:.3f}")
        
        # Method and content info row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Method badge using Streamlit's built-in styling
            if method == 'Hybrid':
                st.success(f"🔥 {method}")
            elif method == 'Content-Based':
                st.info(f"📚 {method}")
            else:
                st.warning(f"🤝 {method}")
        
        with col2:
            st.write(f"📺 **{rec['Content Type']}**")
        
        with col3:
            st.write(f"🌍 **{rec['Language Indicator']}**")
        
        # Stats row
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"👀 **{rec['Hours Viewed Clean']:,}** hours viewed")
        
        with col2:
            st.write(f"⭐ **Popularity:** {rec['Popularity Score']:.3f}")
        
        # Clean separator
        st.divider()

def create_recommendation_chart(recommendations, method):
    """Create a chart showing recommendation scores"""
    if recommendations is None or len(recommendations) == 0:
        return None
    
    score_column = 'Hybrid Score' if 'Hybrid Score' in recommendations.columns else 'Similarity Score'
    
    fig = px.bar(
        recommendations.head(8),
        x=score_column,
        y='Title',
        orientation='h',
        title=f'{method} Recommendation Scores',
        color=score_column,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig

def main():
    st.title("🎬 Netflix Recommendation System")
    st.markdown("**Search for content to Know How the Model Works!**")
    st.markdown("--This is a Demo for the ML model and not an actual app--")
    # Load recommendation engine
    engine, df = get_recommendation_engine()
    
    if engine is None:
        st.error("❌ Failed to load recommendation system. Please check your setup.")
        return
    
    # Initialize session state
    if 'selected_content' not in st.session_state:
        st.session_state.selected_content = None
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False
    if 'rec_method' not in st.session_state:
        st.session_state.rec_method = 'hybrid'
    if 'num_recs' not in st.session_state:
        st.session_state.num_recs = 10
    if 'content_filter' not in st.session_state:
        st.session_state.content_filter = 'All'
    if 'language_filter' not in st.session_state:
        st.session_state.language_filter = 'All'
    
    # Method display mapping
    method_display = {
        'hybrid': '🔥 Hybrid',
        'content': '📚 Content',
        'collaborative': '🤝 Collaborative'
    }
    
    # Center search interface
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    # Search input with settings button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Create list of all titles for selectbox - FIXED PLACEHOLDER
        all_titles = df['Title'].tolist()
        
        # Dynamic selectbox search with proper placeholder
        selected_title = st.selectbox(
            "🔍 Search Netflix Content",
            options=[None] + all_titles,  # None as first option for placeholder
            format_func=lambda x: "🔍 Start typing to search Netflix content..." if x is None else x,
            key="title_search",
            help="Type to search through Netflix titles",
            index=0  
        )
        
        # Show method indicator
        st.markdown(f'<div class="method-indicator">{method_display[st.session_state.rec_method]}</div>', 
                   unsafe_allow_html=True)
    
    with col2:
        # Settings toggle button
        if st.button("⚙️ Settings", key="settings_toggle"):
            st.session_state.show_settings = not st.session_state.show_settings
    
    # Settings container (appears when button clicked)
    if st.session_state.show_settings:
        st.markdown('<div class="settings-container">', unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Recommendation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.rec_method = st.selectbox(
                "🤖 Method:",
                ["hybrid", "content", "collaborative"],
                format_func=lambda x: method_display[x],
                index=["hybrid", "content", "collaborative"].index(st.session_state.rec_method),
                key="method_setting"
            )
            
            st.session_state.num_recs = st.slider(
                "📊 Number of Results:",
                3, 20, st.session_state.num_recs,
                key="num_setting"
            )
        
        with col2:
            st.session_state.content_filter = st.selectbox(
                "📺 Content Type:",
                ["All", "Show", "Movie"],
                index=["All", "Show", "Movie"].index(st.session_state.content_filter),
                key="content_setting"
            )
            
            st.session_state.language_filter = st.selectbox(
                "🌍 Language:",
                ["All", "English", "Korean", "Japanese", "Hindi", "Spanish"],
                index=["All", "English", "Korean", "Japanese", "Hindi", "Spanish"].index(st.session_state.language_filter),
                key="language_setting"
            )
        
        # Apply button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("✅ Apply Settings", key="apply_settings", type="primary"):
                st.session_state.show_settings = False
                st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle title selection
    if selected_title is not None:  # Check for None instead of empty string
        # Get content info
        selected_content = engine.get_content_info(selected_title)
        
        if selected_content is not None:
            st.session_state.selected_content = selected_content
            
            # Display selected content info
            display_content_info(selected_content)
            
            # Generate recommendations
            with st.spinner(f"🤖 Generating {method_display[st.session_state.rec_method]} recommendations..."):
                recommendations = engine.get_recommendations(
                    selected_content['Title'], 
                    top_k=st.session_state.num_recs, 
                    method=st.session_state.rec_method
                )
            
            if recommendations is not None and len(recommendations) > 0:
                # Apply filters if specified
                if st.session_state.content_filter != "All":
                    recommendations = recommendations[recommendations['Content Type'] == st.session_state.content_filter]
                
                if st.session_state.language_filter != "All":
                    recommendations = recommendations[recommendations['Language Indicator'] == st.session_state.language_filter]
                
                if len(recommendations) > 0:
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "📊 Analytics", "🔄 Compare Methods"])
                    
                    with tab1:
                        st.markdown(f"## ✨ Your {method_display[st.session_state.rec_method]} Recommendations")
                        
                        # Display recommendation cards
                        for idx, (_, rec) in enumerate(recommendations.iterrows()):
                            display_recommendation_card(rec, idx)
                    
                    with tab2:
                        st.markdown("## 📊 Recommendation Analytics")
                        
                        # Create and display chart
                        chart = create_recommendation_chart(recommendations, st.session_state.rec_method.title())
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        
                        # Score statistics
                        score_column = 'Hybrid Score' if 'Hybrid Score' in recommendations.columns else 'Similarity Score'
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("🏆 Best Score", f"{recommendations[score_column].max():.3f}")
                        with col2:
                            st.metric("📊 Avg Score", f"{recommendations[score_column].mean():.3f}")
                        with col3:
                            st.metric("🌍 Languages", recommendations['Language Indicator'].nunique())
                        with col4:
                            st.metric("📺 Content Types", recommendations['Content Type'].nunique())
                    
                    with tab3:
                        st.markdown("## 🔄 Compare Different Methods")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        methods = ['hybrid', 'content', 'collaborative']
                        method_names = ['🔥 Hybrid', '📚 Content-Based', '🤝 Collaborative']
                        
                        for i, (method, name) in enumerate(zip(methods, method_names)):
                            with [col1, col2, col3][i]:
                                current = "(Current)" if method == st.session_state.rec_method else ""
                                
                                if st.button(f"{name} {current}", key=f"switch_{method}", disabled=(method == st.session_state.rec_method)):
                                    st.session_state.rec_method = method
                                    st.experimental_rerun()
                                
                                # Show quick preview for other methods
                                if method != st.session_state.rec_method:
                                    preview_recs = engine.get_recommendations(
                                        selected_content['Title'], 
                                        top_k=3, 
                                        method=method
                                    )
                                    
                                    if preview_recs is not None and len(preview_recs) > 0:
                                        st.markdown("**Preview:**")
                                        for idx, (_, rec) in enumerate(preview_recs.iterrows()):
                                            score = rec.get('Hybrid Score', rec.get('Similarity Score', 0))
                                            st.write(f"{idx+1}. {rec['Title'][:25]}... ({score:.2f})")
                
                else:
                    st.warning("😕 No recommendations match your current filters. Try adjusting the settings.")
            
            else:
                st.warning("😕 No recommendations found for this content. Try a different title.")
        
        else:
            st.error("Content not found in database.")
    
    else:
        # Show instructions when no content is selected
        st.markdown("---")
        st.markdown("### 🎯 How to Use the Netflix AI Recommendation System:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔍 Getting Recommendations:**
            1. **Click the search dropdown** above
            2. **Start typing** any Netflix title  
            3. **Select** from the filtered results
            4. **Get instant** AI-powered recommendations!
            
            **💡 Search Tips:**
            - Type movie or show names
            - Try partial titles (e.g., "Wed" for Wednesday)
            - Search works with 19,000+ Netflix titles
            """)
        
        with col2:
            st.markdown("""
            **⚙️ Customization Options:**
            - **Click Settings** to adjust preferences
            - **Try different methods**: Hybrid, Content-Based, Collaborative
            - **Filter results** by content type or language
            - **Adjust number** of recommendations (3-20)
            
            **🎯 Recommendation Methods:**
            - **🔥 Hybrid**: Best of both worlds (recommended)
            - **📚 Content-Based**: Similar content features
            - **🤝 Collaborative**: Popular patterns
            """)
        
        # Quick demo section
        st.markdown("---")
        st.markdown("### ✨ Try These Examples:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Search for:**\n- Wednesday\n- Money Heist\n- Squid Game")
        
        with col2:
            st.info("**Try Methods:**\n- Hybrid (best results)\n- Content-Based\n- Collaborative")
        
        with col3:
            st.info("**Filter by:**\n- Movies vs Shows\n- Language preference\n- Number of results")        
if __name__ == "__main__":
    main()