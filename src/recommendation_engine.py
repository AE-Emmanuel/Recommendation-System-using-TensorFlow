import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class NetflixRecommendationEngine:
    def __init__(self, model, df, encoders, content_representations):
        self.model = model
        self.df = df
        self.encoders = encoders
        self.content_representations = content_representations
    
    def search_content(self, query, max_results=10):
        """Search for content by title"""
        if not query:
            return pd.DataFrame()
        
        matches = self.df[self.df['Title'].str.contains(query, case=False, na=False)]
        return matches[['Title', 'Language Indicator', 'Content Type', 
                       'Hours Viewed Clean', 'Popularity Score']].head(max_results)
    
    def get_content_info(self, title):
        """Get detailed information about a specific title"""
        matches = self.df[self.df['Title'].str.contains(title, case=False, na=False)]
        if not matches.empty:
            return matches.iloc[0]
        return None
    
    def get_recommendations(self, content_title, top_k=10, method='hybrid'):
        """
        Get recommendations for a given title
        """
        # Find the content
        content_matches = self.df[self.df['Title'].str.contains(content_title, case=False, na=False)]
        
        if content_matches.empty:
            return None
        
        content_row = content_matches.iloc[0]
        content_idx = content_row.name
        
        if method == 'content':
            return self._content_based_recommendations(content_idx, top_k)
        elif method == 'collaborative':
            return self._collaborative_recommendations(content_idx, top_k)
        else:  # hybrid
            return self._hybrid_recommendations(content_idx, top_k)
    
    def _content_based_recommendations(self, content_idx, top_k):
        """Content-based recommendations using embeddings"""
        target_repr = self.content_representations[content_idx].reshape(1, -1)
        similarities = cosine_similarity(target_repr, self.content_representations)[0]
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        recommendations = self.df.iloc[similar_indices].copy()
        recommendations['Similarity Score'] = similarities[similar_indices]
        recommendations['Method'] = 'Content-Based'
        
        return recommendations[['Title', 'Language Indicator', 'Content Type', 
                               'Hours Viewed Clean', 'Popularity Score', 'Similarity Score', 'Method']]
    
    def _collaborative_recommendations(self, content_idx, top_k):
        """Collaborative filtering recommendations"""
        content_row = self.df.iloc[content_idx]
        
        same_type = self.df[self.df['Content Type Code'] == content_row['Content Type Code']]
        same_language = self.df[self.df['Language Code'] == content_row['Language Code']]
        
        candidates = pd.concat([same_type, same_language]).drop_duplicates()
        candidates = candidates[candidates.index != content_idx]
        
        recommendations = candidates.nlargest(top_k, 'Popularity Score').copy()
        recommendations['Similarity Score'] = recommendations['Popularity Score']
        recommendations['Method'] = 'Collaborative'
        
        return recommendations[['Title', 'Language Indicator', 'Content Type', 
                               'Hours Viewed Clean', 'Popularity Score', 'Similarity Score', 'Method']]
    
    def _hybrid_recommendations(self, content_idx, top_k):
        """Hybrid recommendations"""
        content_recs = self._content_based_recommendations(content_idx, top_k * 2)
        collab_recs = self._collaborative_recommendations(content_idx, top_k * 2)
        
        all_recommendations = []
        
        # Add content-based recommendations
        if content_recs is not None:
            for _, row in content_recs.iterrows():
                all_recommendations.append({
                    'Title': row['Title'],
                    'Language Indicator': row['Language Indicator'],
                    'Content Type': row['Content Type'],
                    'Hours Viewed Clean': row['Hours Viewed Clean'],
                    'Popularity Score': row['Popularity Score'],
                    'Content Score': row['Similarity Score'],
                    'Collab Score': 0,
                    'Hybrid Score': 0.6 * row['Similarity Score'],
                    'Method': 'Hybrid'
                })
        
        # Add collaborative recommendations
        if collab_recs is not None:
            for _, row in collab_recs.iterrows():
                existing = [r for r in all_recommendations if r['Title'] == row['Title']]
                if existing:
                    existing[0]['Collab Score'] = row['Similarity Score']
                    existing[0]['Hybrid Score'] = (0.6 * existing[0]['Content Score'] + 
                                                 0.4 * row['Similarity Score'])
                else:
                    all_recommendations.append({
                        'Title': row['Title'],
                        'Language Indicator': row['Language Indicator'],
                        'Content Type': row['Content Type'],
                        'Hours Viewed Clean': row['Hours Viewed Clean'],
                        'Popularity Score': row['Popularity Score'],
                        'Content Score': 0,
                        'Collab Score': row['Similarity Score'],
                        'Hybrid Score': 0.4 * row['Similarity Score'],
                        'Method': 'Hybrid'
                    })
        
        if all_recommendations:
            recommendations_df = pd.DataFrame(all_recommendations)
            recommendations_df = recommendations_df.sort_values('Hybrid Score', ascending=False).head(top_k)
            return recommendations_df[['Title', 'Language Indicator', 'Content Type', 
                                      'Hours Viewed Clean', 'Popularity Score', 'Hybrid Score', 'Method']]
        
        return pd.DataFrame()
    
    def get_trending(self, content_type=None, language=None, top_k=10):
        """Get trending content"""
        filtered_df = self.df.copy()
        
        if content_type:
            filtered_df = filtered_df[filtered_df['Content Type'] == content_type]
        
        if language:
            filtered_df = filtered_df[filtered_df['Language Indicator'] == language]
        
        trending = filtered_df.nlargest(top_k, 'Popularity Score')
        return trending[['Title', 'Language Indicator', 'Content Type', 
                        'Hours Viewed Clean', 'Popularity Score', 'Content Age']]
    
    def get_analytics_data(self):
        """Get data for analytics dashboard"""
        analytics = {
            'total_content': len(self.df),
            'content_by_type': self.df['Content Type'].value_counts(),
            'content_by_language': self.df['Language Indicator'].value_counts(),
            'popularity_distribution': self.df['Popularity Score'].describe(),
            'top_content': self.df.nlargest(10, 'Popularity Score')[['Title', 'Content Type', 'Popularity Score']],
            'content_age_distribution': self.df['Content Age'].value_counts().sort_index()
        }
        return analytics