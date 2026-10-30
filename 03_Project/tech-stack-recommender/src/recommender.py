"""Phase 2 & 3: PROCESS & OUTPUT — The Recommendation Engine.

Implements Content-Based Filtering:
1. Map items and user input to a shared vocabulary.
2. Apply TF-IDF weighting.
3. Calculate Cosine Similarity.
4. Sort and filter (Top-N).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TechStackRecommender:
    def __init__(self, df: pd.DataFrame):
        """Initialize with the item dataset.
        
        Args:
            df: DataFrame containing 'Job Role' and 'Skills' columns.
        """
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words="english")
        
        # Fit the vectorizer on the item skills (the vocabulary space)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["Skills"])

    def recommend(self, user_skills: list[str], top_n: int = 3) -> pd.DataFrame:
        """Generate top-N recommendations based on user skills.
        
        Args:
            user_skills: List of skills provided by the user (e.g., ["Python", "Cloud"])
            top_n: Number of recommendations to return.
            
        Returns:
            DataFrame of top-N job roles with their similarity scores.
        """
        if not user_skills:
            raise ValueError("User skills list cannot be empty.")

        # 1. Convert user input list into a single string
        user_profile = " ".join(user_skills)
        
        # 2. Transform user profile using the SAME vocabulary as the items
        user_vector = self.vectorizer.transform([user_profile])
        
        # 3. Calculate Cosine Similarity between user and ALL items
        # Result is a 2D array, so we flatten it to 1D
        sim_scores = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        # 4. Attach scores to the dataframe
        results = self.df.copy()
        results["Match Score"] = sim_scores
        
        # 5. Sort descending by score
        results = results.sort_values(by="Match Score", ascending=False)
        
        # 6. Filter: Top-N and remove scores of exactly 0 (no match)
        top_results = results[results["Match Score"] > 0].head(top_n)
        
        return top_results[["Job Role", "Match Score"]]