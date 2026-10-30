"""Entry point — Project 3: Tech Stack Recommender (DecodeLabs).

Follows the IPO framework:
    INPUT:   Load data, capture user skills.
    PROCESS: TF-IDF Vectorization -> Cosine Similarity -> Sorting.
    OUTPUT:  Display Top-N matches.
"""

from src.data_loader import load_dataset
from src.recommender import TechStackRecommender

def get_user_input() -> list[str]:
    """Prompt the user for at least 3 skills."""
    print("=" * 50)
    print("  DecodeLabs Tech Stack Recommender")
    print("=" * 50)
    print("\nEnter at least 3 skills to find your ideal career path.")
    print("Example: Python, Cloud, Automation")
    print("-" * 50)

    skills = []
    while len(skills) < 3:
        skill = input(f"Skill {len(skills) + 1}: ").strip()
        if skill:
            skills.append(skill)
        else:
            print("  [X] Skill cannot be empty.")
            
    return skills

def main() -> None:
    # 1. INPUT: Load dataset
    print("\n[Loading dataset...]")
    df = load_dataset()
    
    # 2. INPUT: Capture user choices
    user_skills = get_user_input()
    
    # 3. PROCESS: Initialize engine and calculate similarities
    print(f"\n[Analyzing profile: {user_skills}]")
    engine = TechStackRecommender(df)
    
    # 4. OUTPUT: Get and display Top-N results
    recommendations = engine.recommend(user_skills, top_n=3)
    
    print("\n" + "=" * 50)
    print("  TOP 3 RECOMMENDATIONS")
    print("=" * 50)
    
    if recommendations.empty:
        print("No matches found. Try using more standard industry terms.")
    else:
        for rank, row in enumerate(recommendations.itertuples(), start=1):
            score_pct = row._2 * 100  # Convert similarity score to percentage
            print(f"  {rank}. {row._1:<25} Match: {score_pct:.1f}%")
    
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()