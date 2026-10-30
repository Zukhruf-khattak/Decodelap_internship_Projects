# Project 3 — AI Recommendation Logic (Tech Stack Recommender)

**Author:** AI Engineering Intern  
**Organization:** DecodeLabs  
**Batch:** 2026  
**Track:** Artificial Intelligence — Personalization Phase

---

## 1. Executive Summary
Project 1 focused on deterministic logic (hand-written rules). Project 2 focused on passive classification (predicting labels from existing data). Project 3 marks a fundamental shift to **Active Prediction** and personalization. 

The goal of this project is to build a **Content-Based Filtering** recommendation engine—a "Digital Matchmaker." Specifically, we built a **Tech Stack Recommender** that ingests a user's current skills and career goals, maps them to a shared mathematical vector space, and outputs the Top-3 most relevant job roles to prevent "Choice Overload."

## 2. Dataset Overview
- **Source:** `data/raw_skills.csv` (generated via `scripts/create_dataset.py`).
- **Items:** Job Roles (e.g., Data Scientist, DevOps Engineer, Frontend Developer).
- **Features:** A string of associated skills/tags for each role (e.g., `"python, sql, machine learning, statistics"`).
- **Scale:** 9 job roles, each with a curated list of technical skills.

## 3. Methodology — The IPO Framework
The system is built strictly around the Input-Process-Output architecture:

| Phase | Implementation |
| :--- | :--- |
| **INPUT** | The user is prompted to provide a minimum of 3 skills (e.g., `["Python", "Cloud", "Automation"]`). |
| **PROCESS** | The user profile and item profiles are mapped to a shared vector space using TF-IDF. Cosine Similarity is then calculated between the user vector and every job role vector. |
| **OUTPUT** | The results are sorted in descending order and truncated to the Top-N (Top 3) to prevent information overload. |

## 4. Core Concepts & Mathematics

### 4.1 Why Content-Based Filtering?
Recommendation engines generally diverge into two methodologies:
1.  **Collaborative Filtering:** Relies on community behavior ("users who bought this also bought that"). It requires massive historical user data.
2.  **Content-Based Filtering:** Relies on item attributes. It maps a user's preferences directly to the intrinsic properties of the items.

**Project 3 uses Content-Based Filtering** because it allows us to build an effective, immediate engine without the "Cold Start" problem of needing millions of users to train on.

### 4.2 Vector Mapping & TF-IDF Weighting
Machines do not understand words like "Python" or "Web Design"; they understand numbers. We use **Term Frequency-Inverse Document Frequency (TF-IDF)** to convert text into numerical vectors.

- **Term Frequency (TF):** How often a skill appears within a specific job role.
- **Inverse Document Frequency (IDF):** Penalizes skills that appear too frequently across *all* job roles (like "software" or "git").

By multiplying these together, unique and specific skills (e.g., "Kubernetes") exert a much greater impact on the similarity math than generic skills. This solves the flaw of simple binary matching where "3 generic matches" equals "3 specific matches."

### 4.3 Cosine Similarity: The Industry Standard
Once we have weighted vectors, we must measure the "closeness" between the user and the job roles. 

**Why not Euclidean Distance?**
Euclidean distance measures the straight-line distance between two points. It is highly sensitive to magnitude. If a job description is very long and a user's input is short, their Euclidean distance will be high even if they share the exact same core skills.

**Why Cosine Similarity?**
Cosine similarity measures the *angle* between two vectors, rendering it invariant to magnitude. It focuses strictly on the orientation of the preferences. 
$$cos(\theta) = \frac{A \cdot B}{||A|| ||B||}$$
Because our TF-IDF vectors are non-negative, our scores naturally fall between `0.0` (orthogonal/no match) and `1.0` (perfect alignment). We convert this to a percentage for the final output.

## 5. The Cold Start Problem
Even the best pipeline fails without initial data. The "Cold Start" problem occurs when a system encounters a new entity.
- **Item Cold Start:** A newly added job role has no tags. *Our solution:* Content-based filtering is inherently robust here, as new roles can be recommended immediately based on their metadata (skills list).
- **User Cold Start:** A brand-new user has no history. *Our solution:* We force the ingestion step via an onboarding survey. The script requires the user to input at least 3 skills to bootstrap a baseline profile vector, bypassing the cold start.

## 6. Implementation Details (Scikit-Learn)
The engine is built using `scikit-learn`:
1.  `TfidfVectorizer(stop_words="english")` is fitted on the `Skills` column of the dataset to establish the shared vocabulary.
2.  User input is transformed using the *same* vectorizer to ensure vocabulary alignment.
3.  `cosine_similarity(user_vector, tfidf_matrix)` is used to calculate the score against every job role simultaneously.
4.  The resulting array is attached to the DataFrame, sorted descending, and filtered.

## 7. Results & Validation
Running `main.py` yields accurate, explainable results. 

*Example Run:*
- **User Input:** `["Python", "Machine Learning", "SQL"]`
- **Top 3 Output:**
  1. Data Scientist (Match: ~68%)
  2. Machine Learning Engineer (Match: ~45%)
  3. Data Analyst (Match: ~34%)

*Example Run:*
- **User Input:** `["AWS", "Docker", "Linux"]`
- **Top 3 Output:**
  1. DevOps Engineer (Match: ~55%)
  2. Cloud Architect (Match: ~48%)
  3. Cybersecurity Analyst (Match: ~22%)

Unit tests (`pytest`) verify that the engine correctly identifies the highest matching role based on overlapping skills and that it handles empty inputs safely.

## 8. Lessons Learned
- **Math over Intuition:** Binary tag matching is intuitive but flawed. TF-IDF weighting and Cosine Similarity provide a much more nuanced and professional-grade match.
- **Vocabulary Alignment:** The most common failure point in NLP recommenders is mismatched vocabularies. The user must be mapped to the *exact same* feature space as the items.
- **Preventing Overload:** A recommendation engine's job is not just to find matches, but to *filter*. Returning 50 items defeats the purpose of the "Matchmaker." The Top-N truncation is crucial for user experience.

## 9. Conclusion
Project 3 successfully transitions from passive classification to active prediction. By mastering the IPO model, vector mapping (TF-IDF), and similarity mathematics (Cosine Similarity), we have built a system capable of navigating noise and outputting flawless, relevant matches. These are the exact principles powering recommendation engines at Netflix, Amazon, and Spotify. The "Digital Matchmaker" is complete.