"""One-time script to generate data/raw_skills.csv.

Format:
    Job Role,Skills
    Data Scientist,"python, sql, machine learning, statistics, data analysis"
"""
from pathlib import Path
import pandas as pd

def main():
    out_path = Path("data") / "raw_skills.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Core dataset: Job Roles mapped to their required skills
    data = {
        "Job Role": [
            "Data Scientist",
            "Machine Learning Engineer",
            "Backend Developer",
            "Frontend Developer",
            "DevOps Engineer",
            "Cloud Architect",
            "Data Analyst",
            "Product Manager",
            "Cybersecurity Analyst"
        ],
        "Skills": [
            "python, sql, machine learning, statistics, data analysis",
            "python, deep learning, neural networks, tensorflow, pytorch, machine learning",
            "java, python, sql, apis, backend, microservices, git",
            "javascript, react, html, css, frontend, ui, ux",
            "aws, docker, kubernetes, ci cd, linux, automation, git, devops",
            "aws, cloud, architecture, kubernetes, terraform, networking",
            "sql, excel, python, data analysis, tableau, statistics",
            "agile, scrum, communication, product strategy, ui, ux",
            "networking, security, linux, python, penetration testing, cloud"
        ]
    }

    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)
    
    print(f"[OK] Wrote {len(df)} job roles to {out_path}")

if __name__ == "__main__":
    main()