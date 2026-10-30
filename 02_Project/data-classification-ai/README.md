# Data Classification Using AI — DecodeLabs Project 2

Supervised learning on the Iris benchmark. This project moves from hand-written rules (Project 1) to learned patterns, implementing a complete pipeline: **EDA → Split → Scale → Tune → Compare → Evaluate**.

> *"We do not write the rules. We provide history, and the machine derives the logic."*

## 📂 Project Structure
```text
data-classification-ai/
├── main.py                 # Entry point (IPO framework)
├── requirements.txt        # Dependencies
├── pytest.ini              # Pytest configuration
├── .gitignore
├── README.md
│
├── src/
│   ├── data_loader.py      # Reads data/iris.csv (with sklearn fallback)
│   ├── preprocessor.py     # Stratified 80/20 split
│   ├── eda.py              # Numerical EDA summaries
│   ├── model.py            # CV, tuning, algorithm comparison
│   └── visualize.py        # 6 plot generation functions
│
├── tests/
│   └── test_pipeline.py    # 11 end-to-end unit tests
│
├── scripts/
│   └── export_iris.py      # One-time script to generate the CSV
│
├── docs/
│   └── PROJECT_REPORT.md   # Formal project report
│
├── data/                   # (Auto-created) iris.csv
└── outputs/                # (Auto-created) 6 saved PNG plots