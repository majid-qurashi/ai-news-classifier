# AI News Classifier 📰

A modern, machine learning-powered application that classifies news headlines and descriptions into four major categories: **Business**, **Science/Technology**, **Sports**, and **World**.

## 📁 Project Structure

```text
📂 news-classifier/
├── 📁 src/                 # Backend logic & ML scripts
│   ├── 📄 app.py           # Flask main application file
│   └── 📄 model.py         # Model training script
├── 📁 templates/           # Frontend user interface
│   └── 📄 index.html       # News input & classification results
├── 📁 static/              # Assets & Styling
│   ├── 📄 style.css        # Glassmorphic UI styles
│   └── 📄 favicon.svg      # Custom SVG favicon icon
├── 📄 train.csv            # Training dataset
├── 📄 test.csv             # Evaluation dataset
├── 📄 model_pipeline.pkl   # Serialized ML model pipeline
├── 📄 pyproject.toml       # Project dependencies & configuration
├── 📄 .gitignore           # Version control ignore list
└── 📄 README.md            # Project guide & details
```

## 🚀 Getting Started

### 1. Run Application
To start the Flask development server:
```sh
py -m uv run src/app.py
```

### 2. Train Model
To re-train or update the classification model:
```sh
py -m uv run src/model.py
```

## 🛠️ Tech Stack
* **Machine Learning**: Scikit-Learn (TF-IDF, SVM/Logistic Regression)
* **Backend**: Flask (Python)
* **Frontend**: HTML5, CSS3 (Modern Glassmorphic Design)
* **Package Manager**: UV

---
### ✒️ About Developer
* **Developed by**: Majid Qurashi
* **ID**: B-Tech CSE [220365]
* **Portfolio**: [qurashi.vercel.app](https://qurashi.vercel.app)

> [!NOTE]
> The classification pipeline follows a strict flow:
> **Dataset → Preprocessing → Feature Extraction → Model Training → Prediction**
