<div align="center">

# 📰 AI Fake News Detector

### Paste a headline or article. Get an instant **Real vs Fake** verdict with a confidence score.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

[**Features**](#-features) · [**How It Works**](#-how-it-works) · [**Quick Start**](#-quick-start) · [**Project Structure**](#-project-structure) · [**Roadmap**](#-roadmap)

</div>

---

## 🎯 Overview

Misinformation spreads faster than the truth. **AI Fake News Detector** is a machine learning project that classifies news text as **Real** or **Fake** using Natural Language Processing, wrapped in a clean, interactive **Streamlit** web app anyone can use, no coding required.

The repo includes the full experimentation notebook, trained model files, and a ready-to-run web interface.

<!-- 📸 Add a screenshot or GIF of your app here:
<p align="center"><img src="assets/demo.png" alt="App demo" width="700"/></p>
-->

## ✨ Features

- 🧠 **Deep learning + classical ML**: supports a trained **Keras** model and a **TF-IDF + Logistic Regression** pipeline
- ⚡ **Instant predictions**: type a headline, optionally paste the article body, click once
- 📊 **Confidence scores**: see the Real and Fake probability, plus a visual progress bar
- 🛡️ **Graceful model fallback**: the app loads the Keras model first, then the joblib pipeline, and rebuilds a lightweight model automatically if neither is available
- 🚀 **Cached model loading**: fast responses via Streamlit's resource caching
- 📓 **Reproducible**: full training and evaluation workflow in a Jupyter notebook

## 🔬 How It Works

```
 ┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐    ┌────────────────┐
 │ Headline +   │ -> │ Text Vectorization│ -> │ Trained Classifier│ -> │ Real / Fake +  │
 │ Article Text │    │ (TF-IDF / Keras)  │    │ (Keras / LogReg) │    │ Confidence %   │
 └──────────────┘    └───────────────────┘    └──────────────────┘    └────────────────┘
```

1. **Input**: the headline and optional article body are combined into a single text.
2. **Vectorization**: text is converted to numerical features (TF-IDF with unigrams and bigrams for the scikit-learn pipeline).
3. **Classification**: the model outputs the probability that the news is **Real**.
4. **Verdict**: probability above 50% is labeled **Real**; otherwise **Fake**, with the confidence shown.

### Model loading priority

| Priority | File | Type |
|:-:|---|---|
| 1 | `fake_news_model.keras` | TensorFlow / Keras model |
| 2 | `fake_news_model.joblib` | scikit-learn TF-IDF + Logistic Regression pipeline |
| 3 | *(auto-built)* | Lightweight fallback so the app always runs |

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Machine Learning | scikit-learn (TF-IDF, Logistic Regression) |
| Data | NumPy, Pandas |
| Visualization | Matplotlib |
| Web App | Streamlit |
| Model Persistence | joblib, `.keras` |

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/tafrusaidev/Fake-News-Classifier-Model.git
cd Fake-News-Classifier-Model
```

### 2. Create a virtual environment *(recommended)*

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** 🎉

## 💡 Usage

1. Enter a **news headline** in the first field.
2. *(Optional)* Paste the **full article body** for better context.
3. Click **Check News**.
4. Read the verdict:
   - ✅ **REAL NEWS**: with confidence percentage
   - 🚨 **FAKE NEWS**: with confidence percentage

## 📁 Project Structure

```
Fake-News-Classifier-Model/
├── 📓 Fake_News_Classifier (1).ipynb   # Data exploration, training & evaluation
├── 🌐 app.py                           # Streamlit web application
├── 🤖 fake_news_model.keras            # Trained Keras model
├── 📦 fake_news_model.joblib           # Trained scikit-learn pipeline
├── 📋 requirements.txt                 # Python dependencies
└── 📄 README.md                        # You are here
```

## 📈 Model Performance

> Fill this section in with the results from your notebook.

| Metric | Score |
|---|:-:|
| Accuracy | `XX.XX%` |
| Precision | `XX.XX%` |
| Recall | `XX.XX%` |
| F1-Score | `XX.XX%` |

**Dataset:** *add the dataset name and source link here (e.g., Kaggle) and the number of articles used.*

## ⚠️ Limitations & Responsible Use

- This tool gives a **statistical prediction based on writing patterns**, not a fact-check. It does not verify claims against real-world sources.
- Performance depends on the training data. Results may be weaker on topics, regions, or writing styles the model hasn't seen.
- Always cross-check important news with **trusted, independent sources**.

## 🗺️ Roadmap

- [ ] Deploy the app on Streamlit Community Cloud
- [ ] Add URL input with automatic article extraction
- [ ] Experiment with transformer models (BERT / DistilBERT)
- [ ] Add explainability (highlight the words that influenced the prediction)
- [ ] Add a confusion matrix and ROC curve to the app
- [ ] Multi-language support

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m "Add AmazingFeature"`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. Add a `LICENSE` file to the repo to make it official.

## 👤 Author

**tafrusaidev**

[![GitHub](https://img.shields.io/badge/GitHub-tafrusaidev-181717?style=for-the-badge&logo=github)](https://github.com/tafrusaidev)

---

<div align="center">

⭐ **If you found this project useful, please give it a star!** ⭐

</div>
