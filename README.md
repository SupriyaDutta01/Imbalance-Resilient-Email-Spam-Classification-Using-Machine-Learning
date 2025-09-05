# 📧 Imbalance-Resilient Email Spam Classification Using Machine Learning

## 📌 Project Overview
This project focuses on **Spam Email/SMS Detection** while handling **imbalanced textual data**.  

- **Ham (0)** = Legitimate email (negative class)  
- **Spam (1)** = Spam email (positive class)  

The primary goal is to **minimize False Positives (ham → spam misclassification)** since marking a legitimate email as spam is more harmful than missing a spam message.  

---

## 🎯 Objectives
- Build robust **spam classifiers** for imbalanced datasets.  
- Improve **precision (close to 1.0)** while maintaining accuracy & recall.  
- Leverage **data augmentation** to generate realistic spam variations.  
- Compare multiple ML models and finalize using **Voting & Stacking** ensembles.  

---

## 📊 Confusion Matrix Definition
For this project:  

- **TP (True Positive):** Spam correctly predicted as spam  
- **FP (False Positive):** Ham wrongly predicted as spam 🚫 (minimize this)  
- **FN (False Negative):** Spam wrongly predicted as ham  
- **TN (True Negative):** Ham correctly predicted as ham  

👉 **Our focus:** Reduce **FP** → Ensure **precision ≈ 1.0**

---

## 🛠 Techniques Used

### 🔹 Handling Imbalance
- Random Oversampling  
- Class Weight Balancing  

### 🔹 Data Augmentation
1. **BERT-based Contextual Augmentation**  
   - Mask → Predict → Replace strategy with `transformers`  
2. **Back Translation**  
   - Translate (EN → FR/DE/ES/IT) → Back to EN  
3. **LLM-based Paraphrasing**  
   - Using **LLaMA3 via Ollama** to generate semantically valid variants  

### 🔹 Feature Engineering
- **TF-IDF Vectorization** (unigrams + bigrams)  

### 🔹 Models Evaluated
- SVC  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Bagging Classifier  
- Extra Trees  
- Gradient Boosting (GBDT)  
- Naïve Bayes variants (Multinomial, Bernoulli, Gaussian)  
- XGBoost  

### 🔹 Ensemble
- Best performing models selected for **Voting** and **Stacking** classifiers  

---

## ⚙️ Tech Stack
- **Python**  
- **Libraries:**  
  - `scikit-learn`, `xgboost`  
  - `transformers`, `googletrans`, `ollama`  
  - `pandas`, `numpy`  
  - `matplotlib`, `seaborn`  
  - `nltk`, `re`  

---

