# CloudWalk-Digit-Classifier-

This project builds a **spoken digit classifier (0–9)** using the open-source **Free Spoken Digit Dataset (FSDD)** and **classical machine‑learning models**. Audio waveforms are converted to **MFCC features** and evaluated with **Random Forest**, **Decision Tree**, **XGBoost**, and **Bagging (DecisionTree base)** classifiers. Confusion matrices and classification reports are produced to compare models.

> Source notebook: `CloudWalk_Digit_Classification_from_Audio.ipynb`

---

## 📦 What’s inside
- **Dataset**: Automatically cloned FSDD (`Jakobovski/free-spoken-digit-dataset`)
- **Preprocessing**: `librosa` to load audio at 8 kHz and extract mean‑pooled **MFCC** vectors
- **Models**: `RandomForestClassifier`, `DecisionTreeClassifier`, `xgboost.XGBClassifier`, `BaggingClassifier(estimator=DecisionTreeClassifier)`
- **Evaluation**: `train_test_split`, **classification_report**, **confusion_matrix** visualized with `seaborn`

---

## 🗂 Dataset
- **Free Spoken Digit Dataset (FSDD)** — recordings of spoken digits (0–9) by multiple speakers.
- In the notebook, the dataset is cloned and read from:
  ```text
  free-spoken-digit-dataset/recordings/
  ```
- FSDD repository: https://github.com/Jakobovski/free-spoken-digit-dataset  
  Please review the repo for **license / attribution**.

---

## 🔧 Requirements
Create an environment (Python 3.9+ recommended) and install the following:
```bash
pip install numpy pandas matplotlib seaborn librosa scikit-learn xgboost
```
> On some systems, `librosa` requires `soundfile`/`libsndfile`:
> ```bash
> pip install soundfile
> # macOS (Homebrew): brew install libsndfile
> # Linux (apt):      sudo apt-get install -y libsndfile1
> ```

---

## 📊 Results Summary

### Random Forest
- **Accuracy:** ~0.95  
- Strong, balanced performance across all digits  
- Precision/Recall/F1 mostly **≥0.93** for digits `0–9`  
- Slightly lower recall for **digit 3 (0.92)** and **digit 6 (0.87)**  
- **Overall best-performing model**  

### Decision Tree
- **Accuracy:** ~0.79  
- Noticeably weaker than Random Forest  
- Struggles on **digit 3 (F1 ≈ 0.64)**  
- Digits `4, 5, 8` perform better (>0.80 F1), but overall **less consistent**  

### XGBoost
- **Accuracy:** ~0.93  
- Very competitive with Random Forest  
- Better precision/recall balance on **digit 3** than Decision Tree  
- Overall second-best performer  

### Bagging (Decision Tree Base)
- Performs **between Decision Tree and Random Forest**  
- Improves stability compared to a single tree, but does not surpass Random Forest  

---

## 🏆 Key Takeaways
- **Random Forest is the top performer** (~95% accuracy, balanced across digits)  
- **XGBoost is close behind** (strong alternative for difficult digits)  
- **Decision Tree alone is not sufficient** (~79% accuracy)  
- **Bagging helps, but Random Forest remains the most reliable model**  



## 🚀 How to run (Notebook)
1. Open `CloudWalk_Digit_Classification_from_Audio.ipynb` in Jupyter/Colab.
2. Run the **“Cloning the repository”** cell to fetch FSDD.
3. Run the import, dataset loading, feature extraction, and model cells in order.
4. Compare models using the printed **accuracy**/**classification report** and the **confusion matrices**.
