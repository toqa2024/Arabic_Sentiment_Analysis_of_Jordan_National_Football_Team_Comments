# 🇯🇴 Jordan Football Sentiment Analysis

**Arabic NLP | Transformer Fine-Tuning | Model Evaluation | Explainable AI | Streamlit**

An end-to-end Natural Language Processing (NLP) project that analyzes Arabic public comments about the Jordan national football team and classifies them into three sentiment categories: **Pride, Anger, and Neutral**.

This project compares two pretrained Arabic language models, **MARBERT** and **AraBERT**, and explores their predictions through class-level evaluation, TF-IDF analysis, and SHAP-based explainability.

The goal is not only to build a sentiment classifier, but also to establish a reproducible workflow for Arabic text classification, analyze model limitations, and prepare the selected model for interactive inference.

---

## 📌 Project Overview

Football discussions on social media frequently contain informal Arabic, regional dialects, emotional expressions, sarcasm, and context-dependent opinions.

These characteristics make Arabic sentiment classification challenging, particularly when distinguishing neutral comments from emotionally charged opinions.

This project explores the following questions:

- How effectively can pretrained Arabic transformers classify football-related comments?
- How do MARBERT and AraBERT compare on the same dataset?
- Which sentiment categories are easier or more difficult to predict?
- Which words and text segments contribute to individual predictions?
- How can the trained model be prepared for practical use?

### Project Highlights

- Three-class Arabic sentiment classification.
- Comparison of MARBERT and AraBERT.
- Stratified train, validation, and test splits.
- Training-only oversampling to address class imbalance.
- Hyperparameter experiments using validation macro F1-score.
- Class-level evaluation and error analysis.
- TF-IDF exploration of sentiment-related vocabulary.
- SHAP-based investigation of individual predictions.
- A structured pipeline designed to support reproducibility and future deployment.

---

## 🖼️ Project Preview

<!-- Add a real screenshot of your application here when available. -->

<!--
![Application Preview](reports/figures/app_preview.png)
-->

*An interactive application preview can be added after the inference interface has been implemented.*

---

## 🔄 Project Pipeline

The project follows a structured workflow from data loading to model inference and explainability.

![Project Pipeline](reports/figures/project_pipeline.png)

### Pipeline Stages

**1. Data Loading**

Load the cleaned Arabic comments and their corresponding sentiment labels.

**2. Data Validation and Exploration**

- Inspect missing values and duplicate comments.
- Review class distribution.
- Examine representative examples from each class.
- Check for potential duplicate or near-duplicate leakage across data splits.

**3. Data Splitting**

Split the dataset into training, validation, and test sets using stratified sampling:

- Training: 70%
- Validation: 15%
- Test: 15%

A fixed random seed of `42` is used for reproducibility.

**4. Training Data Balancing**

Apply oversampling to the training set only. The validation and test sets retain their original class distributions to provide a more realistic evaluation.

**5. Tokenization and Model Training**

Tokenize the Arabic comments using each model's corresponding tokenizer, then fine-tune MARBERT and AraBERT for three-class sequence classification.

**6. Model Selection**

Compare candidate configurations using validation macro F1-score. The test set is reserved for final evaluation rather than repeated hyperparameter selection.

**7. Final Evaluation and Error Analysis**

Evaluate the selected configurations on the held-out test set using accuracy, precision, recall, F1-score, and confusion matrices. Inspect misclassified examples to understand model limitations.

**8. Explainability**

Use TF-IDF to explore class-associated vocabulary and SHAP to investigate how the model responds to individual comments.

**9. Inference and Deployment**

Prepare a reusable prediction interface and, as a deployment extension, an interactive Streamlit application.

> **Implementation note:** The pipeline describes the project's workflow. Any stage not yet implemented as a reusable script or deployed application should be considered planned work rather than a completed feature.

---

## 🎯 Sentiment Categories

| Class | Description |
|---|---|
| **Pride** | Support, congratulations, positive reactions, and pride in the team. |
| **Anger** | Criticism, frustration, disappointment, and complaints. |
| **Neutral** | Informational or descriptive comments with relatively neutral sentiment. |

These categories represent the labels available in the dataset; actual comments may contain mixed emotions or ambiguous sentiment.

---

## 📊 Dataset

The dataset contains **2,911 cleaned Arabic comments**.

### Class Distribution

| Sentiment | Number of Comments | Percentage |
|---|---:|---:|
| Pride | 1,741 | 59.8% |
| Anger | 652 | 22.4% |
| Neutral | 518 | 17.8% |
| **Total** | **2,911** | **100%** |

![Dataset Class Distribution](reports/figures/class_distribution.png)

### Dataset Columns

| Column | Description |
|---|---|
| `Cleaned_Comment` | Preprocessed Arabic comment. |
| `Class` | Sentiment label. |

### Data Considerations

The dataset is imbalanced, with Pride representing the majority class. This can cause a model to perform better on Pride than on less frequent classes.

The project addresses this issue through training-set oversampling and evaluates each class separately rather than relying on accuracy alone.

**Data availability:** The raw dataset is not included by default. Before publishing it, verify that its source, collection method, and sharing permissions allow redistribution. If redistribution is not permitted, provide instructions for obtaining the data legitimately or keep it outside the repository.

---

## 🧠 Models

Two pretrained Arabic language models were fine-tuned and compared.

### 1. MARBERT

Model: [`UBC-NLP/MARBERT`](https://huggingface.co/UBC-NLP/MARBERT)

MARBERT is an Arabic pretrained language model designed with a focus on Arabic social media language, making it a relevant candidate for informal football discussions.

### 2. AraBERT

Model: [`aubmindlab/bert-base-arabertv02`](https://huggingface.co/aubmindlab/bert-base-arabertv02)

AraBERT is a pretrained Arabic language model used as a comparison baseline for the sentiment classification task.

### Training Configuration

The selected configurations used the following hyperparameters:

| Hyperparameter | Value |
|---|---|
| Learning rate | `2e-5` |
| Batch size | `16` |
| Epochs | `3` |
| Maximum sequence length | `96` |
| Weight decay | `0.02` |
| Random seed | `42` |
| Training data balancing | Oversampling |

Both models were evaluated using the same overall split strategy and comparable training configurations.

Oversampling was applied only to the training set. Validation and test data retained the original class distribution.

---

## 📈 Experimental Results

### Model Comparison

The table below summarizes the selected configurations.

| Metric | AraBERT | MARBERT |
|---|---:|---:|
| Validation Accuracy | 73.91% | 77.80% |
| Validation Macro F1 | 0.676 | **0.717** |
| Validation Weighted F1 | 0.743 | **0.779** |
| Test Accuracy | 76% | 77% |
| Test Macro F1 | 0.68 | **0.69** |
| Test Weighted F1 | 0.76 | Approximately 0.76–0.77 |

![Model Comparison](reports/figures/model_comparison.png)

**MARBERT was selected as the preferred configuration based on validation macro F1-score.**

It achieved a validation macro F1-score of approximately `0.717` and a test macro F1-score of approximately `0.69`.

The results suggest that MARBERT is a promising choice for this dataset. However, the test performance also shows that meaningful classification errors remain, especially for the Neutral class.

*Reported metrics are rounded. The original evaluation reports should be treated as the source of truth when reproducing the experiments.*

### Why Macro F1 Matters

Accuracy measures the overall proportion of correct predictions, but it can hide poor performance on minority classes.

Macro F1 calculates the F1-score independently for each class and then averages the results equally. This makes it particularly useful when comparing models trained on imbalanced data.

---

## 🔬 MARBERT: Test Set Evaluation

The following table summarizes the selected MARBERT configuration on the held-out test set.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Anger | 0.65 | 0.67 | 0.66 | 98 |
| Neutral | 0.55 | 0.51 | 0.53 | 78 |
| Pride | 0.87 | 0.88 | 0.88 | 261 |
| **Macro Average** | **0.69** | **0.69** | **0.69** | **437** |

![MARBERT Confusion Matrix](reports/figures/confusion_matrix_marbert.png)

### Interpretation

- **Pride:** The strongest class, with an F1-score of `0.88`.
- **Anger:** Moderate performance, with an F1-score of `0.66`.
- **Neutral:** The most challenging class, with an F1-score of `0.53`.

The difference in performance across classes suggests that the model can identify explicit expressions of support and pride more reliably than subtle or ambiguous neutral opinions.

---

## 📝 TF-IDF: Exploring Class-Associated Vocabulary

TF-IDF was used to identify terms and phrases that are prominent in comments belonging to each sentiment class.

### Example Terms

| Sentiment | Example Terms |
|---|---|
| Pride | `النشامي`, `مبروك`, `منتخبنا`, `ابطال` |
| Anger | `الحكم`, `ضعيف`, `اخطاء`, `ظلم` |
| Neutral | `الارجنتين`, `الاداء`, `الحكم`, `مستوي` |

![TF-IDF Class Terms](reports/figures/tfidf_class_terms.png)

These terms provide an exploratory view of the language used in the dataset.

TF-IDF is used for vocabulary analysis and interpretation; the transformer models are responsible for making the final sentiment predictions.

Some terms can appear across multiple classes. Their meaning depends on the surrounding context, which is one reason a contextual model can be useful for this task.

---

## 🧩 Explainability with SHAP

SHAP was used to investigate individual predictions and explore which input tokens contributed to the model's output.

The analysis includes representative examples from Pride, Anger, and Neutral comments.

![SHAP Explanation Example](reports/figures/shap_example.png)

### What the Analysis Shows

- The model can assign high predicted probabilities to individual comments.
- Different text segments may contribute differently to a selected class prediction.
- SHAP provides a way to inspect model behavior beyond aggregate evaluation metrics.

### Important Limitations

Arabic tokenization can split words into subword fragments. Consequently, SHAP plots may display fragments rather than complete Arabic words.

These explanations should be interpreted as model-level attribution estimates, not as definitive linguistic explanations or proof of causality.

A high softmax probability also does not guarantee that a prediction is correct or calibrated.

---

## ⚠️ Error Analysis and Limitations

The evaluation highlights several limitations that should guide future improvements.

### 1. Neutral Classification

Neutral has the lowest test F1-score (`0.53`). Neutral comments may contain criticism, match analysis, indirect opinions, or mixed sentiment that makes classification difficult.

### 2. Class Imbalance

Pride accounts for approximately 60% of the dataset. Training-set oversampling helps address the imbalance, but it does not create new linguistic diversity.

### 3. Dialectal Arabic

Informal Arabic, regional dialects, spelling variation, and social media expressions can affect model performance.

### 4. Sarcasm and Context

A comment may use positive-looking words sarcastically or express frustration indirectly. Such cases require contextual interpretation.

### 5. Dataset Size

The dataset contains 2,911 comments. A larger and more diverse dataset could improve generalization.

### 6. Explainability and Tokenization

Subword fragmentation can make SHAP explanations difficult to interpret at the complete-word level.

### 7. Potential Data Leakage

Duplicate and near-duplicate comments should be checked across training, validation, and test splits. Similar comments appearing in multiple splits may lead to overly optimistic evaluation.

These limitations are important when interpreting the reported results and planning future experiments.

---

## 🚀 Inference Pipeline

The intended inference workflow is:

1. Receive an Arabic comment.
2. Apply the same text preparation rules used during training, where applicable.
3. Tokenize the comment using the selected model's tokenizer.
4. Run the fine-tuned MARBERT classifier.
5. Map the predicted class ID to its sentiment label.
6. Return the predicted class and its model score.

The prediction interface can be implemented as a reusable Python function and exposed through a Streamlit application.

**Reproducibility requirement:** The model checkpoint, tokenizer, label mappings, and preprocessing configuration must remain consistent. Do not apply training-set oversampling during inference.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Deep Learning:** PyTorch
- **Transformer Models:** Hugging Face Transformers
- **Arabic NLP:** MARBERT, AraBERT
- **Text Analysis:** TF-IDF
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Development Environment:** Jupyter Notebook
- **Planned Interactive Interface:** Streamlit

---

## 📂 Repository Structure

A suggested repository structure is shown below. Adjust the files to match the implementation actually included in the repository.

```text
jordan-football-sentiment-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_and_shap.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── data_splitting.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── README.md
│
├── reports/
│   └── figures/
│       ├── project_pipeline.png
│       ├── class_distribution.png
│       ├── model_comparison.png
│       ├── confusion_matrix_marbert.png
│       ├── tfidf_class_terms.png
│       └── shap_example.png
│
└── app/
    └── streamlit_app.py
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/jordan-football-sentiment-analysis.git
cd jordan-football-sentiment-analysis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain the actual dependencies used by the project, ideally with tested versions.

---

## ▶️ Running the Project

The notebooks document the main experimental workflow:

1. Explore and validate the dataset.
2. Create stratified training, validation, and test splits.
3. Train and compare MARBERT and AraBERT.
4. Evaluate the selected model.
5. Explore TF-IDF terms and SHAP explanations.

If a reusable inference script or Streamlit application is included, document its actual execution command here.

For example, once `app/streamlit_app.py` is implemented:

```bash
streamlit run app/streamlit_app.py
```

Model checkpoints and dataset files may need to be downloaded or configured separately.

---

## 🔮 Future Work

Potential improvements include:

- Expanding the dataset with more diverse Arabic football comments.
- Reviewing ambiguous examples and label consistency.
- Comparing oversampling with class-weighted loss.
- Investigating class-specific thresholds using validation data.
- Improving Arabic word-level SHAP visualization.
- Evaluating confidence calibration.
- Adding automated tests and configuration files.
- Building and deploying a Streamlit application.
- Documenting a fully reproducible training and inference workflow.

---

## ✅ Conclusion

This project explores Arabic sentiment classification in Jordanian football discussions using pretrained transformer models.

MARBERT achieved the best validation macro F1-score among the selected configurations and reached approximately `77%` test accuracy. The results also reveal a substantial difference in performance across sentiment classes, with Neutral remaining the main challenge.

By combining transformer fine-tuning, TF-IDF exploration, SHAP-based analysis, and class-level error investigation, the project emphasizes not only predictive performance but also model evaluation and interpretability.

The next step is to package the workflow into reproducible scripts and an interactive inference application.

---

## 👤 Author

**YOUR NAME**

- GitHub: [YOUR_GITHUB_PROFILE](https://github.com/YOUR_USERNAME)
- LinkedIn: [YOUR_LINKEDIN_PROFILE](https://www.linkedin.com/in/YOUR_LINKEDIN_USERNAME/)

---

*Feedback and suggestions for improving Arabic sentiment classification are welcome.*
