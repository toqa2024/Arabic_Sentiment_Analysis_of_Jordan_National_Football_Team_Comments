<img width="73" height="59" alt="image" src="https://github.com/user-attachments/assets/14a607dd-c572-4a69-af77-35e89c9e3c0c" />

Arabic Sentiment Anlaysis Of Jordan National Fotball Team 

 Arabic NLP, Transformer-Based Emotion Classification & Explainable AI

An end-to-end Natural Language Processing (NLP) project focused on analyzing emotions expressed in Arabic football-related comments using pretrained Arabic Transformer models.

The project combines Arabic text preprocessing, exploratory data analysis, class balancing, Transformer fine-tuning, hyperparameter optimization, multi-class emotion classification, and SHAP-based explainability.

Three Arabic language models — **AraBERT, MARBERT, and ArabicBERT** — are evaluated to identify emotional expressions in football fan discussions.

The project investigates both classification performance and the linguistic patterns associated with different emotional categories, rather than relying on accuracy alone.

Best Performing Model: MARBERT — 78% Test Accuracy | 0.71 Macro F1-score

Disclaimer: This is an educational and research project. The models and results are intended for experimental analysis and are not presented as production-ready emotion classification systems.



Table of Contents

1. [Project Overview](#project-overview)
2. [Project Pipeline](#project-pipeline)
3. [Dataset](#dataset)
4. [Dataset Preparation and Preprocessing](#dataset-preparation-and-preprocessing)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Data Splitting and Class Balancing](#data-splitting-and-class-balancing)
7. [Transformer-Based Emotion Classification](#transformer-based-emotion-classification)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Classification Results](#classification-results)
10. [Model Evaluation and Visualization](#model-evaluation-and-visualization)
11. [TF-IDF Keyword Analysis](#tf-idf-keyword-analysis)
12. [SHAP Explainability](#shap-explainability)
13. [Interactive Emotion Prediction](#interactive-emotion-prediction)
14. [Key Findings](#key-findings)
15. [Technologies Used](#technologies-used)
16. [Repository Structure](#repository-structure)
17. [How to Run the Project](#how-to-run-the-project)
18. [Limitations and Future Improvements](#limitations-and-future-improvements)
19. [Authors](#authors)



 Project Overview

Football discussions on social media frequently contain emotionally expressive Arabic comments, including reactions to national teams, player performance, match results, and refereeing decisions.

These comments often contain informal Arabic expressions, regional vocabulary, spelling variations, and emotionally charged language.

Automatically identifying emotions in such comments represents an interesting challenge for Arabic Natural Language Processing.

This project investigates whether pretrained Arabic Transformer models can classify football-related comments into three emotion categories:

- Pride: Expressions of pride, celebration, and admiration.
- Anger: Expressions of frustration, disappointment, and dissatisfaction.
- Neutral: Comments without a clearly expressed Pride or Anger emotion.

 Project Objectives

The project implements a multi-stage NLP workflow:

Arabic Text Preprocessing

Clean and normalize Arabic football comments to reduce noise and improve consistency.

Exploratory Data Analysis

Investigate emotion distributions, frequent words, and linguistic patterns through statistical analysis and Arabic WordClouds.

Class Balancing

Address imbalanced emotion categories using training-only random oversampling.

Transformer-Based Classification

Fine-tune and evaluate AraBERT, MARBERT, and ArabicBERT on the same three-class emotion classification task.

Hyperparameter Tuning

Evaluate different learning rates, batch sizes, epochs, and maximum sequence lengths.

Performance Evaluation

Compare models using Accuracy, Precision, Recall, Macro F1-score, Weighted F1-score, and Confusion Matrices.

Explainable AI

Use TF-IDF keyword analysis and SHAP explanations to investigate vocabulary patterns and individual model predictions.

The central objective is to identify the most effective Transformer model for classifying emotions in the available Arabic football-comment dataset.



 Project Pipeline

The complete workflow implemented in the notebook is summarized below.


                  Arabic Football Comments
                            |
                            v
                    Dataset Loading
                            |
                            v
                 Arabic Text Preprocessing
                            |
                            v
                Remove Empty / Duplicate Text
                            |
                            v
                 Exploratory Data Analysis
                      /             \
                     v               v
            Class Distribution   Arabic WordClouds
                     \               /
                      v             v
                   Cleaned Dataset
                            |
                            v
                 Stratified Data Splitting
                     70% / 15% / 15%
                            |
                            v
                 Training Set Oversampling
                            |
                            v
                Transformer Fine-Tuning
                 /          |          \
                v           v           v
             AraBERT      MARBERT    ArabicBERT
                |           |           |
                v           v           v
                  Hyperparameter Tuning
                    (4 Runs per Model)
                            |
                            v
                  Validation Evaluation
                            |
                            v
                 Best Configuration Selection
                            |
                            v
                     Test Evaluation
                            |
                            v
                  Model Performance Analysis
                            |
                            v
             Confusion Matrices & Loss Curves
                            |
                            v
               Word Frequency / TF-IDF Analysis
                            |
                            v
                  SHAP Explainability
                            |
                            v
               Interactive Emotion Prediction

Suggested Visualization: A pipeline diagram illustrating all major stages from dataset preparation to interactive prediction.



 Dataset

The project uses an Arabic football-related comments dataset stored in:

`Nlp_dataset.ods`

The dataset contains comments associated with football discussions, primarily involving expressions related to Jordanian football.

 Original Dataset Structure

The original dataset contains 3,000 Arabic comments and three columns.

| Column | Description |
|---|---|
| `Coments` | Original Arabic football-related comments |
| `Emotion` | Emotion label assigned to each comment |
| `Sentiment` | Additional sentiment information included in the dataset |

The Transformer models use the **Emotion** column as the classification target.

The Sentiment column is available in the original dataset but is not used as the target for the three-class emotion classification experiments.

 Emotion Categories

| Emotion | Description |
|---|---|
| Pride | Pride, celebration, admiration, and positive emotional attachment |
| Anger | Frustration, criticism, dissatisfaction, or anger |
| Neutral | Comments without a clearly expressed Pride or Anger emotion |

 Dataset Size

| Dataset Stage | Number of Comments |
|---|---:|
| Original Dataset | 3,000 |
| Removed Duplicates | 89 |
| Final Cleaned Dataset | **2,911** |

Final Emotion Distribution

| Emotion | Number of Comments | Percentage |
|---|---:|---:|
| Pride | 1,741 | 59.81% |
| Anger | 652 | 22.40% |
| Neutral | 518 | 17.79% |
| Total | 2,911 | 100% |

 Dataset Analysis

The class distribution indicates that Pride represents the majority of comments, accounting for approximately 60% of the cleaned dataset.

Anger and Neutral contain substantially fewer examples.

This imbalance is important because a classification model may achieve relatively high overall accuracy while performing less effectively on minority classes.

Therefore, the project uses random oversampling during training and Macro F1-score for model selection.

Dataset Source: Original project dataset (`Nlp_dataset.ods`). The collection source, annotation procedure, and redistribution permission should be documented before publishing the dataset publicly.



 Dataset Preparation and Preprocessing

 Objective

Arabic text preprocessing is applied to reduce noise, normalize linguistic variations, and prepare the football comments for exploratory analysis and Transformer-based classification.

The implementation performs several cleaning and normalization operations.

 Preprocessing Steps

1. Emoji Removal

Remove emojis and pictographic symbols from comments.

2. URL and Mention Removal

Remove URLs, social media mentions, and hashtag symbols.

3. Non-Arabic Character Removal

Remove unwanted non-Arabic characters and digits.

4. Arabic Diacritics Removal

Remove Arabic diacritical marks.

5. Arabic Character Normalization

Normalize selected Arabic characters to reduce spelling variations.

Examples include:

- أ / إ / آ → ا
- ى → ي
- ؤ / ئ → ء
- ة → ه

6. Repeated Character Reduction

Reduce excessive character repetitions that may appear in informal online comments.

7. Whitespace Normalization

Remove unnecessary spaces and normalize text formatting.

8. Arabic Stopword Removal

Remove common Arabic stopwords using standard and additional project-specific stopword lists.

9. Short Token Filtering

Remove tokens shorter than three characters.

10. Empty Comment Removal

Remove comments that become empty after preprocessing.

11. Duplicate Removal

Remove duplicate comments based on their cleaned text representation.

 Preprocessing Example

Original Comment:

> أداء يفرّح القلب وفخر لكل أردني

Cleaned Comment:

> اداء يفرح القلب وفخر لكل اردني

Preprocessing Results

| Metric | Result |
|---|---:|
| Original Records | 3,000 |
| Removed Duplicate Records | 89 |
| Remaining Records | **2,911** |

The cleaned dataset is exported as:

`Jordan_Cleaned_Comments.csv`

This file is used in the model training and evaluation stages.

 Preprocessing Considerations

Text normalization improves consistency but can also remove meaningful linguistic information.

In particular, stopword removal and aggressive normalization may affect the representation of short or context-dependent emotional expressions.

The project preserves its original preprocessing configuration throughout the experiments.


 Exploratory Data Analysis

Exploratory Data Analysis (EDA) is performed to investigate the structure and linguistic properties of the cleaned Arabic football comments.

The analysis focuses on class distribution, frequently occurring words, and vocabulary differences between emotion categories.

 1. Emotion Distribution Analysis

A class-distribution visualization is generated to examine the number of comments associated with each emotion.

<img width="1069" height="595" alt="image" src="https://github.com/user-attachments/assets/461d52d9-741d-476a-8ec3-eaabe01ff355" />

The results show that:

- Pride is the most frequent emotion.
- Anger represents the second-largest category.
- Neutral contains the fewest examples.

This analysis motivates the use of training-only oversampling and class-sensitive evaluation metrics.

Suggested Figure: Emotion Distribution Bar Chart

 2. Arabic WordCloud Visualization

The notebook generates WordCloud visualizations separately for each emotion class.

The implementation uses:

- WordCloud
- Arabic Reshaper
- Python BiDi
- Pillow
- Matplotlib

Arabic reshaping and bidirectional text processing are used to improve the visualization of Arabic words.

A trophy-shaped WordCloud mask is used to reflect the football-related context of the project.

 WordCloud Categories

| Emotion | Visualization Color |
|---|---|
| Pride | Green |
| Anger | Orange |
| Neutral | Purple |

<img width="1883" height="1082" alt="download" src="https://github.com/user-attachments/assets/d1211cff-00c6-4e4c-b4ca-080eedadcd71" />

 3. Frequent Word Analysis

The notebook examines frequently occurring words in the cleaned text.

The analysis helps identify vocabulary patterns commonly associated with football discussions and each emotion category.

 4. Word–Emotion Heatmap

Word–emotion heatmaps are generated to visualize the association between frequently occurring words and their emotion categories.

The analysis focuses on the 20 most frequent words.

The heatmaps complement the WordClouds by displaying word-frequency relationships in a more structured format.

Suggested Figures:

- AraBERT Word–Emotion Heatmap
  
- <img width="1193" height="1066" alt="image" src="https://github.com/user-attachments/assets/c5adfe88-d22a-4012-a5ec-e309a5a0a4b0" />

- MARBERT Word–Emotion Heatmap
  
- <img width="1188" height="1059" alt="image" src="https://github.com/user-attachments/assets/f1794c55-9652-4737-a381-6d475fcd5781" />

- ArabicBERT Word–Emotion Heatmap

- <img width="1181" height="1059" alt="image" src="https://github.com/user-attachments/assets/e3e397be-d93c-43ef-bd64-f11bd645c7e1" />

These visualizations are descriptive frequency analyses rather than direct explanations of Transformer model decisions.

 Data Splitting and Class Balancing

 Dataset Splitting

The cleaned dataset is divided into training, validation, and test subsets using stratified sampling.

A fixed random seed is used for reproducibility.

Random Seed: `42`

 Data Split Configuration

| Subset | Percentage | Number of Comments |
|---|---:|---:|
| Training | 70% | 2,037 |
| Validation | 15% | 437 |
| Test | 15% | 437 |
| Total | 100% | 2,911 |

Stratification is used to preserve similar class proportions across the three subsets.

 Training Set Oversampling

Because Pride is the majority class, random oversampling is applied to the training subset.

Minority-class examples are sampled with replacement until each emotion category contains the same number of training examples.

 Training Distribution After Oversampling

| Emotion | Training Samples |
|---|---:|
| Pride | 1,218 |
| Anger | 1,218 |
| Neutral | 1,218 |
| Total | 3,654 |

 Validation Distribution

| Emotion | Samples |
|---|---:|
| Pride | 262 |
| Anger | 98 |
| Neutral | 77 |
| Total | 437 |

 Test Distribution

| Emotion | Samples |
|---|---:|
| Pride | 261 |
| Anger | 98 |
| Neutral | 78 |
| Total | 437 |

 Important Methodological Note

Oversampling is applied only to the training data.

The validation and test subsets retain their original distributions, preventing duplicated oversampled training examples from being introduced into these evaluation subsets.

This allows classification performance to be assessed against the original class proportions.



 Transformer-Based Emotion Classification

 Objective

The classification stage compares three pretrained Arabic Transformer models on the same three-class emotion classification problem.

The evaluated models are:

1. AraBERT
2. MARBERT
3. ArabicBERT

Each model is fine-tuned to predict one of the following labels:

- Anger
- Neutral
- Pride

 1. AraBERT

Pretrained Checkpoint: `aubmindlab/bert-base-arabertv02`

AraBERT is a BERT-based language model pretrained for Arabic language understanding tasks.

In this project, AraBERT is fine-tuned to classify Arabic football comments into three emotion categories.

The model serves as one of the main experimental baselines.

 2. MARBERT

Pretrained Checkpoint: `UBC-NLP/MARBERT`

MARBERT is an Arabic Transformer model developed with a focus on Arabic social media language.

Its pretraining characteristics make it particularly relevant to the informal and emotionally expressive language found in online football discussions.

In the recorded experiments, MARBERT achieved the strongest overall classification performance.

 3. ArabicBERT

Pretrained Checkpoint: `asafaya/bert-base-arabic`

ArabicBERT is another pretrained Arabic BERT model evaluated for the same emotion classification task.

It provides an additional comparison point for investigating differences in model performance.

 Classification Workflow

The training workflow includes:

1. Loading the cleaned dataset.
2. Encoding emotion labels.
3. Creating stratified data splits.
4. Applying random oversampling to training examples.
5. Loading the pretrained tokenizer.
6. Tokenizing and encoding Arabic comments.
7. Initializing the model for sequence classification.
8. Fine-tuning the model using PyTorch and Hugging Face Transformers.
9. Evaluating multiple hyperparameter configurations.
10. Selecting the best configuration using validation Macro F1-score.
11. Evaluating the selected model on the held-out test set.

 Model Comparison

| Model | Pretrained Checkpoint | Task |
|---|---|---|
| AraBERT | `aubmindlab/bert-base-arabertv02` | Emotion Classification |
| MARBERT | `UBC-NLP/MARBERT` | Emotion Classification |
| ArabicBERT | `asafaya/bert-base-arabic` | Emotion Classification |

All three models use the same cleaned dataset and evaluation split structure.



 Hyperparameter Tuning

 Objective

Hyperparameter tuning is performed to evaluate how different training configurations affect model performance.

Four experimental configurations are tested for each Transformer.

Total Experiments: 12

 Training Configurations

| Run | Learning Rate | Batch Size | Epochs | Max Length | Weight Decay |
|---|---:|---:|---:|---:|---:|
| Run 1 | 1e-5 | 16 | 3 | 96 | 0.02 |
| Run 2 | 2e-5 | 16 | 3 | 96 | 0.02 |
| Run 3 | 1e-5 | 16 | 4 | 96 | 0.02 |
| Run 4 | 1e-5 | 8 | 3 | 128 | 0.02 |

 Model Selection Criterion

The best configuration is selected based on the highest Validation Macro F1-score.

Macro F1 is calculated by averaging the F1-score across the three emotion classes, giving each class equal weight.

This metric is particularly useful for imbalanced datasets because it does not allow the majority class to dominate the overall evaluation to the same extent as accuracy.

 AraBERT Hyperparameter Results

| Run | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| Run 1 | 73.68% | 0.6696 |
| Run 2 | 73.68% | 0.6724 |
| Run 3 | 72.54% | 0.6553 |
| Run 4 | 72.08% | 0.6543 |

Best AraBERT Configuration: Run 2

- Learning Rate: `2e-5`
- Batch Size: `16`
- Epochs: `3`
- Maximum Sequence Length: `96`
- Weight Decay: `0.02`
- Best Validation Macro F1: 0.6724

 MARBERT Hyperparameter Results

| Run | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| Run 1 | 76.89% | 0.7011 |
| Run 2 | 75.29% | 0.6792 |
| Run 3 | 75.97% | 0.6951 |
| Run 4 | 75.29% | 0.6790 |

Best MARBERT Configuration: Run 1

- Learning Rate: `1e-5`
- Batch Size: `16`
- Epochs: `3`
- Maximum Sequence Length: `96`
- Weight Decay: `0.02`
- Best Validation Macro F1: **0.7011**

### ArabicBERT Hyperparameter Results

| Run | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| Run 1 | 67.73% | 0.6190 |
| Run 2 | 70.94% | 0.6380 |
| Run 3 | 67.96% | 0.6142 |
| Run 4 | 67.05% | 0.6026 |

Best ArabicBERT Configuration: Run 2

- Learning Rate: `2e-5`
- Batch Size: `16`
- Epochs: `3`
- Maximum Sequence Length: `96`
- Weight Decay: `0.02`
- Best Validation Macro F1: 0.6380

 Best Configuration Summary

| Model | Best Run | Validation Accuracy | Validation Macro F1 |
|---|---|---:|---:|
| AraBERT | Run 2 | 73.68% | 0.6724 |
| MARBERT | Run 1 | 76.89% | 0.7011 |
| ArabicBERT | Run 2 | 70.94% | 0.6380 |

MARBERT achieved the highest validation Macro F1 among the evaluated models.



 Classification Results

The selected configurations are evaluated using the held-out test dataset.

Total Test Samples: 437

 Overall Test Performance

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| AraBERT | 76% | 0.68 | 0.76 |
| MARBERT | 78 | 0.71 | 0.79 |
| ArabicBERT | 71% | 0.62 | 0.71 |

 Model Comparison

The recorded results show that MARBERT achieved the highest overall performance.

MARBERT Results:

- Test Accuracy: 78%
- Macro Precision: 0.71
- Macro Recall: 0.72
- Macro F1-score: 0.71
- Weighted F1-score: 0.79

AraBERT achieved the second-highest overall performance, while ArabicBERT achieved the lowest results among the three evaluated models.

 Class-Wise F1-score Comparison

| Emotion | AraBERT | MARBERT | ArabicBERT |
|---|---:|---:|---:|
| Anger | 0.68 | 0.68 | 0.62 |
| Neutral | 0.51 | 0.57 | 0.41 |
| Pride | 0.86 | 0.89 | 0.84 |

### AraBERT Classification Results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Anger | 0.66 | 0.70 | 0.68 | 98 |
| Neutral | 0.51 | 0.51 | 0.51 | 78 |
| Pride | 0.87 | 0.85 | 0.86 | 261 |

Test Accuracy: 76%

 MARBERT Classification Results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Anger | 0.66 | 0.70 | 0.68 | 98 |
| Neutral | 0.56 | 0.58 | 0.57 | 78 |
| Pride | 0.90 | 0.87 | 0.89 | 261 |

Test Accuracy: 78%

 ArabicBERT Classification Results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Anger | 0.59 | 0.65 | 0.62 | 98 |
| Neutral | 0.41 | 0.40 | 0.41 | 78 |
| Pride | 0.85 | 0.83 | 0.84 | 261 |

Test Accuracy: 71%

 Results Interpretation

Pride Classification

Pride achieved the highest F1-score across all three models.

MARBERT recorded the best Pride F1-score of 0.89.

Anger Classification

AraBERT and MARBERT achieved the same Anger F1-score of 0.68.

ArabicBERT achieved a lower Anger F1-score of 0.62.

Neutral Classification

Neutral remained the most challenging category across all three models.

MARBERT achieved the highest Neutral F1-score of 0.57, compared with 0.51 for AraBERT and 0.41 for ArabicBERT.

 Main Classification Finding

The results indicate that MARBERT provides the strongest overall performance in the recorded experiments particularly for Pride and Neutral classification.

However, class-level differences demonstrate that overall accuracy alone is insufficient to fully describe model performance.



 Model Evaluation and Visualization

The project uses multiple evaluation techniques to analyze classification performance and training behavior.

 1. Classification Reports

Classification reports are generated for each selected model.

The reports include:

- Precision
- Recall
- F1-score
- Support
- Accuracy
- Macro Average
- Weighted Average

These metrics provide class-level and overall performance information.

 2. Confusion Matrix Analysis

Confusion matrices are generated to illustrate how predicted emotion labels compare with actual labels.

They provide a visual representation of correct and incorrect predictions across the three emotion categories.

 AraBERT Confusion Matrix

Suggested Figure: AraBERT Confusion Matrix

 MARBERT Confusion Matrix

Suggested Figure: MARBERT Confusion Matrix

 ArabicBERT Confusion Matrix

Suggested Figure: ArabicBERT Confusion Matrix

The confusion matrices help identify classification errors and relationships between emotion categories.

 3. Training and Validation Loss

Training and validation loss visualizations are generated for the selected models.

These plots are used to inspect the recorded optimization behavior during fine-tuning.

Suggested Figure: Training vs. Validation Loss

 4. Word–Emotion Association Heatmaps

The notebook produces word-frequency association heatmaps to illustrate how commonly occurring Arabic terms are distributed across emotion categories.

These visualizations provide descriptive linguistic analysis that complements the classification metrics.



 TF-IDF Keyword Analysis

 Objective

TF-IDF-based keyword analysis is used to identify informative terms associated with the three emotion categories.

This analysis provides an additional perspective on the vocabulary patterns appearing in the Arabic football comments.

 Extracted Keywords

The recorded notebook output identifies several representative terms.

| Emotion | Example Keywords |
|---|---|
| Pride | النشامي، مبروك، منتخبنا، ابطال، التاهل |
| Anger | الحكم، ضعيف، اخطاء، ظلم، للاسف |
| Neutral | الارجنتين، الاداء، الحكم، محلل، مستوي |

 Keyword Interpretation

Pride

Pride-related vocabulary frequently includes expressions of celebration, team identification, and achievement.

Anger

Anger-related vocabulary includes criticism, perceived unfairness, and negative evaluations of performance.

Neutral

Neutral comments contain more descriptive or discussion-oriented vocabulary without necessarily expressing a strongly identifiable emotional reaction.

 Important Note

TF-IDF identifies statistically informative words in the text corpus.

It does not establish that these words are necessarily responsible for the predictions made by the Transformer models.

The project uses SHAP separately to examine individual prediction explanations.



 SHAP Explainability

 Objective

Explainable Artificial Intelligence is incorporated to investigate how individual tokens contribute to model predictions.

The project uses SHAP (SHapley Additive exPlanations) to produce token-level explanations for selected Arabic football comments.

SHAP helps illustrate which tokens contribute positively or negatively to the score associated with a selected emotion class.

 Explainability Workflow

The implemented explainability stage includes:

1. Preparing the Arabic input comment.
2. Applying text cleaning and normalization.
3. Passing the processed text to the trained classification model.
4. Obtaining predicted emotion probabilities.
5. Identifying the predicted emotion class.
6. Generating SHAP explanations.
7. Extracting influential tokens.
8. Visualizing token contributions.

 Model Used for Explainability

The SHAP implementation reuses the `best_trainer` and `tokenizer` variables available after the model training sections.

When the notebook is executed in its original order, these variables correspond to **ArabicBERT**, which is the final trained model in the notebook.

Therefore, the saved SHAP examples should be interpreted as explanations of the ArabicBERT model rather than the best-performing MARBERT model.

 Example 1: Pride Prediction

Input Comment:

> مبروك المنتخب الاردني

Predicted Emotion: Pride

Confidence: 99.79%

Detected Keyword: مبروك

 SHAP Token Contributions

| Token | SHAP Value |
|---|---:|
| مبروك | +0.475249 |
| الاردني | +0.293240 |
| المنتخب | +0.212966 |

The recorded SHAP explanation assigns positive contributions to all three displayed tokens for the Pride class.

The word مبروك has the largest positive contribution in this example.

Suggested Figure: Pride SHAP Explanation

 Example 2: Anger Prediction

Input Comment:

> فرق عربيه ضعيفه
Predicted Emotion: Anger

Confidence: 99.70%

Detected Keyword: ضعيف

The recorded SHAP explanation highlights the contribution of the word ضعيف to the Anger prediction.

The analysis also demonstrates how tokenization can produce subword fragments, which must be considered when interpreting individual token contributions.

Suggested Figure: Anger SHAP Explanation

 Example 3: Neutral Prediction

Input Comment:

> فوز اردني بنكهة مغربية
Predicted Emotion: Neutral

Confidence: 99.07%

Detected Keywords: No strong keyword found

The explanation highlights several tokens and subword fragments associated with the predicted Neutral class.

This example illustrates that a model can produce a high-confidence prediction even when the independent keyword analysis does not identify a strong class-associated term.

Suggested Figure: Neutral SHAP Explanation

 Explainability Findings

SHAP visualizations provide local information about how the model responds to selected inputs.

The examples illustrate:

- Positive and negative token contributions.
- Differences between keyword-based analysis and model-based explanations.
- The effect of subword tokenization on interpretability.
- The relationship between predicted classes and influential input tokens.

 Important Limitations

SHAP values describe estimated contributions to model output scores.

They do not prove that the model understands emotion in a human-like way, and high prediction confidence does not guarantee correctness.

The explanations should be interpreted as local model-behavior analyses rather than causal proof of linguistic reasoning.



 Interactive Emotion Prediction

 Objective

The final notebook section provides an interactive prediction interface for testing Arabic comments.

Users can enter a football-related comment and receive an emotion prediction.

The interactive workflow includes:

1. Reading a user-provided Arabic comment.
2. Cleaning and normalizing the text.
3. Generating a model prediction.
4. Displaying the predicted emotion.
5. Reporting the prediction confidence.
6. Displaying detected emotion-related keywords.
7. Producing a SHAP explanation for the selected prediction.

The interface continues accepting comments until the user enters the exit command.

 Example Predictions

| Input Comment | Predicted Emotion | Confidence |
|---|---|---:|
| مبروك المنتخب الاردني | Pride | 99.79% |
| فرق عربيه ضعيفه | Anger | 99.70% |
| فوز اردني بنكهة مغربية | Neutral | 99.07% |

These are recorded examples from the notebook rather than a separate benchmark evaluation.

The displayed confidence values are model-generated probabilities and should not be interpreted as guarantees of prediction correctness.

 Exit Command

The interactive prediction loop terminates when the user enters:

`خروج`
 Interactive Prediction and SHAP Visualization

The pride comment shape Explanation

<img width="1233" height="737" alt="image" src="https://github.com/user-attachments/assets/b7cd3529-dcfa-4985-93f7-d106e180dd2f" />

The Anger Comment Shape Explanation 

<img width="1222" height="723" alt="image" src="https://github.com/user-attachments/assets/adc7fa07-7692-4896-9cf3-c8a29daaa628" />

The Neutral Comment Shape Explanation

<img width="1222" height="717" alt="image" src="https://github.com/user-attachments/assets/1e4075dd-d7da-4618-97f8-16e4e5ecd0e4" />

 Key Findings:

The experiments provide several observations regarding emotion classification in Arabic football-related comments.

 1. MARBERT Achieved the Highest Overall Performance

MARBERT achieved the highest test accuracy of 78% and Macro F1-score of 0.71.

It outperformed AraBERT and ArabicBERT under the recorded experimental configurations.

 2. Pride Was the Easiest Emotion to Classify

All three models achieved their strongest class-level F1-scores on Pride.

MARBERT achieved the best Pride F1-score of 0.89 .

 3. Neutral Was the Most Challenging Emotion

Neutral consistently received lower F1-scores compared with Pride and Anger.

This suggests that distinguishing neutral expressions from emotionally expressive comments remains difficult in the evaluated dataset.

 4. Class Imbalance Influenced the Evaluation Strategy

The original cleaned dataset contained substantially more Pride examples than Anger or Neutral examples.

Random oversampling was used to balance the training data while preserving the original validation and test distributions.

 5. Macro F1 Was Important for Model Selection

Macro F1 was used to compare configurations because it gives equal evaluation weight to all emotion categories.

This makes it particularly informative when analyzing imbalanced classification datasets.

 6. Explainability Complemented Classification Metrics

TF-IDF analysis and SHAP explanations provided additional information about vocabulary patterns and selected model predictions.

These methods complement predictive performance evaluation but do not replace quantitative testing.

 Overall Finding

MARBERT demonstrated the strongest performance among the three evaluated Arabic Transformer models for the available football emotion classification dataset.

The experiments also highlight the importance of class balancing, detailed evaluation, and careful interpretation of explainability results.



 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Google Colab | Development and experimentation |
| PyTorch | Deep Learning training and inference |
| Hugging Face Transformers | Transformer fine-tuning and tokenization |
| AraBERT | Arabic emotion classification |
| MARBERT | Arabic emotion classification |
| ArabicBERT | Arabic emotion classification |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| scikit-learn | Data splitting, TF-IDF, and evaluation metrics |
| NLTK | Arabic stopword processing |
| PyArabic | Arabic text preprocessing |
| emoji | Emoji handling |
| WordCloud | Word-frequency visualizations |
| arabic-reshaper | Arabic text reshaping |
| python-bidi | Bidirectional Arabic text visualization |
| Matplotlib | Results visualization |
| Seaborn | Statistical visualization and heatmaps |
| SHAP | Model explainability |
| Pillow | WordCloud image-mask processing |
| odfpy | Reading the ODS dataset |



 Repository Structure

The following structure is recommended for publishing the project.

Arabic-Football-Emotion-Analysis/
│
├── README.md
│
├── NLP_P.ipynb
│
├── Nlp_dataset.ods
│
└── images/
    │
    ├── project_pipeline.png
    ├── class_distribution.png
    ├── arabic_wordcloud_by_emotion.png
    ├── arabert_confusion_matrix.png
    ├── marbert_confusion_matrix.png
    ├── arabicbert_confusion_matrix.png
    ├── model_comparison.png
    ├── training_validation_loss.png
    ├── word_emotion_heatmap.png
    ├── shap_pride.png
    ├── shap_anger.png
    └── shap_neutral.png


The `images/` directory is a recommended folder for storing visualizations exported from the notebook.

The image filenames are suggested publication names and must match the actual uploaded files before adding image links to the README.

The dataset should only be published if redistribution is permitted.

The original notebook can be uploaded without modifying its code or recorded results.



 How to Run the Project

 1. Open the Notebook

Open `NLP_P.ipynb` using Google Colab or a compatible Jupyter Notebook environment.

For Google Colab:

1. Upload or open the notebook.
2. Select a GPU-enabled runtime when available.
3. Run the cells in their original order.

A GPU is recommended because the project involves fine-tuning three pretrained Transformer models across multiple configurations.

 2. Prepare the Dataset

Upload the original dataset:

`Nlp_dataset.ods`

The notebook reads the dataset from the following location:

```python
df = pd.read_excel(
    '/content/Nlp_dataset.ods',
    engine='odf'
)
```

The original dataset should contain the following columns:

- `Coments`
- `Emotion`
- `Sentiment`

 3. Install Dependencies

The notebook contains dependency installation commands in the corresponding code cells.

Execute these installation commands before running the related preprocessing, training, or explainability stages.

The project uses libraries including Transformers, PyTorch, scikit-learn, Pandas, SHAP, and Arabic text-processing utilities.

 4. Execute Dataset Preprocessing

Run the preprocessing section to:

- Load the original dataset.
- Clean and normalize Arabic comments.
- Remove empty and duplicate records.
- Generate dataset distribution plots.
- Generate Arabic WordClouds.
- Save the cleaned CSV dataset.

The generated cleaned dataset is:

`Jordan_Cleaned_Comments.csv`

 5. Run Transformer Training

Execute the training sections for:

1. AraBERT
2. MARBERT
3. ArabicBERT

Each model is evaluated using four hyperparameter configurations.

The best configuration is selected according to validation Macro F1-score.

 6. Evaluate the Models

The notebook evaluates the selected models on the test set.

The generated outputs include:

- Classification reports.
- Accuracy and F1-score metrics.
- Confusion matrices.
- Training and validation loss curves.
- Word-frequency association heatmaps.

 7. Run the Explainability Section

Execute the final SHAP analysis section after completing the preceding model training cells.

This section provides:

- TF-IDF keyword analysis.
- Interactive Arabic emotion prediction.
- Prediction confidence values.
- Token-level SHAP explanations.

 8. Run Interactive Predictions

Enter an Arabic football-related comment when prompted.

The notebook displays the predicted emotion and supporting analysis.

Type `خروج` to terminate the interactive prediction loop.

 Reproducibility Note

The results presented in this README are taken from the recorded notebook outputs.

Rerunning the experiments may produce different results depending on package versions, random behavior, computational hardware, and pretrained-model availability.

No model architecture, code implementation, or saved experimental result was changed for this documentation.



 Limitations and Future Improvements

Although this project combines Arabic text preprocessing, Transformer classification, and explainability, several limitations should be considered.

 Current Limitations

1. Dataset Imbalance

Pride accounts for approximately 60% of the cleaned dataset.

Although random oversampling was applied during training, class imbalance remains an important consideration when interpreting the results.

2. Dataset Scope

The experiments are based on a single Arabic football-related comments dataset.

The results do not establish generalization to other dialects, discussion topics, or social media platforms.

3. Neutral Classification
   
Neutral achieved the lowest F1-scores across the three evaluated models.

This highlights the difficulty of distinguishing emotionally neutral comments from comments containing Pride or Anger.

4. Arabic Language Complexity

Dialectal expressions, informal spelling, contextual meaning, and short comments can complicate automated emotion classification.

5. Preprocessing Trade-offs

Text normalization and stopword removal may remove information that contributes to the original emotional meaning.

6. Limited Hyperparameter Search

Four configurations were evaluated for each model.

The selected configurations represent the best results within the tested settings, not necessarily globally optimal configurations.

7. Explainability Limitations

SHAP explanations describe estimated token contributions to specific predictions.

They do not establish causal reasoning or guarantee that predictions are linguistically or contextually correct.

8. Interactive Prediction Scope

The interactive prediction interface demonstrates model behavior on selected comments.

Its examples do not constitute an independent external evaluation.

9. Dataset Documentation

The original collection methodology, annotation procedure, and licensing conditions should be documented before public distribution.

 Future Improvements

Potential future research directions include:

- Evaluate additional Arabic Transformer architectures.
- Expand the dataset with more diverse football-related comments.
- Investigate cross-dialect emotion classification.
- Explore alternative class-balancing strategies.
- Perform detailed error analysis on Neutral predictions.
- Compare different Arabic preprocessing configurations.
- Evaluate the models on independent external datasets.
- Analyze SHAP explanations across multiple Transformer models.
- Investigate probability calibration and prediction uncertainty.
- Add systematic experiment tracking and reproducible dependency versions.
- Develop a web-based demonstration for Arabic football emotion classification.

These improvements are proposed directions for future work and are not implemented modifications to the original project.

Contributors:

-  contribution with { Rama Bawaneh, Besan Abu Zraiq}  GitHub profile link:
-  https://github.com/ramabawaaneh9
-  https://github.com/search?q=Besan-abu-zraiq&type=users

 Disclaimer

This repository is intended for educational and research purposes only.

The reported classification results reflect the dataset, training configurations, and evaluation procedures used in the recorded experiments.

The models have not been independently validated for general-purpose emotion detection across Arabic dialects, platforms, or domains.

Prediction confidence values and SHAP explanations should be interpreted carefully and should not be considered guarantees of correctness.

The project is presented as an academic exploration of Arabic NLP, Transformer-based emotion classification, and Explainable Artificial Intelligence.
