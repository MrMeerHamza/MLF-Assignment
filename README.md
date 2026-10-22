# MLF Assignment 02 – Developing an End-to-End Machine Learning System

Two end-to-end machine learning systems built with Python and scikit-learn and deployed on **Streamlit Community Cloud**:

1. **Titanic Passenger Survival Prediction**: the instructor's code, run and improved.
2. **Heart Disease Prediction**: built by modifying the Titanic code and following the same steps, on the UCI Heart Disease dataset, using only 4 input attributes.

## Live Demo

| System | Live App |
| ------ | -------- |
| Titanic Passenger Survival Prediction | https://sp24-bse-062-titanic.streamlit.app/ |
| Heart Disease Prediction | https://sp24-bse-062-heart-disease.streamlit.app/ |

**GitHub Repository:** https://github.com/MrMeerHamza/MLF-Assignment/

## Student Information

| | |
| --- | --- |
| **Student Name** | Muddasar Ali Ghafoor |
| **Registration No.** | SP24-BSE-062 |
| **Program / Section** | BSSE (A) |
| **Course** | Machine Learning Fundamentals (AIC354) |
| **Submitted To** | Sir Qaiser Habib |
| **Institute** | COMSATS University Islamabad, Lahore Campus |

## Repository Structure

```
MLF-Assignment/
├── Titanic_Survival_Prediction.ipynb      # Task 1 notebook (all ML steps, with outputs)
├── Heart_Disease_Prediction.ipynb         # Task 2 notebook (all ML steps, with outputs)
├── Titanic_Survival_Prediction/
│   ├── app.py                             # Streamlit web app
│   ├── requirements.txt                   # packages for Streamlit Cloud
│   ├── svc_trained_model.pkl              # trained SVC model
│   ├── sample-data.csv                    # dataset (100 passengers)
│   └── *.csv                              # encoded / train / test / prediction files
└── Heart_Disease_Prediction/
    ├── app.py                             # Streamlit web app
    ├── requirements.txt                   # packages for Streamlit Cloud
    ├── svc_trained_model.pkl              # trained SVC model
    ├── processed.cleveland.data           # UCI Cleveland dataset (303 patients)
    ├── heart-disease.names                # UCI attribute descriptions
    ├── sample-data.csv                    # 4 selected attributes + output (297 patients)
    └── *.csv                              # encoded / train / test / prediction files
```

## Machine Learning Cycle

Both notebooks follow the same steps:

1. Import Libraries
2. Load Sample Data
3. Understand and Pre-process Data
4. Feature Extraction (feature selection for Heart Disease)
5. Label Encoding (inputs and output converted to numbers)
6. Training Phase (train/test split, Support Vector Classifier, save model)
7. Testing Phase (accuracy, precision, recall, F1, confusion matrix, cross-validation)
8. Application Phase (predict for a new, unseen input)
9. Feedback Phase
10. Deployment (Streamlit web app)

## Task 1 – Titanic Passenger Survival Prediction

**Inputs:** PClass (First, Second, Third), Gender (Male, Female), Sibling (Zero, One, Two, Three), Embarked (Southampton, Cherbourg, Queenstown)
**Output:** Survived (Yes / No)

### Improvements to the provided code

The same dataset and the same 4 inputs are used, with light fixes:

| # | Problem in the provided code | Fix |
| - | ---------------------------- | --- |
| 1 | `LabelEncoder` encoded **Sibling** alphabetically (One=0, Three=1, Two=2, Zero=3) | Ordinal encoding: Zero=0, One=1, Two=2, Three=3 |
| 2 | `shuffle=False`: the last 20 rows were always the test set | Shuffled, stratified train/test split |
| 3 | Fixed default RBF kernel | SVC kernel chosen by 5-fold cross-validation on training data (**linear** selected) |
| 4 | Accuracy on only 20 test rows is noisy | Repeated stratified 5-fold cross-validation (50 rounds) added |
| 5 | Files never closed, an array compared with `== 1` | `with open(...)`, prediction read with `[0]` |

### Results

| Measure | Original Code | Improved Code |
| ------- | :-----------: | :-----------: |
| Accuracy on 20 test rows | 0.80 | 0.80 |
| **Mean cross-validated accuracy** | **0.677** | **0.740** |

## Task 2 – Heart Disease Prediction

**Dataset:** [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), Cleveland database (`processed.cleveland.data`, 303 patients, 14 attributes).

### Pre-processing
- Removed 6 rows with missing values (`?` in `ca` and `thal`), leaving **297 patients** (160 No / 137 Yes).
- Converted the diagnosis `num` (0 – 4) into **HeartDisease** (0 → No, 1 – 4 → Yes).

### Feature selection (only 4 attributes allowed)
The 8 categorical attributes were ranked by **Cramér's V** (strength of association with heart disease), and the top 4 were kept:

| Rank | Attribute | Cramér's V | Values used |
| :--: | --------- | :--------: | ----------- |
| 1 | Thal (`thal`) | 0.527 | Normal, Fixed Defect, Reversible Defect |
| 2 | ChestPainType (`cp`) | 0.510 | Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic |
| 3 | MajorVessels (`ca`) | 0.493 | Zero, One, Two, Three |
| 4 | ExerciseAngina (`exang`) | 0.414 | Yes, No |

### Results

| Measure | Value |
| ------- | :---: |
| Model | Support Vector Classifier (linear kernel, chosen by cross-validation) |
| Accuracy on 60 test rows | 0.78 |
| Precision / Recall (Heart Disease class) | 0.76 / 0.79 |
| **Mean cross-validated accuracy** | **0.809** |

> This Heart Disease system is for learning purposes only and is **not** a medical diagnosis tool.

## Run Locally

```bash
git clone https://github.com/MrMeerHamza/MLF-Assignment.git
cd MLF-Assignment
pip install numpy pandas scikit-learn==1.9.1 scipy matplotlib seaborn prettytable notebook streamlit
```

Open the notebooks:

```bash
jupyter notebook
```

Run a web app:

```bash
cd Titanic_Survival_Prediction
streamlit run app.py
```

(Use `cd Heart_Disease_Prediction` for the Heart Disease app.)

## Deployment

Each app is deployed on [Streamlit Community Cloud](https://share.streamlit.io) from this repository:

| App | Main file path | Python |
| --- | -------------- | ------ |
| Titanic | `Titanic_Survival_Prediction/app.py` | 3.12 |
| Heart Disease | `Heart_Disease_Prediction/app.py` | 3.12 |

Each app loads its `svc_trained_model.pkl` and label-encodes the user's drop-down choices exactly as in Step 8 of its notebook. `scikit-learn` is pinned to **1.9.1** in `requirements.txt` to match the version the models were trained with.

## Tech Stack

Python · NumPy · Pandas · scikit-learn · SciPy · Matplotlib · Seaborn · PrettyTable · Jupyter Notebook · Streamlit

## Acknowledgements

- Original Titanic Passenger Survival Prediction code: **Mr. Mohsin Afzal** and **Dr. Rao Muhammad Adeel Nawab** ([ml-teaching-notebooks](https://github.com/haqnawaz99/ml-teaching-notebooks)).
- Heart Disease dataset: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
