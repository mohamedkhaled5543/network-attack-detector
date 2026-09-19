# 🛡️ Network Attack Detector

Machine learning model that flags malicious network traffic.
Trained on the UNSW-NB15 dataset. Served through a Streamlit app.

**Live demo:** [<add your Streamlit link here>](https://network-attack-detector-gpvygc2guzznwdhsjbuxyh.streamlit.app/)

## What it does

Upload a CSV of network flow records. The app returns:

- An attack probability for every flow
- A Normal / Attack prediction
- The highest-risk flows
- A downloadable results file
- An adjustable threshold to trade false alarms against missed attacks

## Pipeline

1. Clean: strip column names, drop missing values, drop duplicates
2. Drop `id` and `attack_cat` (`attack_cat` leaks the label)
3. `log1p` on skewed numeric features (chosen on train only)
4. Stratified validation split from the training file
5. One-hot encode `proto`, `service`, `state`. Scale numeric features. Fit on train only.
6. Train Random Forest and XGBoost. Tune XGBoost with GridSearchCV.
7. Score once on the official, untouched test file

## Results

| Model | Test F1 |
|---|---|
| Random Forest | 0.834 |
| XGBoost | **0.837** |
| XGBoost (tuned) | 0.832 |

- Validation accuracy: 0.93
- Test accuracy: 0.85
- Test attack recall: 0.96
- Test normal recall: 0.78

## Key finding

Tuning did not help. All three models score within 0.005 F1 of each other.
The drop from validation (0.93) to test (0.85) likely comes from distribution shift
between the official train and test files. The model is not the bottleneck. The data is.

The model catches 96% of attacks. It also flags about 22% of normal traffic.
The threshold slider in the app lets you trade one for the other.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Try it with `sample_flows.csv`.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `network_bucket.ipynb` | Full training notebook |
| `model.pkl` | Trained XGBoost model |
| `preprocessor.pkl` | Fitted encoder and scaler |
| `skewed_cols.pkl` | Columns that get `log1p` |
| `sample_flows.csv` | 200 test rows for the demo |

## Data

UNSW-NB15, Australian Centre for Cyber Security.
Raw CSVs are not included. Download them from the official source.

## Stack

Python, pandas, scikit-learn, XGBoost, Streamlit
