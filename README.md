# TriageCoPilot

**Multimodal Triage Decision Support with Clinical Interpretability and Equity Auditing**

Submission for the [Triagegeist Competition](https://www.kaggle.com/competitions/triagegeist) — Laitinen-Fredriksson Foundation

---

## Overview

TriageCoPilot is a multimodal clinical decision support proof-of-concept that predicts Emergency Severity Index (ESI) triage acuity from structured patient intake data and free-text chief complaints. Beyond prediction, the system provides:

- **SHAP-based clinical explanations** for each prediction
- **Systematic equity auditing** across demographic subgroups
- **Safety analysis** focused on undertriage of critical patients

## Results

| Metric | Value |
|--------|-------|
| Ensemble QWK | 0.9976 |
| Accuracy | 99.5% |
| Severe undertriage (≥2 levels) | 0 cases |

## Approach

1. **Feature Engineering**: 75+ clinically-motivated features including missingness indicators, clinical threshold flags (NEWS2, shock index, SIRS), NLP keyword matching, and TF-IDF from chief complaints
2. **Modeling**: LightGBM + CatBoost ensemble with 5-fold stratified CV and QWK threshold optimization
3. **Interpretability**: SHAP analysis revealing pain score, NEWS2, GCS, and SpO2 as top predictors
4. **Equity Audit**: Consistent performance across sex, age, arrival mode subgroups (QWK 0.9972–0.9981)

## Repository Structure

```
├── README.md
├── requirements.txt
├── notebook/
│   └── triagegeist-dev.ipynb    # Complete Kaggle notebook
├── writeup/
│   └── writeup.md               # Competition writeup
└── outputs/
    ├── submission.csv            # Test set predictions
    └── figures/                  # Generated plots
```

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Running the Notebook

The notebook is designed to run on Kaggle with GPU enabled:

1. Go to [Kaggle](https://www.kaggle.com)
2. Create a new notebook
3. Add the Triagegeist competition data
4. Enable GPU accelerator
5. Upload and run `triagegeist-dev.ipynb`

**Runtime**: ~80 minutes on Kaggle T4 GPU

### Local Execution

```bash
# Download competition data from Kaggle
kaggle competitions download -c triagegeist

# Run notebook
jupyter notebook notebook/triagegeist-dev.ipynb
```

## Dependencies

- Python 3.10+
- pandas, numpy, scipy
- scikit-learn
- LightGBM
- CatBoost
- SHAP
- matplotlib, seaborn
- transformers (optional, for DeBERTa embeddings)

## Citation

If you use this work, please cite:

```
@misc{triagecopilot2026,
  title={TriageCoPilot: Multimodal Triage Decision Support with Clinical Interpretability and Equity Auditing},
  author={Muhammad Husnain Rasool},
  year={2026},
  note={Triagegeist Competition, Laitinen-Fredriksson Foundation}
}
```

## License

This project is for research and educational purposes only, in compliance with the Triagegeist Non-Commercial Research License.
