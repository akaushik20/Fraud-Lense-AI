# Fraud-Lense-AI

Fraud detection system using IEEE-CIS dataset with multi-stage feature selection and XGBoost modeling.

## Dataset
- **Source**: IEEE-CIS Fraud Detection (Kaggle)
- **Size**: 590,540 transactions, 434 features
- **Imbalance**: 3.5% fraud rate (27.4:1 ratio)

## Progress

### ✅ Data Pipeline
- Helper functions for loading and preprocessing ([src/helper.py](src/helper.py))
- Memory optimization (downcast dtypes)
- Categorical encoding with LabelEncoder

### ✅ Feature Selection ([src/feature_selection.py](src/feature_selection.py))
**5-stage pipeline:**
1. Remove constant features (0 found)
2. Remove high missing >90% (12 removed)
3. Remove correlated >0.95 (124 removed, kept higher target correlation)
4. Remove low variance <0.01
5. XGBoost importance selection (top 150 features)

**Results**: 433 → 150 features (65% reduction)

### ✅ EDA
- ydata-profiling reports (minimal mode for speed)
- Sample size: 50k rows, ~6 min runtime

### 📊 Outputs
- `outputs/feature_selection/selected_features.txt` - Final feature list
- `outputs/feature_selection/feature_importance_scores.csv` - XGBoost rankings
- `outputs/feature_selection/feature_selection_report.txt` - Detailed analysis

## Next Steps
- [ ] Implement training pipeline ([model/train.py](model/train.py))
- [ ] Time-based train/validation split
- [ ] Model evaluation and tuning
- [ ] Deployment artifacts

## Environment
- Python 3.13.4
- Key libraries: pandas, numpy, scikit-learn, xgboost, ydata-profiling