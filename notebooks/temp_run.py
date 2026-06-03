import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

# 1. Load your training data
train = pd.read_csv("../data/processed/train_processed.csv") # Update path if needed

# 2. Define X and y
DROP = ["record_id", "flood_risk_score"]
X = train.drop(columns=DROP)
y = train["flood_risk_score"]

# 3. Define your categorical features list (update with your actual categorical column names)
CAT_FEATURES = [
    "flood_occurrence_current_event",
    "water_presence_flag",
    "urban_rural",
    "road_quality",
    "electricity",
    "water_supply"
]

# Convert categorical features to string to avoid CatBoost float errors
for col in CAT_FEATURES:
    X[col] = X[col].astype(str)

# 4. Make sure your competition_metric_proxy is defined
from sklearn.metrics import r2_score, mean_squared_error
def competition_metric_proxy(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    balanced_error = (mae + rmse) / 2
    r2 = r2_score(y_true, y_pred)
    ev_penalty = max(0, 1 - r2)
    return balanced_error * (1 + ev_penalty)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_cat(trial):
    params = {
        'iterations'        : trial.suggest_int('iterations', 500, 2000),
        'learning_rate'     : trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth'             : trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg'       : trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count'      : trial.suggest_int('border_count', 32, 255),
        'random_strength'   : trial.suggest_float('random_strength', 0, 10),
        'random_seed'       : 42,
        'verbose'           : 0,
        'loss_function'     : 'RMSE',
        'early_stopping_rounds': 50
    }
    
    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for tr_idx, val_idx in kf.split(X):
        Xtr,  Xval  = X.iloc[tr_idx],  X.iloc[val_idx]
        
        # NOTE: np.log1p is used here for y, make sure this is intended!
        ytr,  yval  = np.log1p(y.iloc[tr_idx]), y.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(Xtr, ytr,
                  cat_features=CAT_FEATURES,
                  eval_set=(Xval, np.log1p(yval)),
                  verbose=False)
        
        preds = np.clip(np.expm1(model.predict(Xval)), 0, 1)
        scores.append(competition_metric_proxy(yval, preds))
    
    return np.mean(scores)

# Now run the study
study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(objective_cat, n_trials=50, show_progress_bar=True)

print(f"\nBest CatBoost score : {study_cat.best_value:.4f}")
print("Best params:")
for k, v in study_cat.best_params.items():
    print(f"  {k}: {v}")

# Save config
import json, os
os.makedirs('../configs', exist_ok=True)
with open('../configs/catboost_config.json', 'w') as f:
    json.dump(study_cat.best_params, f, indent=2)
