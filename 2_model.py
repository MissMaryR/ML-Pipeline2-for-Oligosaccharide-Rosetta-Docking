import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# Top 15 SHAP features
top_15_features = [
    'tot_burunsat_pm',
    'user_tag',
    'pro_close',
    'tot_nlsurfaceE_pm',
    'yhh_planarity',
    'total_score_per_residue',
    'SR_2_fa_rep',
    'p_aa_pp',
    'SR_1_nlpstat_pm',
    'SR_1_pstat_pm',
    'if_X_fa_elec',
    'tot_NLconts_pm',
    'omega',
    'tot_total_charge',
    'fa_dun'
]

# Load and aggregate dataset
df = pd.read_excel("Book1.xlsx")
df['GH3'] = df['GH3'].astype(str).str.strip()
df['oligo'] = df['oligo'].astype(str).str.strip()

# Aggregate features by GH3–oligo
agg_df = df.groupby(['GH3', 'oligo'])[top_15_features].mean().reset_index()

# Merge labels
labels = df.groupby(['GH3', 'oligo'])['active'].first().reset_index()
agg_df = pd.merge(agg_df, labels, on=['GH3', 'oligo'])

X = agg_df[top_15_features].copy()
y = agg_df['active'].copy()

# Prepare for CV
from sklearn.metrics import roc_auc_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_preds = []
all_probs = []
all_true = []

threshold = 0.3

# Cross-validation
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    all_preds.extend(y_pred)
    all_probs.extend(y_proba)
    all_true.extend(y_test)

# Plot dataframe
plot_df = pd.DataFrame({
    'TrueLabel': all_true,
    'PredictedProb': all_probs,
    'Correct': np.array(all_preds) == np.array(all_true)
})

plt.figure(figsize=(8, 6))
sns.violinplot(
    data=plot_df,
    x='TrueLabel',
    y='PredictedProb',
    inner=None,
    color='gray',
    alpha=0.3
)
sns.swarmplot(
    data=plot_df,
    x='TrueLabel',
    y='PredictedProb',
    hue='Correct',
    palette={True: 'green', False: 'red'},
    dodge=False,
    size=5
)
plt.xticks([0, 1], ['Inactive (0)', 'Active (1)'])
plt.ylabel("Predicted Probability of Being Active")
plt.xlabel("True Label")
plt.title(f"Model Confidence by True Activity Label (Threshold={threshold})")
plt.tight_layout()
plt.savefig("swarm_violin_top15_thresholded.png", dpi=300)

# SHAP analysis on final model
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

# Mean absolute SHAP values
mean_shap = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
shap_df = pd.Series(mean_shap, index=top_15_features).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
shap_df.plot(kind='barh')
plt.title("SHAP Feature Importances (Mean |SHAP value|) – Top 15")
plt.xlabel("Mean Absolute SHAP Value")
plt.tight_layout()
plt.savefig("shap_bar_plot_top15.png", dpi=300)
