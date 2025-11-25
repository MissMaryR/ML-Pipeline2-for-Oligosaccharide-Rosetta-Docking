import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- CONFIG ---
TOP_N = 30
IDENTIFIERS = ['GH3', 'oligo', 'description', 'active']

# --- Load Data ---
df = pd.read_excel("Book1.xlsx")

# Clean identifiers
df['GH3'] = df['GH3'].astype(str).str.strip()
df['oligo'] = df['oligo'].astype(str).str.strip()

# Identify docking score features only
score_features = [col for col in df.columns if col not in IDENTIFIERS]

# Aggregate by GH3–oligo pair using mean
agg_df = df.groupby(['GH3', 'oligo'])[score_features].mean().reset_index()

# Attach the 'active' label (assumes consistency across replicates)
labels = df.groupby(['GH3', 'oligo'])['active'].first().reset_index()
agg_df = pd.merge(agg_df, labels, on=['GH3', 'oligo'])

# Prepare model inputs
X = agg_df.drop(columns=['GH3', 'oligo', 'active'])
y = agg_df['active']
feature_names = X.columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# SHAP analysis
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test)

# Extract SHAP values for class 1
shap_class1 = shap_values.values[:, :, 1]
mean_abs_shap = pd.Series(
    data=abs(shap_class1).mean(axis=0),
    index=feature_names
).sort_values(ascending=False)

# Plot
top_shap = mean_abs_shap.head(TOP_N)
plt.figure(figsize=(10, 8))
top_shap.sort_values().plot(kind='barh', title=f"Top {TOP_N} SHAP Feature Importances (Class 1 - Active)")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig("shap_top_scores_aggregated.png", dpi=300)

# Print values
print(f"\n=== Top {TOP_N} Predictive Scores by SHAP (Aggregated) ===")
print(top_shap.to_string(float_format="%.4f"))
