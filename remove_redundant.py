import pandas as pd

# Load your Excel data
df = pd.read_excel("Book1.xlsx")

# Keep only numeric columns (drop strings like GH3, oligo, user_tag, etc.)
numeric_df = df.select_dtypes(include='number')

# Compute Pearson correlation matrix
corr_matrix = numeric_df.corr().abs()

# Find highly correlated feature pairs (excluding self-correlation)
threshold = 0.95
upper = corr_matrix.where(
    pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = [
    (column, idx, upper.loc[idx, column])
    for column in upper.columns
    for idx in upper.index
    if upper.loc[idx, column] > threshold
]

# Print or export results
for f1, f2, score in sorted(high_corr_pairs, key=lambda x: -x[2]):
    print(f"{f1} ↔ {f2} = {score:.2f}")
