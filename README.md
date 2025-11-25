# work in progress

# Pipeline for analyzing Rosetta Docking Scores


## 1. run remove_redundant.py to find redundant scores

   run with 
   ```
   python remove_redundant.py
   ```
   needs excel file called Book1.xlsx in same directory\
   keeps only numeric columns\
   threshold is 0.95\
   prints scores that are >0.95 correlated



### my results:

### over 0.95 is highly redundant and one should be dropped:

fa_rep.1 ↔ fa_rep = 1.00\
hbond_sc.1 ↔ hbond_sc = 1.00\
total_score.1 ↔ total_score = 1.00\
total_score_X ↔ dock_grid_grid_X = 1.00\
SR_3_interf_E_1_2 ↔ SR_3_total_score = 1.00\
if_X_fa_rep ↔ SR_3_fa_rep = 1.00\
SR_3_all_cst ↔ all_cst = 1.00\
fa_atr ↔ SR_3 = 1.00\
fa_sol ↔ fa_atr = 0.99\
fa_sol ↔ SR_3 = 0.99\
SR_3 ↔ tot_hbond_pm = 0.98\
fa_atr ↔ tot_hbond_pm = 0.98\
fa_elec ↔ tot_hbond_pm = 0.98\
fa_sol ↔ tot_hbond_pm = 0.98\
if_X_hbond_sc ↔ SR_3_hbond_sc = 0.96\
fa_elec ↔ SR_3 = 0.96\
fa_elec ↔ fa_atr = 0.96\
ref ↔ fa_sol = 0.95\
fa_sol ↔ fa_elec = 0.95\
ref ↔ fa_atr = 0.95

- removed redundant scores and saved as new Book1 excel

## 2. run 1_shap.py to generate a bar plot of features
- use to find important features to focus on
- this trains a Random Forest classifier
- uses SHAP values to explain feature contributions.
- Outputs and plots the top 30 most important features.

run with 
   ```
   conda create -n shap_env python=3.10       
   conda activate shap_env   
   conda install -c conda-forge shap scikit-learn pandas matplotlib openpyxl
   python 1_shap.py
   ```

## 3. run 2_model.py based on the top 15 shap features
- generates a plot based on the top 15 features
- uses
  - pandas for data handling
  - shap for explainable ML
  - sklearn for the classifier and data split
  - matplotlib for plotting
  
- Groups rows by GH3 and oligo, taking the mean of features.
- Then attaches the first active label for each group (assumes it's consistent).
- 80/20 split, stratified to maintain class balance
- Random Forest with 100 trees and balanced class weights (helps with class imbalance).
- Creates a SHAP explainer for the trained model.
- Explains predictions on test data.
- Extracts SHAP values for class 1 (active).
- Computes mean absolute SHAP values per feature.
- Plots the top 30 most important features.
- Saves the plot as shap_top_scores_aggregated.png.
- You get a ranked list and a bar plot of the top predictive docking scores for classifying active compounds.
- This is a typical pipeline for combining feature aggregation, classification, and explainability in molecular or biochemical datasets.
  
run with 
   ```
   python 2_model.py
   ```
