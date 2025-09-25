# === Imports ===
from pathlib import Path
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import pyfixest as pf
from pyfixest.estimation import feols
from linearmodels import IV2SLS


# === Project Root and Key Paths ===
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if '__file__' in globals() else Path.cwd()
)

DATA_DIR = PROJECT_ROOT / '01_Data'
CODE_DIR = PROJECT_ROOT / '02_Code'
# Create the output directory if it doesn't exist
OUTPUT_DIR = PROJECT_ROOT / '03_Output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('CWD:', Path.cwd())
print('DATA_DIR:', DATA_DIR)
print('CODE_DIR:', CODE_DIR)
print('OUTPUT_DIR:', OUTPUT_DIR)

# === Load Data: Require data_yoghurt.csv ===
csv_path = DATA_DIR / 'data_yoghurt.csv'
if not csv_path.exists():
    raise FileNotFoundError(f"Required file not found: {csv_path}")

df = pd.read_csv(csv_path)
print(f'Loaded {csv_path.name}: {df.shape[0]} rows x {df.shape[1]} cols')

# === Product Name Mapping ===
product_names = {
    1: 'Yoplait',
    2: 'Chobani', 
    3: 'Dannon',
    4: 'Stonyfield Farm',
    5: 'Activia'
}
df['product_name'] = df['product'].map(product_names)

with pd.option_context('display.max_columns', 20, 'display.width', 120):
    print(df.head().to_string(index=False))

    varnames = list(df.columns)
    print("Variable names:", varnames)

#########
#  Q1
#########

# === Create Normalized Variables (per gram) ===
# Create normalized versions by dividing by weight (per gram)
df['sugar_per_gram'] = df['sugar'] / df['weight']
df['protein_per_gram'] = df['protein'] / df['weight']
df['calories_per_gram'] = df['calories'] / df['weight']

# === Table: City 1, Period 1 ===
# Filter data for city 1 and period 1
city1_period1 = df[(df['city'] == 1) & (df['period'] == 1)].copy()

# Select columns for the table
table_cols = ['product_name', 'price', 'share', 'weight', 'sugar_per_gram', 'protein_per_gram', 'calories_per_gram']
city1_table = city1_period1[table_cols].copy()

# Capitalize column names and remove underscores
city1_table.columns = ['Product', 'Price', 'Share', 'Weight', 'Sugar Per Gram', 'Protein Per Gram', 'Calories Per Gram']

print("\n=== City 1, Period 1 Table ===")
print(f"Number of observations: {len(city1_table)}")
print("\nTable with price, market share, weight, and normalized variables (per gram):")
with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.precision', 3):
    print(city1_table.to_string(index=False))

# === Export Table to LaTeX ===
# Create output directory if it doesn't exist
OUTPUT_DIR = PROJECT_ROOT / '03_Output'

# Export table to LaTeX file
latex_file = OUTPUT_DIR / 'q1_table.tex'
latex_out = city1_table.to_latex(index=False, float_format='%.3f')

# Extract only the tabular section
lines = latex_out.split('\n')
tabular_lines = []
in_tabular = False
for line in lines:
    if '\\begin{tabular}' in line:
        in_tabular = True
    if in_tabular:
        tabular_lines.append(line)
    if '\\end{tabular}' in line:
        break

tabular_only = '\n'.join(tabular_lines)

with open(latex_file, 'w') as f:
    f.write(tabular_only)

print(f"\nTable exported to: {latex_file}")



#########
#  Q3
#########

# === Additional Variables ===
# residual_share = 1 - sum(share) within ['product','city','period'] group
df['group_share_sum'] = df.groupby(['city', 'period'])['share'].transform('sum')
df['residual_share'] = 1 - df['group_share_sum'] 
df['sharediff'] = np.log(df['share']) - np.log(df['residual_share'])

# negprice = price * -1
df['negprice'] = df['price'] * -1

# === Q3 Regression: sharediff ~ negprice + weight + per-gram measures ===
regressors = ['negprice', 'weight', 'sugar_per_gram', 'protein_per_gram', 'calories_per_gram']
df_reg = df.dropna(subset=['sharediff'] + regressors).copy()
X = sm.add_constant(df_reg[regressors])
y = df_reg['sharediff']

model = sm.OLS(y, X).fit(cov_type='HC1')

summ = summary_col(results=[model],
                   float_format='%0.3f',
                   stars=True,
                   model_names=['Market Share'],
                   info_dict={'N':lambda x: f"{int(x.nobs)}"})

# Rename variables for readability in the LaTeX via text replacement
var_renames = {
    'const': 'Constant',
    'negprice': 'Negative Price',
    'weight': 'Weight',
    'sugar\\_per\\_gram': 'Sugar Per Gram',
    'protein\\_per\\_gram': 'Protein Per Gram',
    'calories\\_per\\_gram': 'Calories Per Gram',
}

latex_out = summ.as_latex()
for old, new in var_renames.items():
    latex_out = latex_out.replace(old, new)

# Extract only the tabular section and filter out R-squared rows
lines = latex_out.split('\n')
tabular_lines = []
in_tabular = False
for line in lines:
    if '\\begin{tabular}' in line:
        in_tabular = True
    if in_tabular:
        # Skip lines containing R-squared
        if 'R-squared' not in line:
            tabular_lines.append(line)
    if '\\end{tabular}' in line:
        break

tabular_only = '\n'.join(tabular_lines)

q3_path = OUTPUT_DIR / 'q3_reg.tex'
with open(q3_path, 'w') as f:
    f.write(tabular_only)

#########
#  Q4: Elasticity Matrix: NOTE !! ADD OUTSIDE OPTION TO THE MATRIX
#########

# === Get coefficient for negative price from regression ===
negprice_coeff = model.params['negprice']
print(f"\n=== Elasticity Matrix (City 1, Period 1) ===")
print(f"Negative price coefficient: {negprice_coeff:.6f}")

# === Get City 1, Period 1 data for elasticity calculation ===
city1_period1_elasticity = df[(df['city'] == 1) & (df['period'] == 1)].copy()
city1_period1_elasticity = city1_period1_elasticity.sort_values('product').reset_index(drop=True)

print(f"Products in city 1, period 1: {sorted(city1_period1_elasticity['product'].unique())}")
print(f"Number of products: {len(city1_period1_elasticity)}")

# === Calculate Elasticity Matrix ===
products = city1_period1_elasticity['product'].values
product_names_list = city1_period1_elasticity['product_name'].values
n_products = len(products)

# Calculate residual share (outside option)
residual_share = 1 - city1_period1_elasticity['share'].sum()
print(f"Residual share (outside option): {residual_share:.4f}")

# Create extended matrix with outside option
n_products_with_outside = n_products + 1
elasticity_matrix = np.zeros((n_products_with_outside, n_products_with_outside))

# Fill the matrix
for i in range(n_products_with_outside):
    for j in range(n_products_with_outside):
        if i == j:  # Own price elasticity (diagonal)
            if i < n_products:  # Regular product
                price_i = city1_period1_elasticity.iloc[i]['price']
                share_i = city1_period1_elasticity.iloc[i]['share']
                # Own price elasticity = (coeff on negative price) * price * (1 - market share) * (-1)
                elasticity_matrix[i, j] = negprice_coeff * price_i * (1 - share_i) * (-1)
            else:  # Outside option (diagonal element is 0 since outside option has no price)
                elasticity_matrix[i, j] = 0.0
        else:  # Cross price elasticity (off-diagonal)
            if j < n_products:
                price_j = city1_period1_elasticity.iloc[j]['price']
                share_j = city1_period1_elasticity.iloc[j]['share']
                # Cross price elasticity = price_j * market_share_j * (coeff on negative price)
                elasticity_matrix[i, j] = price_j * share_j * negprice_coeff
            else:
                # From outside option to regular product (outside option has no price, so elasticity is 0)
                elasticity_matrix[i, j] = 0.0

# === Create DataFrame for the elasticity matrix ===
product_labels = list(product_names_list) + ['Outside Option']
elasticity_df = pd.DataFrame(
    elasticity_matrix,
    index=product_labels,
    columns=product_labels
)

print("\nElasticity Matrix:")
print("Rows: Own product, Columns: Price change of product")
with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.precision', 4):
    print(elasticity_df.to_string())

# === Export to LaTeX ===
elasticity_latex = elasticity_df.to_latex(float_format='%.4f')

# Extract only the tabular section
lines = elasticity_latex.split('\n')
tabular_lines = []
in_tabular = False
for line in lines:
    if '\\begin{tabular}' in line:
        in_tabular = True
    if in_tabular:
        tabular_lines.append(line)
    if '\\end{tabular}' in line:
        break

tabular_only = '\n'.join(tabular_lines)

elasticity_path = OUTPUT_DIR / 'q4_elasticity_matrix.tex'
with open(elasticity_path, 'w') as f:
    f.write(tabular_only)

print(f"\nElasticity matrix exported to: {elasticity_path}")

#########
#  Q5: Diversion Ratio Matrix
#########

# === Calculate Diversion Ratio Matrix  ===
# Diversion ratio from j to k = share_k / (1 - share_j)
# Leave diagonal blank
# Include outside option as 6th row/column

# Calculate residual share (outside option)
residual_share = 1 - city1_period1_elasticity['share'].sum()
print(f"Residual share (outside option): {residual_share:.4f}")

# Create extended matrix with outside option
n_products_with_outside = n_products + 1
diversion_matrix = np.zeros((n_products_with_outside, n_products_with_outside))
diversion_matrix[:] = np.nan  # Initialize with NaN to show as blank

# Fill the matrix (excluding diagonal)
for i in range(n_products_with_outside):
    for j in range(n_products_with_outside):
        if i != j:  # Off-diagonal only
            if i < n_products and j < n_products:
                # Both are regular products
                share_k = city1_period1_elasticity.iloc[i]['share']  # share of product k (row)
                share_j = city1_period1_elasticity.iloc[j]['share']  # share of product j (column)
            elif i < n_products and j == n_products:
                # From outside option to regular product
                share_k = city1_period1_elasticity.iloc[i]['share']  # share of product k (row)
                share_j = residual_share  # share of outside option (column)
            elif i == n_products and j < n_products:
                # From regular product to outside option
                share_k = residual_share  # share of outside option (row)
                share_j = city1_period1_elasticity.iloc[j]['share']  # share of product j (column)
            else:
                # Both are outside option (shouldn't happen, but just in case)
                continue
            
            # Diversion ratio from j to k = share_k / (1 - share_j)
            diversion_matrix[i, j] = share_k / (1 - share_j)

# === Create DataFrame for the diversion ratio matrix ===
product_labels = list(product_names_list) + ['Outside Option']
diversion_df = pd.DataFrame(
    diversion_matrix.T,  # Transpose the matrix to flip axes
    index=product_labels,
    columns=product_labels
)

print("\n=== Diversion Ratio Matrix (City 1, Period 1) ===")
print("Rows: From product, Columns: To product")
print("Diversion ratio from j to k = share_k / (1 - share_j)")
with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.precision', 4):
    print(diversion_df.to_string())

# === Export to LaTeX ===
diversion_latex = diversion_df.to_latex(float_format='%.4f')

# Extract only the tabular section
lines = diversion_latex.split('\n')
tabular_lines = []
in_tabular = False
for line in lines:
    if '\\begin{tabular}' in line:
        in_tabular = True
    if in_tabular:
        tabular_lines.append(line)
    if '\\end{tabular}' in line:
        break

tabular_only = '\n'.join(tabular_lines)

diversion_path = OUTPUT_DIR / 'q5_diversion_matrix.tex'
with open(diversion_path, 'w') as f:
    f.write(tabular_only)

#########
#  Q7: Marginal Cost Calculation
#########

# === Calculate Marginal Costs ===
# Formula: mc_j = p_j - 1/(α(1-s_j))
# where α is the negative price coefficient from the regression

alpha = negprice_coeff
print(f"\n=== Marginal Cost Calculation (City 1, Period 1) ===")
print(f"Alpha (negative price coefficient): {alpha:.6f}")

# Calculate marginal costs for each product
mc_data = []
for i, row in city1_period1_elasticity.iterrows():
    product = row['product']
    price = row['price']
    share = row['share']
    
    # mc_j = p_j - 1/(α(1-s_j))
    mc = price - (1 / (alpha * (1 - share)))
    
    mc_data.append({
        'Product': city1_period1_elasticity.iloc[i]['product_name'],
        'Price': price,
        'Share': share,
        'Marginal Cost': mc
    })

# Create DataFrame with just marginal costs (1 row, 5 columns)
mc_values = [row['Marginal Cost'] for row in mc_data]
mc_df = pd.DataFrame([mc_values], columns=product_names_list)

print("\nMarginal Cost Table (1 row × 5 columns):")
with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.precision', 4):
    print(mc_df.to_string(index=False))

# === Export to LaTeX ===
def create_latex_tabular_mc_simple(df):
    """Create LaTeX tabular section for 1 row × 5 columns marginal cost table"""
    latex_lines = []
    latex_lines.append("\\begin{tabular}{" + "c" * len(df.columns) + "}")
    latex_lines.append("\\hline")
    
    # Header row
    header = " & ".join(df.columns)
    latex_lines.append(header + " \\\\")
    latex_lines.append("\\hline")
    
    # Data row (single row)
    row_str = " & ".join([f"{val:.4f}" for val in df.iloc[0].values])
    latex_lines.append(row_str + " \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    return "\n".join(latex_lines)

tabular_only = create_latex_tabular_mc_simple(mc_df)

mc_path = OUTPUT_DIR / 'q7_marginal_costs.tex'
with open(mc_path, 'w') as f:
    f.write(tabular_only)

print(f"\nMarginal cost table exported to: {mc_path}")

#########
#  Q10: 3-way Fixed Effects
#########
   
#########
# 10a. Regression, Table
######### 

# Run regression with 3-way fixed effects using pyfixest
model_3fe = feols(
    fml='sharediff ~ negprice | product + city + period',
    data=df,
    vcov='HC1'
)


print(model_3fe.summary())

summary_data = []
coef = model_3fe.coef()['negprice']
se = model_3fe.se()['negprice']
pval = model_3fe.pvalue()['negprice']
    
# Add stars based on p-value
stars = ''
if pval < 0.01:
    stars = '***'
elif pval < 0.05:
    stars = '**'
elif pval < 0.1:
    stars = '*'
    
summary_data.append({
    'Variable': var_renames.get('negprice', 'negprice'),
    'Coefficient': f"{coef:.3f}{stars}",
    'Std Error': f"({se:.3f})"
})

# Create LaTeX table
latex_lines = []
latex_lines.append("\\begin{tabular}{lc}")
latex_lines.append("\\hline")
latex_lines.append("& 3 Way FE \\\\")
latex_lines.append("\\hline")

for row in summary_data:
    latex_lines.append(f"{row['Variable']} & {row['Coefficient']} \\\\")
    latex_lines.append(f" & {row['Std Error']} \\\\")

latex_lines.append("\\hline")
latex_lines.append(f"Observations & {model_3fe._N} \\\\")
latex_lines.append("\\hline")
latex_lines.append("\\end{tabular}")

tabular_only_3fe = '\n'.join(latex_lines)

# Export to LaTeX
q10_path = OUTPUT_DIR / 'q10_reg_3way_fe.tex'
with open(q10_path, 'w') as f:
    f.write(tabular_only_3fe)

print(f"\nQ10 regression results exported to: {q10_path}")
print("Note: Regression uses 3-way fixed effects (product, city, period)")

#########
# 10b. Elasticity Matrix
######### 


# === Get coefficient for negative price from regression ===
negprice_coeff = coef
print(f"\n=== Elasticity Matrix (City 1, Period 1) ===")
print(f"Negative price coefficient: {coef:.6f}")

# === Get City 1, Period 1 data for elasticity calculation ===
city1_period1_elasticity = df[(df['city'] == 1) & (df['period'] == 1)].copy()
city1_period1_elasticity = city1_period1_elasticity.sort_values('product').reset_index(drop=True)

print(f"Products in city 1, period 1: {sorted(city1_period1_elasticity['product'].unique())}")
print(f"Number of products: {len(city1_period1_elasticity)}")

# === Calculate Elasticity Matrix ===
products = city1_period1_elasticity['product'].values
product_names_list = city1_period1_elasticity['product_name'].values
n_products = len(products)

# Calculate residual share (outside option)
residual_share = 1 - city1_period1_elasticity['share'].sum()
print(f"Residual share (outside option): {residual_share:.4f}")

# Create extended matrix with outside option
n_products_with_outside = n_products + 1
elasticity_matrix = np.zeros((n_products_with_outside, n_products_with_outside))

# Fill the matrix
for i in range(n_products_with_outside):
    for j in range(n_products_with_outside):
        if i == j:  # Own price elasticity (diagonal)
            if i < n_products:  # Regular product
                price_i = city1_period1_elasticity.iloc[i]['price']
                share_i = city1_period1_elasticity.iloc[i]['share']
                # Own price elasticity = (coeff on negative price) * price * (1 - market share) * (-1)
                elasticity_matrix[i, j] = negprice_coeff * price_i * (1 - share_i) * (-1)
            else:  # Outside option (diagonal element is 0 since outside option has no price)
                elasticity_matrix[i, j] = 0.0
        else:  # Cross price elasticity (off-diagonal)
            if j < n_products:
                price_j = city1_period1_elasticity.iloc[j]['price']
                share_j = city1_period1_elasticity.iloc[j]['share']
                # Cross price elasticity = price_j * market_share_j * (coeff on negative price)
                elasticity_matrix[i, j] = price_j * share_j * negprice_coeff
            else:
                elasticity_matrix[i, j] = 0.0

# === Create DataFrame for the elasticity matrix ===
product_labels = list(product_names_list) + ['Outside Option']
elasticity_df = pd.DataFrame(
    elasticity_matrix,
    index=product_labels,
    columns=product_labels
)

print("\nElasticity Matrix:")
print("Rows: Own product, Columns: Price change of product")
with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.precision', 4):
    print(elasticity_df.to_string())

# === Export to LaTeX ===
elasticity_latex = elasticity_df.to_latex(float_format='%.4f')

# Extract only the tabular section
lines = elasticity_latex.split('\n')
tabular_lines = []
in_tabular = False
for line in lines:
    if '\\begin{tabular}' in line:
        in_tabular = True
    if in_tabular:
        tabular_lines.append(line)
    if '\\end{tabular}' in line:
        break

tabular_only = '\n'.join(tabular_lines)

elasticity_path = OUTPUT_DIR / 'q10_elasticity_matrix.tex'
with open(elasticity_path, 'w') as f:
    f.write(tabular_only)

print(f"\nElasticity matrix exported to: {elasticity_path}")



#########
#  Q11: MC with 3-way FE
#########

alpha = negprice_coeff
print(f"\n=== Marginal Cost Calculation (City 1, Period 1) ===")
print(f"Alpha (negative price coefficient): {alpha:.6f}")

# Calculate marginal costs for each product
mc_data = []
for i, row in city1_period1_elasticity.iterrows():
    product = row['product']
    price = row['price']
    share = row['share']
    
    # mc_j = p_j - 1/(α(1-s_j))
    mc = price - (1 / (alpha * (1 - share)))
    
    mc_data.append({
        'Product': city1_period1_elasticity.iloc[i]['product_name'],
        'Price': price,
        'Share': share,
        'Marginal Cost': mc
    })

# Create DataFrame with just marginal costs (1 row, 5 columns)
mc_values = [row['Marginal Cost'] for row in mc_data]
mc_df = pd.DataFrame([mc_values], columns=product_names_list)

print("\nMarginal Cost Table (1 row × 5 columns):")
with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.precision', 4):
    print(mc_df.to_string(index=False))

# === Export to LaTeX ===
def create_latex_tabular_mc_simple(df):
    """Create LaTeX tabular section for 1 row × 5 columns marginal cost table"""
    latex_lines = []
    latex_lines.append("\\begin{tabular}{" + "c" * len(df.columns) + "}")
    latex_lines.append("\\hline")
    
    # Header row
    header = " & ".join(df.columns)
    latex_lines.append(header + " \\\\")
    latex_lines.append("\\hline")
    
    # Data row (single row)
    row_str = " & ".join([f"{val:.4f}" for val in df.iloc[0].values])
    latex_lines.append(row_str + " \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    return "\n".join(latex_lines)

tabular_only = create_latex_tabular_mc_simple(mc_df)

mc_path = OUTPUT_DIR / 'q11_marginal_costs.tex'
with open(mc_path, 'w') as f:
    f.write(tabular_only)

print(f"\nMarginal cost table exported to: {mc_path}")

#########
#  Q13: 3-Way FE with IV (distance × diesel)
#########

#########
# 13a. Regression, Table
######### 

# === Create instrumental variable ===
# Instrument: distance × diesel
df['distance_diesel'] = df['distance'] * df['diesel']

print(f"\n=== 3-Way FE Regression with IV (distance × diesel) ===")
print("Instrumenting for negprice with distance × diesel")

# Run 3-way FE regression with IV using pyfixest
model_3fe_iv = feols(
    fml='sharediff ~ 1 | product + city + period | negprice ~ distance_diesel',
    data=df,
    vcov='HC1'
)

print(model_3fe_iv.summary())

# Extract coefficient and statistics
coef_iv = model_3fe_iv.coef()['negprice']  # IV coefficient
se_iv = model_3fe_iv.se()['negprice']
pval_iv = model_3fe_iv.pvalue()['negprice']

# Add stars based on p-value
stars_iv = ''
if pval_iv < 0.01:
    stars_iv = '***'
elif pval_iv < 0.05:
    stars_iv = '**'
elif pval_iv < 0.1:
    stars_iv = '*'

# Create LaTeX table
latex_lines_iv = []
latex_lines_iv.append("\\begin{tabular}{lc}")
latex_lines_iv.append("\\hline")
latex_lines_iv.append("& 3 Way FE + IV \\\\")
latex_lines_iv.append("\\hline")

latex_lines_iv.append(f"Negative Price & {coef_iv:.3f}{stars_iv} \\\\")
latex_lines_iv.append(f" & ({se_iv:.3f}) \\\\")

latex_lines_iv.append("\\hline")
latex_lines_iv.append(f"Observations & {model_3fe_iv._N} \\\\")
latex_lines_iv.append("\\hline")
latex_lines_iv.append("\\end{tabular}")

tabular_only_3fe_iv = '\n'.join(latex_lines_iv)

# Export to LaTeX
q13_path = OUTPUT_DIR / 'q13_reg_3way_fe_iv.tex'
with open(q13_path, 'w') as f:
    f.write(tabular_only_3fe_iv)

print(f"\nQ13 regression results exported to: {q13_path}")
print("Note: Regression uses 3-way fixed effects with IV (distance × diesel)")

#########
# 13b. Elasticity Matrix
######### 


# === Get coefficient for negative price from regression ===
negprice_coeff = coef_iv
print(f"\n=== Elasticity Matrix (City 1, Period 1) ===")
print(f"Negative price coefficient: {coef:.6f}")

# === Get City 1, Period 1 data for elasticity calculation ===
city1_period1_elasticity = df[(df['city'] == 1) & (df['period'] == 1)].copy()
city1_period1_elasticity = city1_period1_elasticity.sort_values('product').reset_index(drop=True)

print(f"Products in city 1, period 1: {sorted(city1_period1_elasticity['product'].unique())}")
print(f"Number of products: {len(city1_period1_elasticity)}")

# === Calculate Elasticity Matrix  ===
products = city1_period1_elasticity['product'].values
product_names_list = city1_period1_elasticity['product_name'].values
n_products = len(products)

# Calculate residual share (outside option)
residual_share = 1 - city1_period1_elasticity['share'].sum()
print(f"Residual share (outside option): {residual_share:.4f}")

# Create extended matrix with outside option
n_products_with_outside = n_products + 1
elasticity_matrix = np.zeros((n_products_with_outside, n_products_with_outside))

# Fill the matrix
for i in range(n_products_with_outside):
    for j in range(n_products_with_outside):
        if i == j:  # Own price elasticity (diagonal)
            if i < n_products:  # Regular product
                price_i = city1_period1_elasticity.iloc[i]['price']
                share_i = city1_period1_elasticity.iloc[i]['share']
                # Own price elasticity = (coeff on negative price) * price * (1 - market share) * (-1)
                elasticity_matrix[i, j] = negprice_coeff * price_i * (1 - share_i) * (-1)
            else:  # Outside option (diagonal element is 0 since outside option has no price)
                elasticity_matrix[i, j] = 0.0
        else:  # Cross price elasticity (off-diagonal)
            if j < n_products:
                # Both are regular products
                price_j = city1_period1_elasticity.iloc[j]['price']
                share_j = city1_period1_elasticity.iloc[j]['share']
                # Cross price elasticity = price_j * market_share_j * (coeff on negative price)
                elasticity_matrix[i, j] = price_j * share_j * negprice_coeff
            else:
                elasticity_matrix[i, j] = 0.0

# === Create DataFrame for the elasticity matrix ===
product_labels = list(product_names_list) + ['Outside Option']
elasticity_df = pd.DataFrame(
    elasticity_matrix,
    index=product_labels,
    columns=product_labels
)

print("\nElasticity Matrix:")
print("Rows: Own product, Columns: Price change of product")
with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.precision', 4):
    print(elasticity_df.to_string())

# === Export to LaTeX ===
elasticity_latex = elasticity_df.to_latex(float_format='%.4f')

# Extract only the tabular section
lines = elasticity_latex.split('\n')
tabular_lines = []
in_tabular = False
for line in lines:
    if '\\begin{tabular}' in line:
        in_tabular = True
    if in_tabular:
        tabular_lines.append(line)
    if '\\end{tabular}' in line:
        break

tabular_only = '\n'.join(tabular_lines)

elasticity_path = OUTPUT_DIR / 'q13_elasticity_matrix.tex'
with open(elasticity_path, 'w') as f:
    f.write(tabular_only)

print(f"\nElasticity matrix exported to: {elasticity_path}")

#########
#  Q14: MC with 3-way FE
#########

alpha = negprice_coeff
print(f"\n=== Marginal Cost Calculation (City 1, Period 1) ===")
print(f"Alpha (negative price coefficient): {alpha:.6f}")

# Calculate marginal costs for each product
mc_data = []
for i, row in city1_period1_elasticity.iterrows():
    product = row['product']
    price = row['price']
    share = row['share']
    
    # mc_j = p_j - 1/(α(1-s_j))
    mc = price - (1 / (alpha * (1 - share)))
    
    mc_data.append({
        'Product': city1_period1_elasticity.iloc[i]['product_name'],
        'Price': price,
        'Share': share,
        'Marginal Cost': mc
    })

# Create DataFrame with just marginal costs (1 row, 5 columns)
mc_values = [row['Marginal Cost'] for row in mc_data]
mc_df = pd.DataFrame([mc_values], columns=product_names_list)

print("\nMarginal Cost Table (1 row × 5 columns):")
with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.precision', 4):
    print(mc_df.to_string(index=False))

# === Export to LaTeX ===
def create_latex_tabular_mc_simple(df):
    """Create LaTeX tabular section for 1 row × 5 columns marginal cost table"""
    latex_lines = []
    latex_lines.append("\\begin{tabular}{" + "c" * len(df.columns) + "}")
    latex_lines.append("\\hline")
    
    # Header row
    header = " & ".join(df.columns)
    latex_lines.append(header + " \\\\")
    latex_lines.append("\\hline")
    
    # Data row (single row)
    row_str = " & ".join([f"{val:.4f}" for val in df.iloc[0].values])
    latex_lines.append(row_str + " \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    return "\n".join(latex_lines)

tabular_only = create_latex_tabular_mc_simple(mc_df)

mc_path = OUTPUT_DIR / 'q14_marginal_costs.tex'
with open(mc_path, 'w') as f:
    f.write(tabular_only)

print(f"\nMarginal cost table exported to: {mc_path}")

#########
#  Q15: Iterative Pricing Algorithm
#########

# === Run IV regression with explicit product dummies ===
print(f"\n=== IV Regression with Explicit Product Dummies ===")

# Create dummy variables for products, cities, and periods
products = sorted(df['product'].unique())
cities = sorted(df['city'].unique())
periods = sorted(df['period'].unique())

# Create dummy variables (excluding reference categories)
for product in products[:-1]:
    df[f'product_{product}'] = (df['product'] == product).astype(int)
for city in cities[:-1]:
    df[f'city_{city}'] = (df['city'] == city).astype(int)
for period in periods[:-1]:
    df[f'period_{period}'] = (df['period'] == period).astype(int)

# Prepare regression variables
all_dummy_vars = ' + '.join([f'product_{p}' for p in products[:-1]] + 
                           [f'city_{c}' for c in cities[:-1]] + 
                           [f'period_{p}' for p in periods[:-1]])

# Run IV regression using pyfixest with explicit dummies
model_iv_dummies = feols(
    fml=f'sharediff ~ {all_dummy_vars} | negprice ~ distance_diesel',
    data=df,
    vcov='HC1'
)

print("\nIV Regression Results with Product Dummies:")
print(model_iv_dummies.summary())

# Extract alpha
alpha_iv = model_iv_dummies.coef()['negprice']

# Extract all dummy coefficients
coef_dict = model_iv_dummies.coef()

# Get product dummy coefficients
product_fe_coeffs = {product: coef_dict.get(f'product_{product}', 0.0) for product in products}

# Get city dummy coefficient for city 1
city1_fe_coeff = coef_dict.get('city_1', 0.0)

# Get period dummy coefficient for period 1
period1_fe_coeff = coef_dict.get('period_1', 0.0)

# Get intercept (constant term)
intercept = coef_dict.get('Intercept', 0.0)

# Use the IV alpha coefficient
alpha = alpha_iv

# Extract residuals (error terms) from the regression
residuals = model_iv_dummies.resid()
# Create xi_dict with residuals for each product in city 1, period 1
# Filter data for city 1, period 1 to get the residuals for our target market
city1_period1_mask = (df['city'] == 1) & (df['period'] == 1)
city1_period1_data = df[city1_period1_mask].copy()
city1_period1_residuals = residuals[city1_period1_mask]

# Create xi_dict mapping product to its residual
xi_dict = {}
for i, (_, row) in enumerate(city1_period1_data.iterrows()):
    product = row['product']
    xi_dict[product] = city1_period1_residuals[i]

print(f"\nXi values (residuals) for city 1, period 1:")
for product in sorted(xi_dict.keys()):
    print(f"  Product {product}: {xi_dict[product]:.6f}")

# Define target city and period for the algorithm
target_city = 1
target_period = 1

# === Get marginal costs from Q14 ===
# We already calculated these above, let's extract them
mc_values = mc_df.iloc[0].values  # Get the marginal cost values
mc_dict = {product: mc_values[i] for i, product in enumerate(products)}
print(f"Marginal costs: {mc_dict}")

# === Iterative Pricing Algorithm ===
def iterative_pricing_algorithm(alpha, product_fe_coeffs, xi_dict, mc_dict, products, max_iter=1000, tolerance=1e-5):
    """
    Iterative pricing algorithm to find equilibrium prices.
    
    Parameters:
    - alpha: price coefficient from regression
    - product_fe_coeffs: dictionary of product fixed effect coefficients
    - xi_dict: dictionary of unobserved product characteristics (error terms) by product
    - mc_dict: dictionary of marginal costs by product
    - products: list of product IDs
    - max_iter: maximum number of iterations
    - tolerance: convergence tolerance
    
    Returns:
    - equilibrium_prices: dictionary of equilibrium prices by product
    - equilibrium_shares: dictionary of equilibrium market shares by product
    - iteration_info: dictionary with convergence information
    """
    
    # Initialize price vector
    prices = np.ones(len(products))
    
    print(f"\nStarting iterative pricing algorithm...")
    print(f"Initial prices: {prices}")
    print(f"Alpha: {alpha:.6f}")
    print(f"Xi values: {xi_dict}")
    print(f"Tolerance: {tolerance}")
    
    iteration = 0
    converged = False
    
    while iteration < max_iter and not converged:
        iteration += 1
        
        # Store previous prices for convergence check
        prev_prices = prices.copy()
        
        # Calculate market shares for current prices
        # u_j = intercept + alpha * negprice + product_fe_coeff + city_fe_coeff + period_fe_coeff + error
        # negprice = -price (negative of price)
        # For city 1, period 1, we use those specific fixed effects
        
        utilities = []
        for i, product in enumerate(products):
            negprice = -prices[i]
            product_fe = product_fe_coeffs[product]
            city_fe = city1_fe_coeff
            period_fe = period1_fe_coeff
            xi_j = xi_dict[product]
            u_j = intercept + alpha * negprice + product_fe + city_fe + period_fe + xi_j
            utilities.append(u_j)
        
        utilities = np.array(utilities)
        
        # Calculate market shares: s_j = u_j / sum_i(u_i)
        shares = np.exp(utilities) / (np.sum(np.exp(utilities)) + 1)
        market_shares = shares
        
        # Calculate new prices: p_j = mc_j + 1/((1-alpha)*s_j)
        new_prices = np.zeros(len(products))
        for i, product in enumerate(products):
            mc_j = mc_dict[product]
            s_j = market_shares[i]
            new_prices[i] = mc_j + 1 / ((1 - s_j) * alpha)
        
        prices = new_prices
        
        # Check convergence: sum of absolute differences
        price_diff = np.sum(np.abs(prices - prev_prices))
        
        if iteration % 10 == 0 or price_diff < tolerance:
            print(f"Iteration {iteration}: Price diff = {price_diff:.8f}")
            print(f"  Prices: {prices}")
            print(f"  Shares: {market_shares}")
        
        if price_diff < tolerance:
            converged = True
            print(f"\nConverged after {iteration} iterations!")
            print(f"Final price difference: {price_diff:.8f}")
    
    if not converged:
        print(f"\nWarning: Algorithm did not converge after {max_iter} iterations")
        print(f"Final price difference: {price_diff:.8f}")
    
    # Create output dictionaries
    equilibrium_prices = {product: prices[i] for i, product in enumerate(products)}
    equilibrium_shares = {product: market_shares[i] for i, product in enumerate(products)}
    
    iteration_info = {
        'iterations': iteration,
        'converged': converged,
        'final_price_diff': price_diff,
        'tolerance': tolerance
    }
    
    return equilibrium_prices, equilibrium_shares, iteration_info

# === Run the algorithm ===
equilibrium_prices, equilibrium_shares, iteration_info = iterative_pricing_algorithm(
    alpha=alpha,
    product_fe_coeffs=product_fe_coeffs,
    mc_dict=mc_dict,
    products=products,
    xi_dict=xi_dict
)

# === Compare with observed data ===
print(f"\n=== Comparison with Observed Data (City 1, Period 1) ===")
city1_period1_obs = df[(df['city'] == 1) & (df['period'] == 1)].copy()
city1_period1_obs = city1_period1_obs.sort_values('product').reset_index(drop=True)

# === Export results to LaTeX ===
# Create a comparison table
comparison_data = []
for i, row in city1_period1_obs.iterrows():
    product = row['product']
    product_name = row['product_name']
    obs_price = row['price']
    obs_share = row['share']
    eq_price = equilibrium_prices[product]
    eq_share = equilibrium_shares[product]
    
    comparison_data.append({
        'Product': product_name,
        'Observed Price': f"{obs_price:.4f}",
        'Equilibrium Price': f"{eq_price:.4f}",
        'Observed Share': f"{obs_share:.4f}",
        'Equilibrium Share': f"{eq_share:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)

# Create LaTeX table
latex_lines = []
latex_lines.append("\\begin{tabular}{lcccc}")
latex_lines.append("\\hline")
latex_lines.append("Product & Observed Price & Equilibrium Price & Observed Share & Equilibrium Share \\\\")
latex_lines.append("\\hline")

for _, row in comparison_df.iterrows():
    latex_lines.append(f"{row['Product']} & {row['Observed Price']} & {row['Equilibrium Price']} & {row['Observed Share']} & {row['Equilibrium Share']} \\\\")

latex_lines.append("\\hline")
latex_lines.append("\\end{tabular}")

tabular_only = '\n'.join(latex_lines)

# Export to LaTeX
q15_path = OUTPUT_DIR / 'q15_equilibrium_prices.tex'
with open(q15_path, 'w') as f:
    f.write(tabular_only)

print(f"\nEquilibrium results exported to: {q15_path}")

#########
#  Q16: Merged Firm Pricing (Chobani + Dannon)
#########

# === Merged Firm Iterative Pricing Algorithm ===
def merged_firm_pricing_algorithm(alpha, product_fe_coeffs, mc_dict, products, max_iter=1000, tolerance=1e-4):
    """
    Iterative pricing algorithm for merged firm (Chobani + Dannon).
    
    Products 2 (Chobani) and 3 (Dannon) are now merged and internalize cross-price effects.
    Their best response functions become:
    p_c = original_fn + (p_d - mc_d) * s_d / (1 - s_c)
    p_d = original_fn + (p_c - mc_c) * s_c / (1 - s_d)
    
    Products 1, 4, 5 remain unchanged (independent pricing).
    """
    
    # Initialize price vector (start with equilibrium prices from previous algorithm)
    prices = np.ones(len(products))
    
    print(f"\nStarting merged firm pricing algorithm...")
    print(f"Initial prices: {prices}")
    print(f"Alpha: {alpha:.6f}")
    print(f"Tolerance: {tolerance}")
    print(f"Merged products: Chobani (2) and Dannon (3)")
    
    iteration = 0
    converged = False
    
    while iteration < max_iter and not converged:
        iteration += 1
        
        # Store previous prices for convergence check
        prev_prices = prices.copy()
        
        # Calculate market shares for current prices
        utilities = []
        for i, product in enumerate(products):
            negprice = -prices[i]
            product_fe = product_fe_coeffs[product]
            city_fe = city1_fe_coeff
            period_fe = period1_fe_coeff
            xi_j = xi_dict[product]
            u_j = intercept + alpha * negprice + product_fe + city_fe + period_fe + xi_j
            utilities.append(u_j)
        
        utilities = np.array(utilities)
        
        # Calculate market shares: s_j = u_j / sum_i(u_i)
        shares = np.exp(utilities) / (np.sum(np.exp(utilities)) + 1)
        market_shares = shares
        
        # Calculate new prices
        new_prices = np.zeros(len(products))
        
        for i, product in enumerate(products):
            mc_j = mc_dict[product]
            s_j = market_shares[i]
            
            if product == 2:  # Chobani (merged firm)
                # p_c = original_fn + (p_d - mc_d) * s_d / (1 - s_c)
                # Original function: mc_c + 1/(α(1-s_c))
                original_price = mc_j + 1 / (alpha * (1 - s_j))
                
                # Find Dannon's current price and market share
                dannon_idx = products.index(3)
                p_d = prices[dannon_idx]
                s_d = market_shares[dannon_idx]
                mc_d = mc_dict[3]
                
                # Add cross-price effect
                cross_effect = (p_d - mc_d) * s_d / (1 - s_j)
                new_prices[i] = original_price + cross_effect
                
            elif product == 3:  # Dannon (merged firm)
                # p_d = original_fn + (p_c - mc_c) * s_c / (1 - s_d)
                # Original function: mc_d + 1/(α(1-s_d))
                original_price = mc_j + 1 / (alpha * (1 - s_j))
                
                # Find Chobani's current price and market share
                chobani_idx = products.index(2)
                p_c = prices[chobani_idx]
                s_c = market_shares[chobani_idx]
                mc_c = mc_dict[2]
                
                # Add cross-price effect
                cross_effect = (p_c - mc_c) * s_c / (1 - s_j)
                new_prices[i] = original_price + cross_effect
                
            else:  # Products 1, 4, 5 (unchanged behavior)
                # Standard pricing: p_j = mc_j + 1/(α(1-s_j))
                new_prices[i] = mc_j + 1 / (alpha * (1 - s_j))
        
        prices = new_prices
        
        # Check convergence: sum of absolute differences
        price_diff = np.sum(np.abs(prices - prev_prices))
        
        if iteration % 10 == 0 or price_diff < tolerance:
            print(f"Iteration {iteration}: Price diff = {price_diff:.8f}")
            print(f"  Prices: {prices}")
            print(f"  Shares: {market_shares}")
        
        if price_diff < tolerance:
            converged = True
            print(f"\nConverged after {iteration} iterations!")
            print(f"Final price difference: {price_diff:.8f}")
    
    if not converged:
        print(f"\nWarning: Algorithm did not converge after {max_iter} iterations")
        print(f"Final price difference: {price_diff:.8f}")
    
    # Create output dictionaries
    equilibrium_prices = {product: prices[i] for i, product in enumerate(products)}
    equilibrium_shares = {product: market_shares[i] for i, product in enumerate(products)}
    
    iteration_info = {
        'iterations': iteration,
        'converged': converged,
        'final_price_diff': price_diff,
        'tolerance': tolerance
    }
    
    return equilibrium_prices, equilibrium_shares, iteration_info

# === Run the merged firm algorithm ===
print(f"\n=== Merged Firm Analysis (Chobani + Dannon) ===")
merged_prices, merged_shares, merged_info = merged_firm_pricing_algorithm(
    alpha=alpha,
    product_fe_coeffs=product_fe_coeffs,
    mc_dict=mc_dict,
    products=products
)

# === Export comparison to LaTeX ===
comparison_data = []
for product in products:
    product_name = product_names[product]
    pre_price = equilibrium_prices[product]
    post_price = merged_prices[product]
    change = post_price - pre_price
    pct_change = (change / pre_price) * 100
    
    comparison_data.append({
        'Product': product_name,
        'Pre-Merger Price': f"{pre_price:.4f}",
        'Post-Merger Price': f"{post_price:.4f}",
        'Percent Change': f"{pct_change:.2f}\%"
    })

comparison_df = pd.DataFrame(comparison_data)

# Create LaTeX table
latex_lines = []
latex_lines.append("\\begin{tabular}{lccc}")
latex_lines.append("\\hline")
latex_lines.append("Product & Pre-Merger Price & Post-Merger Price & Percent Change \\\\")
latex_lines.append("\\hline")

for _, row in comparison_df.iterrows():
    latex_lines.append(f"{row['Product']} & {row['Pre-Merger Price']} & {row['Post-Merger Price']} & {row['Percent Change']} \\\\")

latex_lines.append("\\hline")
latex_lines.append("\\end{tabular}")

tabular_only = '\n'.join(latex_lines)

# Export to LaTeX
q16_path = OUTPUT_DIR / 'q16_merger_comparison.tex'
with open(q16_path, 'w') as f:
    f.write(tabular_only)

print(f"\nMerger comparison results exported to: {q16_path}")

#########
#  Q17: Welfare Analysis of Merger
#########

# === Calculate Consumer Surplus (Inclusive Value) ===
def calculate_consumer_surplus(prices, alpha, product_fe_coeffs, city_fe, period_fe, products):
    """
    Calculate consumer surplus using the inclusive value formula:
    CS = ln(Sum(exp(u_j))) * (1/alpha)
    where u_j includes all products (0 to 5) including outside option (u_0 = 0)
    """
    
    # Calculate utilities for all products
    utilities = []
    for i, product in enumerate(products):
        negprice = -prices[i]
        product_fe = product_fe_coeffs[product]
        xi_j = xi_dict[product]
        u_j = intercept + alpha * negprice + product_fe + city_fe + period_fe + xi_j
        utilities.append(u_j)
    
    # Add outside option (u_0 = 0)
    utilities.append(0.0)
    sum_exp_utilities = np.sum(np.exp(utilities))
    # Calculate: ln(Sum(exp(u_j))) * (1/alpha)
    sum_exp_utilities = np.sum(np.exp(utilities))
    inclusive_value = np.log(sum_exp_utilities) * (1 / alpha)
    
    return inclusive_value

# === Calculate welfare for both scenarios ===
print(f"\n=== Welfare Analysis of Merger ===")

# Original prices (pre-merger)
original_prices_array = np.array([equilibrium_prices[product] for product in products])
cs_original = calculate_consumer_surplus(
    original_prices_array, alpha, product_fe_coeffs, city1_fe_coeff, period1_fe_coeff, products
)

# Merged firm prices (post-merger)
merged_prices_array = np.array([merged_prices[product] for product in products])
cs_merged = calculate_consumer_surplus(
    merged_prices_array, alpha, product_fe_coeffs, city1_fe_coeff, period1_fe_coeff, products
)

# Calculate welfare change
welfare_change = cs_merged - cs_original

# === Calculate total expenditure at original prices ===
# Get original market shares
original_shares = np.array([equilibrium_shares[product] for product in products])

# Calculate total expenditure (prices * shares for products 1-5, excluding outside option)
total_expenditure = 0
for i, product in enumerate(products):
    total_expenditure += original_prices_array[i] * original_shares[i]

# Calculate welfare change as percentage of total expenditure
welfare_change_pct_exp = (welfare_change / total_expenditure) * 100

# === Display results ===
print(f"Pre-Merger Consumer Surplus:")
print(f"  CS = ln(Sum(exp(u_j))) * (1/alpha): {cs_original:.6f}")
print(f"Post-Merger Consumer Surplus:")
print(f"  CS = ln(Sum(exp(u_j))) * (1/alpha): {cs_merged:.6f}")
print(f"")
print(f"Total Expenditure at Original Prices: {total_expenditure:.6f}")
print(f"")
print(f"Welfare Change:")
print(f"  Change in CS: {welfare_change:.6f}")
print(f"  As % of Total Expenditure: {welfare_change_pct_exp:.2f}%")

#########
# Q20 Nested Logit Model with IV and Fixed Effects
#########

# === Create Nested Groups ===
# Nest 1: Chobani (2) and Dannon (3)
# Nest 2: Yoplait (1), Stonyfield Farm (4), Activia (5)

df['nest'] = 0  # Initialize
df.loc[df['product'].isin([2, 3]), 'nest'] = 1  # Chobani and Dannon
df.loc[df['product'].isin([1, 4, 5]), 'nest'] = 2  # Others

# === Calculate share_in_nest ===
# For each product, calculate its share within its nest
df['nest_share_sum'] = df.groupby(['city', 'period', 'nest'])['share'].transform('sum')
df['share_in_nest'] = df['share'] / df['nest_share_sum']
df['l_share_in_nest'] = np.log(df['share_in_nest'])

# === Create Second Instrument: Number of Unique Products in Nest ===
# For each nest, count the number of unique products
df['products_in_nest'] = df.groupby(['city', 'period', 'nest'])['product'].transform('nunique')


# === Run Nested Logit Regression with IV and 3-way Fixed Effects ===
# Redefine all_dummy_vars as a list for linearmodels
all_dummy_vars_list = [f'product_{p}' for p in products[:-1]] + \
                      [f'city_{c}' for c in cities[:-1]] + \
                      [f'period_{p}' for p in periods[:-1]]

# Create design matrices
y = df['sharediff']
endog = df[['negprice', 'l_share_in_nest']]  # Endogenous variables
exog = df[all_dummy_vars_list]  # Exogenous variables (dummies)
instruments = df[['distance_diesel', 'products_in_nest']]  # Instruments

# Add constant to exogenous variables
exog = sm.add_constant(exog)

# Run IV regression using linearmodels
model_nested_iv = IV2SLS(y, exog, endog, instruments).fit(cov_type='robust')

print(model_nested_iv.summary)

# === Extract Coefficients and Statistics ===
coef_negprice = model_nested_iv.params['negprice']
se_negprice = model_nested_iv.std_errors['negprice']
pval_negprice = model_nested_iv.pvalues['negprice']

coef_share_nest = model_nested_iv.params['l_share_in_nest']
se_share_nest = model_nested_iv.std_errors['l_share_in_nest']
pval_share_nest = model_nested_iv.pvalues['l_share_in_nest']

# Add stars based on p-values
def get_stars(pval):
    if pval < 0.01:
        return '***'
    elif pval < 0.05:
        return '**'
    elif pval < 0.1:
        return '*'
    else:
        return ''

stars_negprice = get_stars(pval_negprice)
stars_share_nest = get_stars(pval_share_nest)

# === Create LaTeX Table ===
latex_lines = []
latex_lines.append("\\begin{tabular}{lc}")
latex_lines.append("\\hline")
latex_lines.append("& Nested Logit + IV \\\\")
latex_lines.append("\\hline")

latex_lines.append(f"Negative Price & {coef_negprice:.3f}{stars_negprice} \\\\")
latex_lines.append(f" & ({se_negprice:.3f}) \\\\")
latex_lines.append("")
latex_lines.append(f"Log Share in Nest & {coef_share_nest:.3f}{stars_share_nest} \\\\")
latex_lines.append(f" & ({se_share_nest:.3f}) \\\\")

latex_lines.append("\\hline")
latex_lines.append(f"Observations & {len(df)} \\\\")
latex_lines.append("\\hline")
latex_lines.append("\\end{tabular}")

tabular_only_nested = '\n'.join(latex_lines)

# === Export to LaTeX ===
nested_path = OUTPUT_DIR / 'q20_nested_logit_iv.tex'
with open(nested_path, 'w') as f:
    f.write(tabular_only_nested)

print(f"\nNested Logit IV regression results exported to: {nested_path}")

#########
# Q21 Nested Logit Own and Cross Price Elasticities
#########

# Calculate own and cross price elasticities using nested logit formulas
# Own: (alpha*price/(1-rho)) * (1 - rho*shareinnest - (1-rho)*share) * (-1)
# Cross_jk (same nest): (alpha*price_k/(1-rho)) * (rho*shareinnest_k + (1-rho)*share_k)
# Cross_jk (different nest): alpha*price_k*share_k

# Extract nested logit parameters
alpha_nested = coef_negprice  # Coefficient on negprice
rho = coef_share_nest  # Coefficient on share_in_nest

# Get data for city 1, period 1
city1_period1_data = df[(df['city'] == 1) & (df['period'] == 1)].copy()

# Calculate residual share (outside option)
residual_share = 1 - city1_period1_data['share'].sum()
print(f"Residual share (outside option): {residual_share:.4f}")

# Create extended matrix with outside option
n_products_with_outside = len(products) + 1
elasticity_matrix = np.zeros((n_products_with_outside, n_products_with_outside))

for i in range(n_products_with_outside):
    for k in range(n_products_with_outside):
        if i < len(products) and k < len(products):
            # Both are regular products
            product_j = products[i]
            product_k = products[k]
            
            # Get data for product k
            row_k = city1_period1_data[city1_period1_data['product'] == product_k].iloc[0]
            price_k = row_k['price']
            share_k = row_k['share']
            share_in_nest_k = row_k['share_in_nest']
            nest_k = row_k['nest']
            
            # Get nest for product j
            row_j = city1_period1_data[city1_period1_data['product'] == product_j].iloc[0]
            nest_j = row_j['nest']
            
            if i == k:  # Own price elasticity
                # Own: (alpha*price/(1-rho)) * (1 - rho*shareinnest - (1-rho)*share) * (-1)
                elasticity = (alpha_nested * price_k / (1 - rho)) * (1 - rho * share_in_nest_k - (1 - rho) * share_k) * (-1)
            else:  # Cross price elasticity
                if nest_j == nest_k:  # Same nest
                    # Cross_jk (same nest): (alpha*price_k/(1-rho)) * (rho*shareinnest_k + (1-rho)*share_k)
                    elasticity = (alpha_nested * price_k / (1 - rho)) * (rho * share_in_nest_k + (1 - rho) * share_k)
                else:  # Different nest
                    # Cross_jk (different nest): alpha*price_k*share_k
                    elasticity = alpha_nested * price_k * share_k
            
            elasticity_matrix[i, k] = elasticity
            
        elif i < len(products) and k == len(products):
            # From outside option to regular product (outside option has no price, so elasticity is 0)
            elasticity_matrix[i, k] = 0.0
            
        elif i == len(products) and k < len(products):
            # From regular product to outside option
            product_k = products[k]
            row_k = city1_period1_data[city1_period1_data['product'] == product_k].iloc[0]
            price_k = row_k['price']
            share_k = row_k['share']
            # Cross price elasticity = alpha*price_k*share_k (same as different nest case)
            elasticity = alpha_nested * price_k * share_k
            elasticity_matrix[i, k] = elasticity
            
        else:
            # Both are outside option (diagonal element is 0 since outside option has no price)
            elasticity_matrix[i, k] = 0.0

# Create DataFrame for elasticity matrix
product_labels = [product_names[p] for p in products] + ['Outside Option']
elasticity_df = pd.DataFrame(
    elasticity_matrix,
    index=product_labels,
    columns=product_labels
)

print("Nested Logit Price Elasticity Matrix:")
print(elasticity_df.round(4))

# Create LaTeX table for elasticity matrix
def create_elasticity_latex_table(df):
    latex_lines = []
    latex_lines.append("\\begin{tabular}{l" + "c" * len(df.columns) + "}")
    latex_lines.append("\\hline")
    
    # Header row
    header = "& " + " & ".join(df.columns) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")
    
    # Data rows
    for idx, row in df.iterrows():
        row_str = f"{idx} & " + " & ".join([f"{val:.3f}" for val in row.values]) + " \\\\"
        latex_lines.append(row_str)
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    return "\n".join(latex_lines)

elasticity_latex = create_elasticity_latex_table(elasticity_df)

# Export elasticity matrix to LaTeX
elasticity_path = OUTPUT_DIR / 'q21_nested_elasticity_matrix.tex'
with open(elasticity_path, 'w') as f:
    f.write(elasticity_latex)

print(f"\nNested Logit elasticity matrix exported to: {elasticity_path}")

#########
# Q21b. Nested Logit Diversion Matrix
#########

# Calculate nested logit diversion ratios using the formulas:
# Same nest: D_jk = (rho * shareinnest_k + (1-rho) * s_k) / (1 - rho * shareinnest_j - (1-rho) * s_j)
# Not same nest: D_jk = ((1-rho) * s_k) / (1 - rho * shareinnest_j - (1-rho) * s_j)

# Get data for city 1, period 1
city1_period1_diversion = df[(df['city'] == 1) & (df['period'] == 1)].copy()

# Calculate residual share (outside option)
residual_share = 1 - city1_period1_diversion['share'].sum()
print(f"Residual share (outside option): {residual_share:.4f}")

# Create extended matrix with outside option
n_products_with_outside = len(products) + 1
diversion_matrix_nested = np.zeros((n_products_with_outside, n_products_with_outside))
diversion_matrix_nested[:] = np.nan  # Initialize with NaN to show as blank

# Fill the matrix (excluding diagonal)
for i in range(n_products_with_outside):
    for j in range(n_products_with_outside):
        if i != j:  # Off-diagonal only
            if i < len(products) and j < len(products):
                # Both are regular products
                product_j = products[j]  # From product (column)
                product_k = products[i]  # To product (row)
                
                # Get data for both products
                row_j = city1_period1_diversion[city1_period1_diversion['product'] == product_j].iloc[0]
                row_k = city1_period1_diversion[city1_period1_diversion['product'] == product_k].iloc[0]
                
                s_j = row_j['share']  # share of product j (from)
                s_k = row_k['share']  # share of product k (to)
                shareinnest_j = row_j['share_in_nest']  # share in nest of product j
                shareinnest_k = row_k['share_in_nest']  # share in nest of product k
                nest_j = row_j['nest']  # nest of product j
                nest_k = row_k['nest']  # nest of product k
                
                # Calculate denominator: 1 - rho * shareinnest_j - (1-rho) * s_j
                denominator = 1 - rho * shareinnest_j - (1 - rho) * s_j
                
                if nest_j == nest_k:  # Same nest
                    # D_jk = (rho * shareinnest_k + (1-rho) * s_k) / denominator
                    numerator = rho * shareinnest_k + (1 - rho) * s_k
                else:  # Different nest
                    # D_jk = ((1-rho) * s_k) / denominator
                    numerator = (1 - rho) * s_k
                
                diversion_matrix_nested[i, j] = numerator / denominator
                
            elif i < len(products) and j == len(products):
                # From outside option to regular product
                product_k = products[i]  # To product (row)
                row_k = city1_period1_diversion[city1_period1_diversion['product'] == product_k].iloc[0]
                s_k = row_k['share']
                shareinnest_k = row_k['share_in_nest']
                
                # From outside option: denominator = 1 - residual_share = sum of all product shares
                denominator = 1 - residual_share
                
                # For outside option, we use the same nest formula but with outside option characteristics
                # Outside option is not in any nest, so we treat it as different nest
                numerator = (1 - rho) * s_k
                diversion_matrix_nested[i, j] = numerator / denominator
                
            elif i == len(products) and j < len(products):
                # From regular product to outside option
                product_j = products[j]  # From product (column)
                row_j = city1_period1_diversion[city1_period1_diversion['product'] == product_j].iloc[0]
                s_j = row_j['share']
                shareinnest_j = row_j['share_in_nest']
                
                # Calculate denominator: 1 - rho * shareinnest_j - (1-rho) * s_j
                denominator = 1 - rho * shareinnest_j - (1 - rho) * s_j
                
                # To outside option: numerator = (1-rho) * residual_share
                numerator = (1 - rho) * residual_share
                diversion_matrix_nested[i, j] = numerator / denominator

# Create DataFrame for the nested logit diversion ratio matrix
product_labels_nested = [product_names[p] for p in products] + ['Outside Option']
diversion_df_nested = pd.DataFrame(
    diversion_matrix_nested,
    index=product_labels_nested,
    columns=product_labels_nested
)

print("\n=== Nested Logit Diversion Ratio Matrix (City 1, Period 1) ===")
print("Rows: To product, Columns: From product")
print("Same nest: D_jk = (rho * shareinnest_k + (1-rho) * s_k) / (1 - rho * shareinnest_j - (1-rho) * s_j)")
print("Different nest: D_jk = ((1-rho) * s_k) / (1 - rho * shareinnest_j - (1-rho) * s_j)")
with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.precision', 4):
    print(diversion_df_nested.to_string())

# === Export to LaTeX ===
diversion_latex_nested = diversion_df_nested.to_latex(float_format='%.4f')

# Extract only the tabular section
lines = diversion_latex_nested.split('\n')
tabular_lines = []
in_tabular = False
for line in lines:
    if '\\begin{tabular}' in line:
        in_tabular = True
    if in_tabular:
        tabular_lines.append(line)
    if '\\end{tabular}' in line:
        break

tabular_only_nested = '\n'.join(tabular_lines)

diversion_path_nested = OUTPUT_DIR / 'q21_diversion_matrix.tex'
with open(diversion_path_nested, 'w') as f:
    f.write(tabular_only_nested)

print(f"\nNested Logit diversion matrix exported to: {diversion_path_nested}")

#########
# Q22 Nested Logit Marginal Costs for City 1 Period 1
#########

# Calculate nested logit marginal costs using the formula:
# mc = p - (1-rho)/(alpha*(1-(rho*shareinnest) - (1-rho)*share))

nested_mc_results = {}

for _, row in city1_period1_data.iterrows():
    product = row['product']
    price = row['price']
    share = row['share']
    share_in_nest = row['share_in_nest']
    
    # Calculate marginal cost using nested logit formula
    denominator = alpha_nested * (1 - (rho * share_in_nest) - (1 - rho) * share)
    mc_nested = price - (1 - rho) / denominator
    
    nested_mc_results[product] = {
        'price': price,
        'share': share,
        'share_in_nest': share_in_nest,
        'mc_nested': mc_nested
    }
    

# Create DataFrame for nested logit marginal costs (MC only)
nested_mc_df = pd.DataFrame({
    'Product': list(nested_mc_results.keys()),
    'Nested_MC': [nested_mc_results[p]['mc_nested'] for p in nested_mc_results.keys()]
})

print(nested_mc_df.round(4))

# Create LaTeX table for nested logit marginal costs (MC only) - single row format
def create_nested_mc_latex_table(df):
    # Product labels mapping
    product_labels = {
        1: "Yoplait",
        2: "Chobani", 
        3: "Dannon",
        4: "Stonyfield Farm",
        5: "Activia"
    }
    
    latex_lines = []
    latex_lines.append("\\begin{tabular}{l" + "c" * len(df) + "}")
    latex_lines.append("\\hline")
    
    # Header row with product labels
    header = "Nested MC & " + " & ".join([product_labels[int(row['Product'])] for _, row in df.iterrows()]) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")
    
    # Single data row with marginal costs
    data_row = "& " + " & ".join([f"{row['Nested_MC']:.3f}" for _, row in df.iterrows()]) + " \\\\"
    latex_lines.append(data_row)
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    return "\n".join(latex_lines)

nested_mc_latex = create_nested_mc_latex_table(nested_mc_df)

# Export nested logit marginal costs to LaTeX
nested_mc_path = OUTPUT_DIR / 'q22_nested_marginal_costs.tex'
with open(nested_mc_path, 'w') as f:
    f.write(nested_mc_latex)

print(f"\nNested Logit marginal costs exported to: {nested_mc_path}")

#########
# Q23 Nested Logit Post-Merger Pricing Algorithm
#########

# Define which products are merged (assuming products 2 and 3 merge)
merged_products = [2, 3]  # Chobani and Dannon
unmerged_products = [1, 4, 5]  # Others

# Define nest assignments (same as in nested logit model)
# Nest 1: Products 2 and 3 (Chobani and Dannon)
# Nest 2: Products 1, 4, and 5 (Others)
nests = [2, 1, 1, 2, 2]  # nest for each product

# Get marginal costs from nested logit calculation
nested_mc_dict = {product: nested_mc_results[product]['mc_nested'] for product in products}

# Extract residuals (error terms) from the regression
nested_residuals = model_nested_iv.resids
# Create xi_dict with residuals for each product in city 1, period 1
# Filter data for city 1, period 1 to get the residuals for our target market
city1_period1_mask = (df['city'] == 1) & (df['period'] == 1)
city1_period1_data = df[city1_period1_mask].copy()
city1_period1_residuals = nested_residuals[city1_period1_mask]

# Create xi_dict mapping product to its residual
nested_xi_dict = {}
for i, (_, row) in enumerate(city1_period1_data.iterrows()):
    product = row['product']
    nested_xi_dict[product] = city1_period1_residuals[i]

print(f"\nXi values (residuals) for city 1, period 1:")
for product in sorted(nested_xi_dict.keys()):
    print(f"  Product {product}: {nested_xi_dict[product]:.6f}")

nested_coef_dict = model_nested_iv.params

# Get intercept
nested_intercept = nested_coef_dict['const']

# Get product dummy coefficients for nested logit
nested_product_fe_coeffs = {product: nested_coef_dict.get(f'product_{product}', 0.0) for product in products}

# Get city dummy coefficient for city 1
nested_city1_fe_coeff = nested_coef_dict.get('city_1', 0.0)

# Get period dummy coefficient for period 1
nested_period1_fe_coeff = nested_coef_dict.get('period_1', 0.0)

# Nested logit pricing algorithm for post-merger
def nested_logit_merged_pricing_algorithm(alpha, rho, mc_dict, products, merged_products, max_iter=1000, tolerance=1e-5):
    """
    Nested logit pricing algorithm for merged firms
    """
    
    # Initialize prices
    prices = np.array([1.0] * len(products))  # Start with price = 1
    
    print(f"Starting nested logit merged pricing algorithm...")
    print(f"Max iterations: {max_iter}")
    print(f"Tolerance: {tolerance}")
    
    iteration = 0
    converged = False
    
    while iteration < max_iter and not converged:
        iteration += 1
        
        # Store previous prices for convergence check
        prev_prices = prices.copy()
        
        # Calculate market shares for current prices
        # For nested logit, we need to calculate shares within nests
        utilities = []
        for i, product in enumerate(products):
            negprice = -prices[i]
            product_fe = nested_product_fe_coeffs[product]
            city_fe = nested_city1_fe_coeff
            period_fe = nested_period1_fe_coeff
            xi_j = nested_xi_dict[product]
            intercept = nested_intercept
            u_j = intercept + alpha * negprice + product_fe + city_fe + period_fe + xi_j
            utilities.append(u_j)
        
        utilities = np.array(utilities)
        
        # Calculate market shares: s_j = u_j / sum_i(u_i)
        shares = np.exp(utilities) / (np.sum(np.exp(utilities)) + 1)
        market_shares = shares
        
        # Calculate share_in_nest for each product
        share_in_nest = []
        for i, product in enumerate(products):
            nest = nests[i]
            # Calculate total share in this nest
            nest_total_share = sum(market_shares[j] for j in range(len(products)) if nests[j] == nest)
            share_in_nest.append(market_shares[i] / nest_total_share)
        
        # Calculate new prices
        new_prices = np.zeros(len(products))
        
        for i, product in enumerate(products):
            mc_j = mc_dict[product]
            s_j = market_shares[i]
            s_in_nest_j = share_in_nest[i]
            
            if product in merged_products:
                
                # Find the other merged product
                other_merged = [p for p in merged_products if p != product][0]
                other_idx = products.index(other_merged)
                
                p_other = prices[other_idx]
                mc_other = mc_dict[other_merged]
                s_other = market_shares[other_idx]
                s_in_nest_other = share_in_nest[other_idx]
                
                # Calculate the merged product price
                denominator = alpha * (1 - rho * s_in_nest_j - (1 - rho) * s_j)
                cross_numerator = alpha * (p_other - mc_other) * (rho * s_in_nest_other + (1 - rho) * s_other)
                
                new_prices[i] = mc_j + ((1-rho) + cross_numerator) / denominator
                
            else:
                denominator = alpha * (1 - rho * s_in_nest_j - (1 - rho) * s_j)
                new_prices[i] = mc_j + (1 - rho) / denominator
        
        prices = new_prices
        
        # Check convergence: sum of absolute differences
        price_diff = np.sum(np.abs(prices - prev_prices))
        
        if iteration % 10 == 0 or price_diff < tolerance:
            print(f"Iteration {iteration}: Price diff = {price_diff:.8f}")
            print(f"  Prices: {prices}")
            print(f"  Shares: {market_shares}")
        
        if price_diff < tolerance:
            converged = True
            print(f"\nConverged after {iteration} iterations!")
            print(f"Final price difference: {price_diff:.8f}")
    
    if not converged:
        print(f"\nDid not converge after {max_iter} iterations")
        print(f"Final price difference: {price_diff:.8f}")
    
    return prices, market_shares

# Run the nested logit merged pricing algorithm
nested_merged_prices, nested_merged_shares = nested_logit_merged_pricing_algorithm(
    alpha_nested, rho, nested_mc_dict, products, merged_products
)

# Create results dictionary
nested_merged_results = {}
for i, product in enumerate(products):
    nested_merged_results[product] = {
        'price': nested_merged_prices[i],
        'share': nested_merged_shares[i]
    }

print(f"\nNested Logit Post-Merger Results:")
for product in products:
    print(f"Product {product}: Price = {nested_merged_results[product]['price']:.4f}, Share = {nested_merged_results[product]['share']:.4f}")

# Create new merger comparison table with nested logit results
def create_nested_merger_comparison_latex():
    # Product labels mapping
    product_labels = {
        1: "Yoplait",
        2: "Chobani", 
        3: "Dannon",
        4: "Stonyfield Farm",
        5: "Activia"
    }
    
    latex_lines = []
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("\\hline")
    latex_lines.append("Product & Pre-Merger & Post-Merger (Simple) & Post-Merger (Nested) & Change (Nested) \\\\")
    latex_lines.append("\\hline")
    
    for product in products:
        pre_price = equilibrium_prices[product]
        post_price = merged_prices[product]
        nested_post_price = nested_merged_results[product]['price']
        change = nested_post_price - pre_price
        
        latex_lines.append(f"{product_labels[product]} & {pre_price:.3f} & {post_price:.3f} & {nested_post_price:.3f} & {change:+.3f} \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    return "\n".join(latex_lines)

nested_merger_latex = create_nested_merger_comparison_latex()

# Export new merger comparison table with nested logit results to LaTeX
nested_merger_path = OUTPUT_DIR / 'q23_merger_comparison_nested.tex'
with open(nested_merger_path, 'w') as f:
    f.write(nested_merger_latex)

print(f"\nNew merger comparison table with nested logit results exported to: {nested_merger_path}")

#########
# Q24 Nested Logit Welfare Analysis
#########

# Extract nested logit parameters and fixed effects
alpha_nested_welfare = coef_negprice  # Coefficient on negprice
rho_welfare = coef_share_nest  # Coefficient on share_in_nest

# Calculate consumer surplus using nested logit inclusive value formula
def calculate_nested_consumer_surplus(prices, alpha, rho, product_fe_coeffs, city_fe, period_fe, products, nests):
    """
    Calculate consumer surplus using the nested logit inclusive value formula:
    CS = ln(Sum(exp(u_j))) * (1/alpha)
    where u_j includes all products (0 to 5) including outside option (u_0 = 0)
    """
    
    # Calculate utilities for all products
    utilities = []
    for i, product in enumerate(products):
        negprice = -prices[i]
        product_fe = nested_product_fe_coeffs[product]
        xi_j = nested_xi_dict[product]
        u_j = intercept + alpha * negprice + product_fe + city_fe + period_fe + xi_j
        utilities.append(u_j)
    
    # Add outside option (u_0 = 0)
    utilities.append(0.0)
    
    # Calculate inclusive value: ln(Sum(exp(u_j))) * (1/alpha)
    sum_exp_utilities = np.sum(np.exp(utilities))
    inclusive_value = np.log(sum_exp_utilities) * (1 / alpha)
    
    return inclusive_value

# Calculate welfare for both scenarios using nested logit
print(f"\n=== Nested Logit Welfare Analysis ===")

# Original prices (pre-merger) - using the original equilibrium prices
original_prices_array = np.array([equilibrium_prices[product] for product in products])
cs_original_nested = calculate_nested_consumer_surplus(
    original_prices_array, alpha_nested_welfare, rho_welfare, 
    nested_product_fe_coeffs, nested_city1_fe_coeff, nested_period1_fe_coeff, 
    products, nests
)

# Nested logit merged firm prices (post-merger)
nested_merged_prices_array = np.array([nested_merged_results[product]['price'] for product in products])
cs_merged_nested = calculate_nested_consumer_surplus(
    nested_merged_prices_array, alpha_nested_welfare, rho_welfare,
    nested_product_fe_coeffs, nested_city1_fe_coeff, nested_period1_fe_coeff,
    products, nests
)

# Calculate welfare change
welfare_change_nested = cs_merged_nested - cs_original_nested

# Calculate total expenditure at original prices
original_shares = np.array([equilibrium_shares[product] for product in products])
total_expenditure = 0
for i, product in enumerate(products):
    total_expenditure += original_prices_array[i] * original_shares[i]

# Calculate welfare change as percentage of total expenditure
welfare_change_pct_exp_nested = (welfare_change_nested / total_expenditure) * 100

# Display results
print(f"Pre-Merger Consumer Surplus (Nested Logit):")
print(f"  CS = ln(Sum(exp(u_j))) * (1/alpha): {cs_original_nested:.6f}")
print(f"Post-Merger Consumer Surplus (Nested Logit):")
print(f"  CS = ln(Sum(exp(u_j))) * (1/alpha): {cs_merged_nested:.6f}")
print(f"")
print(f"Total Expenditure at Original Prices: {total_expenditure:.6f}")
print(f"")
print(f"Welfare Change (Nested Logit):")
print(f"  Change in CS: {welfare_change_nested:.6f}")
print(f"  As % of Total Expenditure: {welfare_change_pct_exp_nested:.2f}%")

# Create LaTeX table for nested logit welfare analysis
def create_nested_welfare_latex_table():
    latex_lines = []
    latex_lines.append("\\begin{tabular}{lc}")
    latex_lines.append("\\hline")
    latex_lines.append("& Nested Logit \\\\")
    latex_lines.append("\\hline")
    latex_lines.append(f"Pre-Merger CS & {cs_original_nested:.6f} \\\\")
    latex_lines.append(f"Post-Merger CS & {cs_merged_nested:.6f} \\\\")
    latex_lines.append(f"Change in CS & {welfare_change_nested:.6f} \\\\")
    latex_lines.append(f"Change as % of Expenditure & {welfare_change_pct_exp_nested:.2f}\\% \\\\")
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    return "\n".join(latex_lines)

nested_welfare_latex = create_nested_welfare_latex_table()

# Export nested logit welfare analysis to LaTeX
nested_welfare_path = OUTPUT_DIR / 'q24_nested_welfare_analysis.tex'
with open(nested_welfare_path, 'w') as f:
    f.write(nested_welfare_latex)

print(f"\nNested logit welfare analysis exported to: {nested_welfare_path}")