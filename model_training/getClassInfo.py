import pandas as pd

# Read the CSV file
df = pd.read_csv("KKI_phenotypic.csv")

# Select only the 'ScanDir ID' and 'DX' columns
selected_columns = df[['ScanDir ID', 'DX']]

# Save the selected columns to a new CSV file
selected_columns.to_csv("class.csv", index=False)

df = pd.read_csv("class.csv")
df['DX_GROUP'] = df['DX_GROUP'].replace(0, 2)
df.to_csv("class_info.csv", index=False)