import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

"""
This script does the following:
- for each nation, append the corresponding Olympic NOC country code
- for each series, fit a linear model and use it to extrapolate missing values
"""


# Load NOC values to append
noc_labels = pd.read_csv('data/noc_countries.csv')

# load economic data
economicData = pd.read_csv('data/Economic/Data.csv')
economicData = economicData.assign(**{'2024': np.nan})
economicData = pd.merge(economicData, noc_labels,
                        left_on='Country Name', right_on='country', how='left')
economicData.drop(columns=['country', 'Country Name',
                  'Country Code', 'Series Name'], inplace=True)
economicData.rename(columns=lambda x: x.split(' ')[0], inplace=True)

# type check
for col in economicData.columns[1:-1]:  # Skip 'Series' and 'noc' columns
    economicData[col] = pd.to_numeric(economicData[col], errors='coerce')

rows = economicData.iloc[:, 1:-1]
years = np.array(rows.columns.values)

missing_series = []
for row_i in range(0, rows.shape[0]):
    row = rows.iloc[row_i]
    values = row[years].values

    non_nan_indices = ~np.isnan(values)
    existing_years = years[non_nan_indices].reshape(-1, 1)
    existing_values = values[non_nan_indices]

    missing_indices = np.isnan(values)
    missing_years = years[missing_indices].reshape(-1, 1)

    if len(existing_years) > 1 and len(missing_years) > 0:
        model = LinearRegression()
        model.fit(existing_years, existing_values)

        missing_indices = np.isnan(values)
        missing_years = years[missing_indices].reshape(-1, 1)

        predicted_values = model.predict(missing_years)
        missing_years.flatten()

        rows.loc[row_i, years[missing_indices]] = predicted_values

    else:
        missing_series.append(economicData.iloc[row_i]['Series'])

    economicData.iloc[row_i, 1:-1] = rows.iloc[row_i]

# from visualizing CAN data, the 'SI.POV.NAHC' series must be dropped to retain CAN training data
NAHC_indices = economicData[economicData['Series'] == "SI.POV.NAHC"].index
economicData = economicData.drop(NAHC_indices)

missing_series = np.unique(
    np.array(economicData[economicData.isna().any(axis=1)]["noc"]).astype(str))
mask = economicData['noc'].isin(missing_series)
economicData = economicData[~mask]
economicData.dropna(inplace=True)

economicData.to_csv('out/economic_data.csv', index=False)
