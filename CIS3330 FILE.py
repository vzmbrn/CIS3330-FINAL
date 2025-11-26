# Import pandas and read NUMBERS.csv
import pandas as pd
import matplotlib.pyplot as plt
# For linear regression
from sklearn.linear_model import LinearRegression #might switch to logistic regression
import numpy as np

# Read the NUMBERS.csv file, skipping metadata rows
numbers_df = pd.read_csv('NUMBERS.csv', skiprows=3)
print(numbers_df.head())

# Read the NOTES.csv file
notes_df = pd.read_csv('NOTES.csv')
print(notes_df.head())

# Create a box plot for the numerical columns in numbers_df
# Remove non-numeric columns for plotting
numeric_data = numbers_df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(16, 6))
plt.boxplot(numeric_data.values, vert=False)
plt.title('Box Plot of PPP Conversion Factor Data')
plt.xlabel('Value')
plt.yticks([])
plt.tight_layout()
plt.show()

# Example: Linear regression for Australia PPP conversion factor over time
# Filter for Australia
australia_row = numbers_df[numbers_df['Country Name'] == 'Australia']
years = np.array([int(col) for col in numbers_df.columns if col.isdigit()]).reshape(-1, 1)
values = australia_row.loc[:, australia_row.columns.str.isdigit()].values.flatten().astype(float)

# Remove missing values
mask = ~np.isnan(values)
years_clean = years[mask]
values_clean = values[mask]

if len(years_clean) > 1:
	model = LinearRegression()
	model.fit(years_clean, values_clean)
	predicted = model.predict(years_clean)
	plt.figure(figsize=(10, 5))
	plt.scatter(years_clean, values_clean, label='Actual')
	plt.plot(years_clean, predicted, color='red', label='Linear Regression')
	plt.title('Linear Regression: Australia PPP Conversion Factor Over Time')
	plt.xlabel('Year')
	plt.ylabel('PPP Conversion Factor')
	plt.legend()
	plt.tight_layout()
	plt.show()
else:
	print('Not enough data for linear regression.')

