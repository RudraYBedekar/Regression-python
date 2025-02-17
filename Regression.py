import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_mo import LinearRegression
import numpy as np
url = "https://tidesandcurrents.noaa.gov/sltrends/data/8727520_meantrend.csv"
data = pd.read_csv(url, skiprows=3, on_bad_lines='skip', engine='python')
# first few rows
print(data.head())
# Load the dataset
data_url = "https://tidesandcurrents.noaa.gov/sltrends/data/8727520_meantrend.csv"
aa= pd.read_csv(data_url, header=None, skiprows=10, engine="python")
# descriptive column
bb= ['Year', 'Month', 'SeaLevel_mm', 'Extra_Col1', 'Extra_Col2', 'Extra_Col3', 'Misc']
# relevant columns
aa= seadata[['Year', 'SeaLevel_mm']]
# columns to numeric
seadata['Year'] = pd.to_numeric(seadata['Year'], errors='coerce')
seadata['SeaLevel_mm'] = pd.to_numeric(seadata['SeaLevel_mm'], errors='coerce')
# delete rows
aa= seadata.dropna()
# processed dataset
print("Processed Data:")
print(seadata.head())
# linear regression
X_values = seadata[['Year']].values.reshape(-1, 1) # Predictor: Year
y_values = seadata['SeaLevel_mm'].values # Target: Sea Level
reg_mo = LinearRegression()
reg_mo.fit(X_values, y_values)
# predictions
predicted_levels = reg_mo.predict(X_values)
# Plot data and regression
plt.figure(figsize=(10, 6))
plt.scatter(seadata['Year'], seadata['SeaLevelmm'], color='blue', label='Observed Data')
plt.plot(seadata['Year'], predicted_levels, color='red', label='Trend Line')
plt.title('Sea LevelRise: Cedar Key Trend')
plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
plt.legend()
plt.grid(True)
plt.show()
# regression mo performance
squaredvalue= reg_mo.score(X_values, y_values)
print(f"R-squared Score: {r_squared_value:.4f}")
print("Analysis: Higher R-squared values suggest better mo fit.")
# mo improvement
print("To improve the mo, consider:")
print("- Incorporating additional factors like temperature, tides, or weather patterns.")
print("- Trying advanced mos, such as polynomial regression.")
print("- Expanding the dataset with more historical or nearby station data.")
# elevation Cedar Key
cedarelevation= 1500 # Elevation in mm (1.5 meters converted to mm)
# Predict future levels
predict= np.arange(seadata['Year'].max(), 2100, 1).reshape(-1, 1)
fpredict= reg_mo.predict(years_to_predict)
# Identify sea level
if np.any(fpredict> cedar_key_elevation):
crossing_year = years_to_predict[fpredict> cedar_key_elevation][0][0]
print(f"The sea level is projected to surpass Cedar Key's elevation in {int(crossing_year)}.")
else:
print("The sea level is not projected to surpass Cedar Key's elevation by 2100.")
Mean
# elevation Cedar Key (in mm)
dd= 1500
# Predict future
fyears= np.arange(data['Year'].min(), 2100).reshape(-1, 1)
future_sea_levels = mo.predict(future_years)
exeyears= future_years[future_sea_levels > mean_elevation_cedar_key]
if exceed_years.size > 0:
first_exceed_year = exceed_years[0][0]
print(f"The sea level is predicted to exceed Cedar Key's elevation in the year {first_exceed_year}.")
else:
print("The sea level do not predicted to exceed Cedar Key's elevation by 2100.")
# mean elevation
print(f"Mean elevation of Cedar Key: {dd/ 1000} meters.")
residual
# Extract
X = data[['Year']].values
y = data[slevels'].values
mo = LinearRegression()
mo.fit(X, y)
y_pred = mo.predict(X)
residuals = y - y_pred
# Plot the residuals
plt.figure(figsize=(10, 6))
plt.scatter(X, residuals, color='red', label='Residuals')
plt.axhline(0, color='black',linewidth=1) # Horizontal line at 0
plt.title('Residual Plot for Sea Level Prediction at Cedar Key')
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()
# standard deviation of residuals
print(f"Mean of residuals: {residuals.mean():.4f}")
print(f"Standard Deviation of residuals: {residuals.std():.4f}")
Sea level data at Cedar Key shows an upward trend over time, according to the report. The plot's red
line, which represents the linear regression model, clearly illustrates the sea levels' upward trend. This
implies that sea levels at Cedar Key have been rising gradually. This pattern is supported by the scatter
plot of the observed data points (blue dots). Although more sophisticated models might be required for
more accuracy, the model offers a helpful tool for comprehending and forecasting future sea levels.
