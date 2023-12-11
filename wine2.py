import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
import squarify

palette = sns.color_palette("Spectral")

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('wine.csv')

# Preparing the data for regression analysis
X = df.drop(['quality'], axis = 1)  # Independent variables (excluding 'quality')
y = df['quality']                 # Dependent variable ('quality')

# Adding a constant term to the model (for the intercept)
X = sm.add_constant(X)

# Create and fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())

df.shape

df.isnull().sum()

# Scaling
# quality > 6.5 = good
# quality < 6.5 = bad
df['quality'] = df['quality'].apply(lambda x: 1 if x > 6.5 else 0)

# Sampling
X = df.drop(['quality'], axis = 1)
y = df.quality

# Adding a constant to the model (intercept)
X = sm.add_constant(X)

# Creating the regression model
model = sm.OLS(y, X).fit()

# Convert the p-values to standard decimal notation
p_values_decimal = model.pvalues.apply(lambda x: format(x, '.7f'))

# Display the adjusted p-values
p_values_decimal

# Data Visializations
def viz_insights(field):
    plt.figure(figsize = (15, 5))
    sns.histplot(data = df, x = field, hue = "quality", kde = True, palette = palette, bins = 20, multiple = "stack", alpha = .3)
    plt.legend(['bad wine', 'good wine'])
    if field == "density": plt.title(f"\n{field.capitalize()} of red wine\n\n")
    else: plt.title(f"\n{field} contents in red wine\n\n")
    plt.figtext(0.75, 0.3, f'{df[field].describe()}')
    plt.show()
    
viz_insights("fixed acidity")
viz_insights("volatile acidity")
viz_insights("citric acid")
viz_insights("residual sugar")
viz_insights("chlorides")
viz_insights("free sulfur dioxide")
viz_insights("total sulfur dioxide")
viz_insights("density")
viz_insights("pH")
viz_insights("sulphates")
viz_insights("alcohol")