# Imported Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

"""Stage 1: Linear regression with one independent variable

Description

In the first stage, let's start with the simplest linear model — it will include 
salary as a dependent variable and the player's rating as the only predictor.

Objectives

The linear approximation that shows the relationship between rating and salary 
is salary = k * rating + b where k is the slope of the linear regression model 
and b is its intercept.

The goal is to fit such a model to the train data, after splitting it to a train 
(70%) and test sets, find the coefficient values k and b, predict the salary 
with the fitted model on the test data, and calculate the MAPE (Mean Average 
Percentage Error).

"""

data = pd.read_csv('Data/data.csv')

# Extracting the predictor and target data sets and splitting then into train
# and test sets
X, y = data[['rating']], data['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=100)

# Fitting the train data set to a Linear Regression model with an intercept
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the salary from the test set with the fitted model then
# calculating MAPE
prediction = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, prediction)

# Printing the coefficient, intercept, and MAPE value rounded to 5 decimal
# places
slope = round(model.coef_[0], 5)
intercept = round(model.intercept_, 5)

print("The slope, intercept, and MAPE for the linear regression model \
with one independent variable are:")
print(slope, intercept, round(mape, 5), end='\n\n')

"""Stage 2: Linear regression with predictor transformation

Description

When drawing a scatterplot of rating vs salary, the relationship between these
two variables seems to be different from linear and looks like a polynomial 
function. Let's try to raise the rating by several degrees and see whether it 
improves the score.

Objectives

Make the same steps as in Stage 1 (split the data to train and test sets and 
fit to a LinearRegression model then calculate MAPE), but do it 3 times, for 
each time raising the predictor to the power of 2, 3, then 4. Print the best 
MAPE obtained.

"""

# Create a function that will output the mape score given the power to which we
# want to raise the predictor


def predictor_transform(power: int) -> float:
    """Return the MAPE score after fitting <data> to a Linear Regression model
    raising the power of the predictor <data[['rating]] by <power>."""

    # Extracting the predictor and target data sets and splitting then into
    # train and test sets
    X, y = data[['rating']] ** power, data['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=100)

    # Fitting the train data set to a Linear Regression model with an intercept
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicting the salary from the test set with the fitted model then
    # calculating MAPE
    prediction = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, prediction)

    return round(mape, 5)


required_powers = [2, 3, 4]
mape_scores = [predictor_transform(power) for power in required_powers]

print(f'The minimum MAPE score is {min(mape_scores)} when testing raising the \
power of the predictor by 2, 3, or 4. This MAPE score was obtained when the \
power of the predictor was raised to \
{required_powers[mape_scores.index(min(mape_scores))]}', end='\n\n')

"""Stage 3: Linear regression with many independent variables

Description

In the previous stages, you used only one independent variable. Now, your task 
is to include other variables into a model.

Objectives

Do the same as in Stage 1 (fit the data to a Linear Regression model), but 
include all numeric columns (not just <rating>) other than the target for the 
predictors.
Print the model coefficients.

"""

# Extracting the predictor and target data sets and splitting then into train
# and test sets
X, y = data[['rating', 'draft_round', 'age', 'experience', 'bmi']], \
    data['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=100)

# Fitting the train data set to a Linear Regression model with an intercept
model = LinearRegression()
model.fit(X_train, y_train)

# Determining the coefficients of the model
model_coefficients = list(model.coef_)
print("The coefficients for the Linear Regression model with multiple \
independent variables included (5 total) are:")
print(', '.join([str(num) for num in model_coefficients]), end='\n\n')

"""Stage 4: Test for multi-collinearity and variables selection

Description

If you have a linear regression with many variables, some of them may be 
correlated. This way, the performance of the model may decrease. A crucial 
step is to check the model for multi-collinearity and exclude the variables 
with a strong correlation with other variables.

Objectives

1 - Calculate the correlation matrix for the numeric variables
2 - Find the variables where the correlation coefficient is greater than 0.2
3 - Split the predictors and the target into training and test sets. Use 
test_size=0.3
4 - Fit the linear models for salary prediction based on the subsets of other 
variables. The subsets are as follows:
    - First, try to remove each of the variables found in step 2
    - Second, remove each possible pair of these variables
5 - Make predictions and print the lowest MAPE rounded to 5 decimal places

"""

# 1 - Calculate the correlation matrix for the numeric variables
numeric_data = data[['rating', 'draft_round', 'age', 'experience', 'bmi']]
corr = numeric_data.corr()

# 2 - Find the variables where the correlation coefficient is greater than 0.2
correlated_features = []
for index, row in corr.iterrows():
    if row.between(0.2, 1, inclusive='left').any():
        correlated_features.append(index)

# 3 - Split the predictors and the target into training and test sets. Use
# test_size=0.3
X, y = numeric_data, data['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=100)

# Creating two parallel lists that will store all MAPE scores and the
# corresponding feature(s) that were deleted
all_mape_scores = []
deleted_features = []

# Removing only one of the items in <correlated_features>
for feature in correlated_features:
    # Remove the <feature> column from the train and test sets
    X_train_f, X_test_f = X_train.loc[:,
                          X_train.columns != feature], \
                          X_test.loc[:, X_test.columns != feature]

    # Fitting the train data set to a Linear Regression model
    model = LinearRegression()
    model.fit(X_train_f, y_train)

    # Predicting the salary from the test set with the fitted model then
    # calculating MAPE
    prediction = model.predict(X_test_f)
    mape = mean_absolute_percentage_error(y_test, prediction)

    all_mape_scores.append(round(mape, 5))
    deleted_features.append([feature])

# Removing a pair of the features (keeping only one of the
# <correlated_features> at a time)
i = 0
for feature in correlated_features:
    excluded = correlated_features.pop(i)

    # Remove other features that aren't currently assigned to <feature>
    X_train_f, X_test_f = X_train.loc[:, ~X_train.columns.isin(
        correlated_features)], X_test.loc[:,
                               ~X_test.columns.isin(correlated_features)]

    # Fitting the train data set to a Linear Regression model
    model = LinearRegression()
    model.fit(X_train_f, y_train)

    # Predicting the salary from the test set with the fitted model then
    # calculating MAPE
    prediction = model.predict(X_test_f)
    mape = mean_absolute_percentage_error(y_test, prediction)

    all_mape_scores.append(round(mape, 5))
    deleted_features.append(correlated_features[:])

    # Insert back <feature> currently saved in <excluded> to
    # <correlated_features>
    correlated_features.insert(i, excluded)

    i = i + 1

print("The minimum MAPE score obtained after excluding variables with a strong \
correlation with other variables is:")
print(min(all_mape_scores))

"""Stage 5: Deal with negative predictions

Description

A linear model may predict negative values. However, such values can be 
meaningless because the salary can't be negative. In this stage, handle 
negative predictions.

Objectives

1 - Choose the predictors that gave the best MAPE metric in Stage 4.
2 - Split predictors and the target into train and test parts. Use test_size=0.3
3 - Fit the model
4 - Predict the salaries
5 - Try two techniques to deal with negative predictions:
    - replace the negative values with 0
    - replace the negative values with the median of the training part of y
6 - Calculate the MAPE for every two options and print the best as a floating 
number rounded to five decimal places.

"""

# Determining which feature(s) were deleted for the best MAPE socre obtained
# in Stage 4
min_mape_features = deleted_features[all_mape_scores.index(min(all_mape_scores))]
print(f'The excluded feature(s) were {", ".join(min_mape_features)}',
      end='\n\n')

# Remove the <min_mape_features> column(s) from the train and test sets
X_train, X_test = X_train.loc[:, ~X_train.columns.isin(min_mape_features)], \
    X_test.loc[:, ~X_test.columns.isin(min_mape_features)]

# Fitting the train data set to a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the salary from the test set with the fitted model
prediction = model.predict(X_test)
mape_of_techniques = []
techniques = []

# Technique one to deal with negative predictions: replace the negative values
# with 0
tech_1_predictions = prediction.copy()
tech_1_predictions[tech_1_predictions < 0] = 0

mape = mean_absolute_percentage_error(y_test, tech_1_predictions)
mape_of_techniques.append(round(mape, 5))
techniques.append('negative values were replaced with 0')

# Technique two to deal with negative predictions: replace the negative values
# with the median of the training part of y
tech_2_predictions = prediction.copy()
tech_2_predictions[tech_2_predictions < 0] = np.median(y_train)

mape = mean_absolute_percentage_error(y_test, tech_2_predictions)
mape_of_techniques.append(round(mape, 5))
techniques.append('negative values were replaced with the median of the \
training part of y')

print(f'The minimum MAPE score, {min(mape_of_techniques)}, was obtained when \
{techniques[mape_of_techniques.index(min(mape_of_techniques))]}')
