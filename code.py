# Imported Packages
import pandas as pd
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

