# Imported Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

