# Salary Prediction

### About
Linear regression is one of the simplest yet powerful tools for finding 
regularities in data and using them for prediction. It is widely applied both 
in science and practice. In this project, you will learn how to apply 
scikit-learn library to fit linear models, use them for prediction, compare 
the models, and select the best one. You will also learn how to carry out 
testing for certain issues with data.

### Learning Outcomes of the Project:
Practice fitting linear models with scikit-learn to predict values on the 
unknown data. Apply polynomial feature engineering, test your data for 
multi-collinearity, and evaluate models with the MAPE score.

### Learning Outcomes of Each Stage of the Project:
_Stage 1_ : Fit a simple model with one predictor and evaluate it.

_Stage 2_ : Use the linear model to handle the polynomial relationship between independent and dependent variables.

_Stage 3_ : Fit a linear model with many independent variables and compare it with the previous models.

_Stage 4_ : Check whether the variables have a high correlation and try to improve the model's performance by removing them.

_Stage 5_ : Get rid of negative predictions and see whether the model performance improves.

# General Info

To learn more about this project, please visit 
[HyperSkill Website - Salary Prediction](https://hyperskill.org/projects/287).

This project's difficulty has been labelled as __Challenging__ where this is how 
HyperSkill describes each of its four available difficulty levels:

- __Easy Projects__ - if you're just starting
- __Medium Projects__ - to build upon the basics
- __Hard Projects__ - to practice all the basic concepts and learn new ones
- __Challenging Projects__ - to perfect your knowledge with challenging tasks

This Repository contains one .py file and one folder:

    code.py - Contains the code used to complete the data analysis requirements

    Data repository - Contains the data.csv files that contain the data

Project was built using python version 3.11.3

# Description of Data Set

It contains the following 9 columns:

- Numerical features
  - `rating`
  - `draft_round`
  - `age`
  - `experience`
  - `bmi`


- Non-numerical features
  - `team`
  - `position`
  - `country`


- Target prediction
  - `salary`

# How to Run

Download the files to your local repository and open the project in your choice 
IDE and run the project. Different models were used to fit the data, evaluated 
with the Mean Absolute Percentage Error (MAPE) metric, according to each Stage's 
docstrings. Please read each Stage's docstring to know the requirements.
