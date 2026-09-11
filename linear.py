# Multiple linear regression from scratch: predict rent from several features.
# READING GUIDE: This program learns rent from size, BHK, bathrooms, city and furnishing.
# A feature is an input column; a target is the known output (Rent).
# A row is one listing. Multiple linear regression means several inputs, one target.
# Prediction = bias + w1*x1 + w2*x2 + ...; training learns the weights and bias.
# Run from a terminal: python linear.py
# Python executes top to bottom. Indentation groups function/loop/conditional bodies.
# # starts a comment. = assigns; == compares. [] creates lists or selects elements.
# {} creates dictionaries here: mappings from keys to values. Strings are quoted text.
# object.method(...) calls a function belonging to an object; name=value inside a
# function call supplies a named argument. Parentheses also group expressions.
# 'from ... import ...' loads a particular object. Path handles filesystem paths.
from pathlib import Path

# Imports load libraries; 'as' supplies short aliases. NumPy (np) handles numerical
# arrays, pandas (pd) handles tables, and matplotlib.pyplot (plt) draws plots.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path.home() returns your user folder. / joins path components when used with a
# Path object (rather than dividing numbers). Change this path if you move the CSV.
dataset_path = Path.home() / 'Downloads' / 'archive' / 'House_Rent_Dataset.csv'
# CSV means comma-separated values. read_csv reads the file into a DataFrame:
# a table with named columns and indexed rows. It infers types from the file.
# A single DataFrame column is a Series. The original file is not modified.
data = pd.read_csv(dataset_path)

# Keep the same reproducible 80/20 split as the size-only model.
# Training data teaches the weights; test data checks performance on other rows.
# default_rng creates a random-number generator. Seed 42 makes its sequence
# reproducible; the number 42 itself has no special significance.
rng = np.random.default_rng(42)
# len(data) counts rows. permutation shuffles every integer row position once.
# Shuffling positions keeps each listing paired with its rent and other features.
indices = rng.permutation(len(data))
# 80% goes to training. int discards the fractional part, making a row count.
split_index = int(0.8 * len(data))
# iloc selects by integer POSITION, not index label. [:split_index] means from
# the start up to, but NOT including, split_index. [split_index:] means the rest.
# copy() makes a separate table so later edits do not change the source table.
train_rows = data.iloc[indices[:split_index]].copy()
test_rows = data.iloc[indices[split_index:]].copy()
# These lists contain column names. Numeric features are quantities. BHK is the
# bedroom/hall/kitchen designation (such as 2 BHK); Bathroom is a bathroom count.
# Categorical features are labels: Mumbai is not numerically greater than Delhi.
numeric_features = ['Size', 'BHK', 'Bathroom']
categorical_features = ['City', 'Furnishing Status']

# Learn categories from training data only. One-hot encoding makes 0/1 columns.
# Omit the first category of each feature as the reference category.
# A dictionary comprehension builds one entry for each categorical column.
# train_rows[name] selects that column. dropna() excludes missing values only
# while finding categories; it does NOT remove missing rows from the dataset.
# unique() finds distinct labels; sorted() puts them in a reproducible order.
# Only TRAINING rows decide the categories, avoiding information leakage from test.
categories = {
    name: sorted(train_rows[name].dropna().unique())
    for name in categorical_features
}
# + concatenates lists. This nested list comprehension loops over columns then
# their categories. [1:] skips the first category; it becomes the reference.
# f'{name}={category}' is an f-string: braces insert values into text.
# One-hot encoding represents a category with 0/1 indicator ('dummy') columns.
# For example City=Mumbai is 1 for Mumbai and 0 otherwise.
# Dropping one indicator avoids redundant columns with the bias: the reference
# category is represented by zeros in all indicators for that categorical feature.
feature_names = numeric_features + [
    f'{name}={category}'
    for name in categorical_features
    for category in categories[name][1:]
]


# def defines reusable instructions; data_rows is the function's input parameter.
# The indented body runs only when this function is called. It accepts a DataFrame
# or a list of row dictionaries, allowing identical processing of new houses.
def encode_features(data_rows):
    # DataFrame converts the input into a table. Selecting the combined list of
    # columns keeps only model inputs; copy() prevents modifying the original data.
    frame = pd.DataFrame(data_rows)[numeric_features + categorical_features].copy()
    # apply calls pd.to_numeric on each selected column. Numeric text such as '1800'
    # becomes a number. errors='raise' stops on invalid text instead of hiding it.
    # Assignment back to frame replaces these columns with the converted values.
    frame[numeric_features] = frame[numeric_features].apply(pd.to_numeric, errors='raise')
    # for repeats its indented body once per categorical feature name.
    for name in categorical_features:
        # isin checks each value against allowed labels and returns booleans.
        # all() is True only if every row passes; 'not' reverses that answer.
        # Unknown/missing categories cause an error rather than being mislabeled.
        if not frame[name].isin(categories[name]).all():
            # raise stops execution with an explanatory exception (an error object).
            raise ValueError(f'Unknown or missing category in {name}.')
        # Fixed training categories keep dummy columns consistent for new data.
        # Categorical fixes allowed labels and their order to the TRAINING categories.
        # Even a one-house input will therefore produce the same encoded columns.
        frame[name] = pd.Categorical(frame[name], categories=categories[name])
    # get_dummies replaces categorical columns with numeric indicator columns.
    # columns specifies what to encode; drop_first=True omits each reference label.
    # prefix_sep='=' names columns like City=Mumbai. dtype=float uses 0.0/1.0 values.
    # float is a numerical type supporting decimal values.
    encoded = pd.get_dummies(
        frame, columns=categorical_features, drop_first=True,
        prefix_sep='=', dtype=float,
    )
    # Convert the prepared DataFrame to NumPy for gradient descent.
    # Select feature_names to enforce the same column order for every input.
    # to_numpy drops table labels and returns a matrix of shape (rows, features).
    result = encoded[feature_names].to_numpy(dtype=float)
    # isfinite rejects NaN (missing/undefined numbers) and infinity. all() checks
    # every entry. Numeric missing values are rejected, not automatically filled.
    if not np.isfinite(result).all():
        raise ValueError('Features must contain finite numbers.')
    # return sends this matrix back to the caller; execution continues there.
    return result


# Call the same encoding function for both partitions; learned categories remain
# fixed. X is a 2D feature matrix, with one row per house and one column per input.
x_train = encode_features(train_rows)
x_test = encode_features(test_rows)
# Select Rent separately so the input cannot reveal the answer. to_numpy(float)
# produces a 1D numerical target array, aligned with the feature matrix's rows.
y_train = train_rows['Rent'].to_numpy(dtype=float)
y_test = test_rows['Rent'].to_numpy(dtype=float)
# 'and' requires both target arrays to pass the finite-number check.
if not (np.isfinite(y_train).all() and np.isfinite(y_test).all()):
    raise ValueError('Rent must contain finite numbers.')

# Scale each column separately using training statistics only.
# Standardization: (value - training mean) / training standard deviation.
# Mean is the average; standard deviation measures spread around that average.
# Example: (1200 - 1000)/200 = 1, one standard deviation above the mean.
# axis=0 aggregates down rows, giving a separate result for each feature column.
# Scaling helps gradient descent handle inputs with very different numerical scales.
x_mean = np.mean(x_train, axis=0)
# NumPy's std uses population standard deviation by default (ddof=0).
s_std = np.std(x_train, axis=0)
# Boolean indexing selects zero standard deviations. Replace them with 1 to avoid
# division by zero; constant training columns become zero after centering.
s_std[s_std == 0] = 1.0
# Broadcasting applies the per-column mean and std to every row automatically.
# Test data must use TRAINING statistics too, not its own mean and std.
x_train_scaled = (x_train - x_mean) / s_std
x_test_scaled = (x_test - x_mean) / s_std
# print writes to the terminal without changing the model. len counts rows.
print('Training rows:', len(x_train))
print('Test rows:', len(x_test))
print('Encoded features:', feature_names)
# items() yields dictionary key/value pairs. values[0] selects the first label.
# This comprehension creates a small dictionary showing reference categories.
print('Reference categories:', {name: values[0] for name, values in categories.items()})

# Each feature now gets its own weight. @ means matrix multiplication:
# prediction = w1*x1 + w2*x2 + ... + bias.
# shape describes array dimensions; shape[1] is the number of feature columns.
# zeros initializes one weight per feature. Weights and bias are PARAMETERS:
# the model learns them. The bias is the additive intercept/baseline.
w = np.zeros(x_train.shape[1])
b = 0.0
# Learning rate and epoch count are HYPERPARAMETERS: settings chosen for training.
# Learning rate controls step size. Too large may diverge; too small learns slowly.
learning_rate = 0.01
# An epoch is one pass over the training data. This is FULL-BATCH training:
# all rows contribute to one update per epoch. Mini-batch uses smaller groups;
# stochastic gradient descent uses one row per update. These change optimization,
# not the relationships the underlying linear model is capable of representing.
epochs = 1000
# [] creates an empty list. We will append training losses for the second chart.
cost_history = []
# m counts training examples so the gradient formulas can average over them.
m = len(x_train_scaled)

# range(epochs) produces 0,1,...,epochs-1. This loop repeats training updates.
for epoch in range(epochs):
    # @ is matrix multiplication: (m, features) @ (features,) gives m predictions.
    # Each prediction sums that row's weighted features. Adding b broadcasts one
    # shared bias to all rows. y_pred is often written y-hat in mathematics.
    y_pred = x_train_scaled @ w + b
    # Positive error means overprediction; negative means underprediction.
    error = y_pred - y_train
    # ** means exponentiation. MSE = mean((prediction - actual)**2).
    # Squaring prevents positive/negative errors cancelling. Twice the error costs
    # four times as much squared loss. mean averages across houses.
    # LOSS is the numerical objective training minimizes; it tells us which weights
    # fit better. MSE uses squared rent units, so a large number alone is not a bug.
    mse = np.mean(error ** 2)
    # append stores the next loss value at the list's end, BEFORE this epoch's update.
    cost_history.append(mse)

    # X.T computes one gradient for each feature's weight.
    # A gradient is the derivative (slope) of loss with respect to a parameter:
    # it describes loss change for a small parameter change. dw contains one per weight.
    # .T transposes X (swaps rows/columns). X.T @ error sums x_j*error for each feature.
    # The factor 2 comes from differentiating a square; dividing by m averages.
    dw = (2 / m) * (x_train_scaled.T @ error)
    # Bias has an implicit input of 1 for every row, so db = 2*mean(error).
    # np.sum adds array values. Both gradients use the same old weights and bias.
    db = (2 / m) * np.sum(error)
    # Subtract the gradient to move DOWNHILL in loss; multiply by learning rate
    # to control the step size. The next line applies the same rule to the bias.
    w = w - learning_rate * dw
    b = b - learning_rate * db
    # % is remainder: print at 0, 200, 400, ... rather than every epoch.
    # An f-string interpolates values; :.2f displays two decimal places without
    # rounding the actual stored numbers used in the model.
    if epoch % 200 == 0:
        print(f'Epoch={epoch}, training MSE={mse:.2f}')

# \n begins a new printed line. These weights correspond to STANDARDIZED inputs.
# Each is the effect of a one-standard-deviation feature change with others fixed.
# Dummy columns were standardized too: weights are not raw city rent differences.
# Coefficients describe fitted associations, not proof of cause or importance.
print('\nLearned weights (for standardized features):')
# zip pairs each feature name with its corresponding weight. The loop unpacks
# each pair into name and weight, then prints it.
for name, weight in zip(feature_names, w):
    print(f'  {name}: {weight:.2f}')
print(f'Bias: {b:.2f}')

# Evaluate final parameters on TEST data; there are no weight updates here.
y_test_pred = x_test_scaled @ w + b
# abs removes the error sign. MAE = average absolute prediction error, in rent
# units. It answers how far predictions are from actual rents on average.
mae = np.mean(np.abs(y_test_pred - y_test))
# This is TEST MSE, overwriting the variable used for training MSE. Stored
# cost_history still contains training losses. Test MSE is not used to learn weights.
mse = np.mean((y_test_pred - y_test) ** 2)
# sqrt is square root. RMSE restores rent units but still emphasizes big errors.
# A model can have better MAE and worse RMSE if it makes a few bigger mistakes.
rmse = np.sqrt(mse)
# Residual means actual minus predicted. ss_res sums squared residuals.
ss_res = np.sum((y_test - y_test_pred) ** 2)
# ss_total sums squared deviations from the TEST mean. It is the error from
# predicting the evaluation set's average for every house, used as the R2 baseline.
# This mean is used for evaluation, not fed back into training.
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
# R2 = 1 - model_squared_error / mean_baseline_squared_error.
# 1 means perfect; 0 matches the mean baseline; negative is worse than that baseline.
# R2 is NOT percentage accuracy. 0.48 means about 48% less squared error than
# the mean baseline on these rows. 'a if condition else b' chooses a value.
# If all actual rents are equal, the denominator is zero; report NaN (undefined).
r2 = 1 - ss_res / ss_total if ss_total > 0 else float('nan')
print('\nMODEL PERFORMANCE (rent in original dataset units)')
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
# :.4f formats four decimal places.
print(f'R2: {r2:.4f}')
# The comparison produces booleans; sum counts True as 1 and False as 0.
# Ordinary linear regression has no constraint preventing negative rent estimates.
print('Negative test predictions:', np.sum(y_test_pred < 0))
print('\nFirst 10 test predictions:')
# head(10) selects ten rows; to_dict('records') converts them to a list of
# dictionaries. [:10] selects matching array entries. zip traverses all three
# sequences together and unpacks the entries into row, actual and predicted.
for row, actual, predicted in zip(test_rows.head(10).to_dict('records'), y_test[:10], y_test_pred[:10]):
    # row['Size'] retrieves a dictionary value. Adjacent strings inside parentheses
    # are joined into one message, allowing this print statement to span two lines.
    print(f"Size={row['Size']}, City={row['City']}, "
          f'Actual={actual:.2f}, Predicted={predicted:.2f}')

# Edit these example details to predict rent for a different house.
# This dictionary supplies ALL features for one illustrative prediction.
# Use known category spellings. There is no actual target rent here to score.
new_house = {
    'Size': 1800,
    'BHK': 3,
    'Bathroom': 2,
    'City': 'Kolkata',
    'Furnishing Status': 'Semi-Furnished',
}
# [new_house] wraps the dictionary in a list, representing one table row.
# Reuse the SAME encoding, feature order, and training statistics as before.
new_house_scaled = (encode_features([new_house]) - x_mean) / s_std
# The calculation returns a one-element array; [0] extracts its scalar value.
predicted_rent = (new_house_scaled @ w + b)[0]
print('\nExample house:', new_house)
print(f'Predicted rent: {predicted_rent:.2f} (original dataset units)')

# Multiple inputs cannot be represented by a single size-versus-rent line.
# Perfect predictions lie on the diagonal in this actual-vs-predicted plot.
# figure() creates a separate canvas so charts do not overlap.
plt.figure()
# CHART 1: each dot represents one TEST listing. Horizontal position is actual
# rent, vertical is predicted rent. Above the diagonal = overprediction; below
# = underprediction. Vertical distance to the diagonal is the error magnitude.
# Below y=0 is a negative rent prediction. Points near the line are more accurate.
# scatter draws dots; alpha=0.4 means partly transparent, revealing overlap.
# label names this plotted series for the legend. This alpha is not Ridge alpha.
plt.scatter(y_test, y_test_pred, alpha=0.4, label='Test listings')
# Array min/max find extremes; Python min/max combine the actual and predicted
# ranges. The diagonal will span both sets, including any negative estimates.
lower = min(y_test.min(), y_test_pred.min())
upper = max(y_test.max(), y_test_pred.max())
# plot connects supplied (x,y) coordinates. Identical x/y endpoints give y=x.
# 'r--' means a red dashed line: the perfect-prediction reference.
plt.plot([lower, upper], [lower, upper], 'r--', label='Perfect prediction')
# xlabel/ylabel name axes; title supplies the chart heading. These change
# presentation only; rent remains in the original dataset's numerical units.
plt.xlabel('Actual rent (original dataset units)')
plt.ylabel('Predicted rent (original dataset units)')
plt.title('Multiple Linear Regression: Actual vs Predicted Rent')
# legend displays the labels attached to the dots and reference line.
plt.legend()
# tight_layout adjusts spacing so axis labels and titles fit.
plt.tight_layout()

# figure() creates a separate canvas so charts do not overlap.
plt.figure()
# CHART 2: TRAINING MSE versus epoch. With only y-values supplied, plot uses
# indices 0,1,2,... as x-values, matching the epoch numbers in this program.
# Falling loss means improved training fit. A plateau means little further gain,
# not necessarily accurate predictions. Rising/oscillating loss may mean a
# learning rate that is too large. More epochs alone cannot fix an unsuitable model.
# This chart alone cannot diagnose overfitting: compare validation loss too.
# Overfitting = learning training-specific details that fail on new data.
# Underfitting = a model too restrictive to capture useful relationships.
# The curve records loss before each update, not after the final weight update.
plt.plot(cost_history)
plt.xlabel('Epoch')
plt.ylabel('Training Mean Squared Error')
plt.title('Gradient Descent: Cost vs Epoch')
# tight_layout adjusts spacing so axis labels and titles fit.
plt.tight_layout()
# show displays both figures; desktop Python typically waits until you close
# the windows. It does not save images. Figure 1 evaluates predictions; figure 2
# shows optimization progress. Good training progress is not proof of good test fit.
plt.show()
