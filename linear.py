# this is the linear regression model 1 from scratch
# here for the liner regression we will use a very small dataset to train the model and then we will test it (in 4:1 ratio respectively)
#  x is  the input feature and y is the output label and we will satrt with custom data no pandas for data handling


# data
import numpy as np
import matplotlib.pyplot as plt 
x=np.array([120,  150,  170, 180,  210,440,530,1234,3000, 5000

],dtype=float)
y=np.array([32,54,24,11,65,89,43,55,23,45
            ],dtype=float)
# x is the house size and y is the price of the house in thousands
# now  we  spit the data for test and train feat  and to make in configurable we use a variable

split_index=8
x_train=x[:split_index]
y_train=y[:split_index]
x_test=x[split_index:]
y_test=y[split_index:]

print("the training data is of length=",len(x_train))
print("the test data is of length=",len(x_test))

# now we scale the features as the amount are big in number so me scale it down using standard deviation and mean
x_mean=np.mean(x_train)
s_std=np.std(x_train)

#we got the dividing factor but we have to store  the new values too

x_train_scaled=(x_train-x_mean)/s_std
x_test_scaled=(x_test-x_mean)/s_std

print("the old training data is ",x_train)
print("the scaled training data is=",x_train_scaled)

# now we move tho acctual linear regression model
# here we will use the y=mx +c 
# y_hat=w*x+b
# y_hat is the predicted value
# w is the weight  , it tell us what is the relevance of that feature for the price  , loation can change the price and similarly  the number of the  room s 
# x is the features  like the size no of room etc
# now b is the bias 

# in the  start the model knows nothing so w=0,b=0
w=0.0
b=0.0

#  npw lets  move to hyperparammeter -which are changablw and deduced mathematicalyy
# such as batch size , number of hidden layer 
learning_rate=0.01
epochs=1000

cost_history=[]

# now we need to know what this cost history and cost function is 
# the cost function is the difference between the predicted value and actual value and we will use the mean squared error as the cost function
# basically tell how bad the model predictions are 
# we use mean squared error =  1/n(prediction - actual)^2

# and there is gradient descent which is the optimization algorithm to minimize the cost function and it is used to update the weights and bias of the model
# simply change the w and b 
# gradient tell which direction is up and we move  opposite to that
# as we need to reach the lowest point where the cost is min , not local min but the global min of the  cost

m=len(x_train_scaled)

for epoch in range(epochs):
    y_pred=w*x_train_scaled + b
    error =y_pred-y_train
    mse=np.mean(error**2)
    cost_history.append (mse)

    # now commes gradient  they tell us how w(weight) and b(bias) shou;d change
    # dw= gradient of the cost w.r.t w and similarly db for bias
    dw=(2/m)*np.sum(error*x_train_scaled)
    db=(2/m)*np.sum(error)
    # we have calculated the changes that we need to make in w and b now we update them

    w=w-learning_rate*dw
    b=b-learning_rate*db

    # print hte trainig progress

    if epoch%200==0:
        print(f"epoch={epoch},mse={mse},w={w},b={b}")

print ("training is completed")

print("learned weight:", w)
print("learned bias:", b)

y_test_pred=w*x_test_scaled + b
print("predictions:")
for size , actual,predicted in zip(
    x_test, y_test, y_test_pred
):
    print(f" Size: {size}, Actual: {actual}, Predicted: {predicted}")
mae=np.mean(np.abs(y_test_pred -y_test))
mse =np.mean((y_test_pred -y_test)**2)
rmse=np.sqrt(mse)
# Usually:
#
# R² = 1
# Excellent fit
#
# R² = 0
# Model is basically no better than predicting the mean.
#
# R² can also be negative if the model is very poor.

ss_res = np.sum((y_test - y_test_pred) ** 2)

ss_total = np.sum(
    (y_test - np.mean(y_test)) ** 2
)

r2 = 1 - (ss_res / ss_total)


print("\nMODEL PERFORMANCE")
print("--------------------------")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)
new_house_size = 1800


# IMPORTANT:
# The model was trained using SCALED values.
#
# Therefore new data must be transformed using
# exactly the SAME mean and std.

new_house_scaled = (
    new_house_size - x_mean
) / s_std


predicted_price = (
    w * new_house_scaled + b
)


print(
    f"\nPredicted price for "
    f"{new_house_size} sqft house: "
    f"{predicted_price:.2f} lakh"
)


# ------------------------------------------------------------
# 12. VISUALIZE THE REGRESSION LINE
# ------------------------------------------------------------
#
# Visualization helps us understand what linear regression
# actually learned.
#
# Dots = real data
# Line = model prediction


# Create many X values so our line looks smooth.

X_line = np.linspace(
    min(x),
    max(x),
    100
)


# Scale them using our training statistics.

X_line_scaled = (
    X_line - x_mean
) / s_std


# Predict corresponding prices.

y_line = (
    w * X_line_scaled + b
)


plt.scatter(
    x,
    y,
    label="Actual House Prices"
)

plt.plot(
    X_line,
    y_line,
    label="Linear Regression"
)

plt.xlabel("House Size (sqft)")
plt.ylabel("House Price (Lakhs)")

plt.title(
    "House Price Prediction Using Linear Regression"
)

plt.legend()

plt.show()


# ------------------------------------------------------------
# 13. VISUALIZE COST DURING TRAINING
# ------------------------------------------------------------
#
# If gradient descent is working correctly,
# cost should decrease.
#
# Ideally we should see something like:
#
# 1000
# 500
# 200
# 50
# 10
# 2
# ...
#
# Then it starts flattening.

plt.plot(cost_history)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")

plt.title(
    "Gradient Descent: Cost vs Epoch"
)

plt.show()
