import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('data_classification.csv', header=None)
true_x = []
true_y = []
false_x = []
false_y = []
for item in data.values:
    if item[2] == 1:
        true_x.append(item[0])
        true_y.append(item[1])
    else: 
        false_x.append(item[0])
        false_y.append(item[1])
plt.scatter(true_x, true_y, marker='o', c='b')
plt.scatter(false_x, false_y, marker='s', c='r')
plt.xlabel("Số giờ học")
plt.ylabel("Số giờ ngủ")
#plt.show()
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def phan_chia(p):
    if p>=0.5:
        return 1
    else:
        return 0

def predict(features, weights):
    z = np.dot(features, weights)
    return sigmoid(z)

def cost_function(features, labels, weights):
    n = len(labels)
    predictions = predict(features, weights)
    cost_class1 = -labels * np.log(predictions)
    cost_class2 = -(1 - labels) * np.log(1 - predictions)
    cost = cost_class1 + cost_class2
    return cost.sum() / n
# Dung gradient descent de cap nhat weight
def update_weight(features, labels, weights, learning_rate):
    n = len(labels)
    predictions = predict(features, weights)
    gd = np.dot(features.T, (predictions - labels))
    gd = gd / n
    weights = weights - gd * learning_rate
    return weights

def train(features, labels, weights, learning_rate, iter):
    cost_hs = []
    alpha = 0.00001  
    beta = 0.5  
    for i in range(iter):
        weights_old = weights.copy()  # Luu trong so cu
        weights = update_weight(features, labels, weights, learning_rate)
        cost = cost_function(features, labels, weights)
        cost_old = cost_function(features, labels, weights_old)
        while cost > cost_old - alpha * np.dot((weights - weights_old).T, (weights - weights_old)):
            learning_rate *= beta  # Giam learning rate
            weights = update_weight(features, labels, weights_old, learning_rate)
            cost = cost_function(features, labels, weights)
        cost_hs.append(cost)
    return weights, cost_hs

# Chuyen du lieu thanh mang numpy
features = np.array(data.values[:, :2])
print("fe")
print(features)
labels = np.array(data.values[:, 2])
#Chia tap du lieu thanh tap train va tap test
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
# Them cot bias vao tap train_features
train_features = np.insert(train_features, 0, 1, axis=1)
# Khoi tao trong so
weights = np.zeros(train_features.shape[1])
# Thiet lap tham so huan luyen
learning_rate = 1
iterations = 20000

# Huan luyen mo hinh
trained_weights, cost_history = train(train_features, train_labels, weights, learning_rate, iterations)
print(cost_history)
# Ve duong du doan
x_values = np.array([np.min(features[:, 1]), np.max(features[:, 1])])
y_values = -(trained_weights[0] + trained_weights[1]*x_values) / trained_weights[2]
plt.plot(x_values, y_values, label='Decision Boundary')
plt.legend()
plt.show()
# Truc quan hoa qua trinh huan luyen
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()
#du doan 1 gia tri
new_feature = np.array([0.9, 9.45])
new_feature_with_bias = np.insert(new_feature, 0, 1)
prediction = predict(new_feature_with_bias, trained_weights)
print(prediction)
# du doan nhan cho tap test
test_predictions = [phan_chia(predict(np.insert(feature, 0, 1), trained_weights)) for feature in test_features]

# Tinh do chinh xac tren tap test
accuracy = accuracy_score(test_labels, test_predictions)
print("Accuracy on test set:", accuracy)
