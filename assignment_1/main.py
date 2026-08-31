import numpy as np
np.random.seed(7)

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

np.random.shuffle(dataset)
splitratio = 0.8
split = int(len(dataset) * splitratio)

X_train = dataset[:split, 0:8]
X_val = dataset[split:, 0:8]

Y_train = dataset[:split, 8]
Y_val = dataset[split:, 8]

# Distance (Euclidean Distance function)
def euclidean_distance(one, two):
    return np.linalg.norm(one - two)

# Prediction
def knn_predict(training_data, training_labels, test_point, k):
    distances = []

    for i in range(len(training_data)):
        distance = euclidean_distance(test_point, training_data[i])
        distances.append((distance, training_labels[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    predicted_label = max(set(k_nearest_labels),key=k_nearest_labels.count)

    return predicted_label

# Evaluation
TP = TN = FP = FN = 0
predictions = []

for i in range(len(X_val)):
    x = X_val[i]
    y = Y_val[i]

    pred = knn_predict(X_train, Y_train, x, k=8)
    predictions.append(pred)

    if y == 1 and pred == 1:
        TP += 1
    elif y == 1 and pred == 0:
        FN += 1
    elif y == 0 and pred == 1:
        FP += 1
    elif y == 0 and pred == 0:
        TN += 1

accuracy = (TP + TN) / (TP + TN + FP + FN)
recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1_score = 2 * TP / (2 * TP + FP + FN)

predictions = np.array(predictions)
mse = np.mean((Y_val - predictions) ** 2)

print(f"Accuracy:  {accuracy:.2f} ({accuracy * 100:.2f}%)")
print(f"Recall:    {recall:.2f} ({recall * 100:.2f}%)")
print(f"Precision: {precision:.2f} ({precision * 100:.2f}%)")
print(f"F1 Score:  {f1_score:.2f} ({f1_score * 100:.2f}%)")
print(f"MSE:       {mse:.2f}")

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)