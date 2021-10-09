import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def prediction(x, w, b):
    score = np.dot(x, w) + b
    return np.where(score >= 0, 1, -1)

def modelFit(learningRate, epochs, x, y):
    b = 0.0
    w = np.zeros(x.shape[1])

    for _ in range(epochs):
        for xi, yi in zip(x, y):
            # Update the value
            update = learningRate * (yi - prediction(xi, w, b))
            b += update
            w += update * xi

    return b, w


if __name__ == "__main__":
    # Load dataset
    print("Download dataset . . .")
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df.head()

    # Extract labels
    y = df.iloc[0:100, 4].values

    # Extract features
    x = df.iloc[0:100, 0:2].values

    # Plot data
    plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
    plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x',label='Versicolour')

    plt.show()

    # Features standardization
    y = np.where(y == 'Iris-setosa', 1, -1)
    x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
    x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

    # Split TrainSet from TestSet
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # Train the model and show results
    epochs = 10
    learningRate = 0.01
    b, weights = modelFit(learningRate,epochs,x_train, y_train)

    # Test the model
    errors = 0
    for i in range(len(x_test)):
        print("--------------------------------------------------------")
        value = str(y_test[i])
        predicted_value = str(prediction(x_test[i], weights, b))
        print("Real Value: " + value)
        print("Predicted Value: " + predicted_value)
        if value != predicted_value:
            errors += 1
        print("--------------------------------------------------------")

    print("Prediction errors: {}".format(errors))
