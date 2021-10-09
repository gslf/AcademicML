import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from pegasus import Pegasus

if __name__ == "__main__":
    # Training parameters
    learningRate = 0.01
    epochs = 10000

    # Read data
    data = pd.read_csv('penguins_size.csv')

    # Features standardization
    data = data.dropna()
    data = data.drop(['sex', 'island', 'flipper_length_mm', 'body_mass_g'], axis=1)
    data = data[data['species'] != 'Chinstrap']

    X = data.drop(['species'], axis=1)

    ss = StandardScaler()
    X = ss.fit_transform(X) 

    y = data['species']
    species = {'Adelie': -1, 'Gentoo': 1}
    y = [species[item] for item in y]
    y = np.array(y) 

    X = np.delete(X, 182, axis=0)
    y = np.delete(y, 182, axis=0)

    # Split TrainSet and TestSet
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

    # Plot Data
    plt.figure(figsize=(11, 5))
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='orange', label='Adelie')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='gray', label='Gentoo')
    plt.show()

    # Train the model
    p = Pegasus(X_train.shape[1], learningRate, epochs)
    p.fit(X_train, y_train)

    # Test the model
    preditions = p.predict(X_test)
    print(accuracy_score(preditions, y_test))
