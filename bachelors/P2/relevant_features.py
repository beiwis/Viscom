from matplotlib import pyplot as plt
import numpy as np
import functions as f
import pandas as pd
from sklearn import datasets

# -- Load Iris dataset.
####################################################
iris = datasets.load_iris()
x = iris.data
feature_names = iris.feature_names
y = iris.target
target_names = iris.target_names
# print(target_names)
labels = list(range(len(target_names))) # List of labels [0, 1, 2]

# -- Create a DataFrame with it.
####################################################
data = dict(
    [(n, x[:,i]) for i, n in enumerate(feature_names)] +
    [("flower name", target_names[y]), ("label", y)]
  )
df_iris = pd.DataFrame(data)
# print('\n\ndataset: \n', df_iris)
print(df_iris)

df_iris_shuffle = f.shuffle_data(df_iris)

df_train_complete, df_test = f.df_split(df_iris_shuffle, 0.3)
df_train, df_val = f.df_split(df_train_complete, 0.3)

x_train_complete = df_train_complete.iloc[:,:4].to_numpy()
y_train_complete = df_train_complete.iloc[:,5].to_numpy()
x_train = df_train.iloc[:,:4].to_numpy()
y_train = df_train.iloc[:,5].to_numpy()
x_val = df_val.iloc[:,:4].to_numpy()
y_val = df_val.iloc[:,5].to_numpy()
x_test = df_test.iloc[:,:4].to_numpy()
y_test = df_test.iloc[:,5].to_numpy()

idx_knn = f.get_knn(x_test, x_train_complete, k=3)
y_test_hat = f.get_classes(y_train_complete, idx_knn)

# Plot the distribution of classes along each variable
fig, axs = plt.subplots(2, 2, figsize=(16, 8))
for i, feature in enumerate(feature_names):
    ax = axs[i // 2, i % 2]
    for label in labels:
        ax.hist(x_train[y_train == label, i], bins=20, alpha=0.5, label=target_names[label])
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.legend()
plt.tight_layout()
plt.show()

accuracy_scores = []
for i in range(len(feature_names)):
    # Train the k-NN model using only the selected feature
    x_train_feature = x_train[:, i].reshape(-1, 1)
    x_val_feature = x_val[:, i].reshape(-1, 1)
    idx_knn_feature = f.get_knn(x_val_feature, x_train_feature, k=3)
    y_val_hat_feature = f.get_classes(y_train, idx_knn_feature)
    
    # Calculate the accuracy score
    accuracy = np.mean(y_val_hat_feature == y_val)
    accuracy_scores.append(accuracy)
# Find the index of the feature with the highest accuracy score
best_feature_index = np.argmax(accuracy_scores)
best_feature = feature_names[best_feature_index]
print(f"The most relevant feature (validation set) is: {best_feature}")

accuracy_scores = []
for i in range(len(feature_names)):
    # Train the k-NN model using only the selected feature
    x_train_feature = x_train[:, i].reshape(-1, 1)
    x_test_feature = x_test[:, i].reshape(-1, 1)
    idx_knn_feature = f.get_knn(x_test_feature, x_train_feature, k=3)
    y_test_hat_feature = f.get_classes(y_train, idx_knn_feature)
    
    # Calculate the accuracy score
    accuracy = np.mean(y_test_hat_feature == y_test)
    accuracy_scores.append(accuracy)
# Find the index of the feature with the highest accuracy score
best_feature_index = np.argmax(accuracy_scores)
best_feature = feature_names[best_feature_index]

print(f"The most relevant feature (test set) is: {best_feature}")


##########################################################
#             Second part of the assignment             #
#########################################################

accuracy_scores = []
best_features_val = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        # Train the k-NN model using only the selected features
        x_train_features = x_train[:, [i, j]]
        x_val_features = x_val[:, [i, j]]
        idx_knn_features = f.get_knn(x_val_features, x_train_features, k=3)
        y_val_hat_features = f.get_classes(y_train, idx_knn_features)
        
        # Calculate the accuracy score
        accuracy = np.mean(y_val_hat_features == y_val)
        accuracy_scores.append(accuracy)
        best_features_val.append((feature_names[i], feature_names[j]))
# Find the indices of the two features with the highest accuracy scores
best_features_indices_val = np.argsort(accuracy_scores)[-1:]
best_features_val = [best_features_val[i] for i in best_features_indices_val]
print("The most relevant features (validation set) are:")
for feature in best_features_val:
    print(f"{feature[0]} and {feature[1]}")

accuracy_scores = []
best_features_test = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        # Train the k-NN model using only the selected features
        x_train_features = x_train[:, [i, j]]
        x_test_features = x_test[:, [i, j]]
        idx_knn_features = f.get_knn(x_test_features, x_train_features, k=3)
        y_test_hat_features = f.get_classes(y_train, idx_knn_features)
        
        # Calculate the accuracy score
        accuracy = np.mean(y_test_hat_features == y_test)
        accuracy_scores.append(accuracy)
        best_features_test.append((feature_names[i], feature_names[j]))
# Find the indices of the two features with the highest accuracy scores
best_features_indices_test = np.argsort(accuracy_scores)[-1:]
best_features_test = [best_features_test[i] for i in best_features_indices_test]
print("The most relevant features (test set) are:")
for feature in best_features_test:
    print(f"{feature[0]} and {feature[1]}")