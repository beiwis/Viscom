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

# print(y_train_complete)
y_test_hat = f.get_classes(y_train_complete, idx_knn)
# print(f"\ny_test_hat (size = {y_test_hat.shape}):\n{ y_test_hat}\n----------------------------------------------------\n")

# plot_confusion_matrix(y_test, y_test_hat, labels, target_names)
 
# t_acc = f.metrics(y_test, y_test_hat, labels, target_names)
# print(f"\n{t_acc: .3f}\n----------------------------------------------------\n")


# -- Search the optimal K for the K-NN algorithm using validation set
##############################################################################
# best_k = 0
# best_acc = 0
# for k in range(1, 6, 2): # [1, 3, 5]
#   print(f"Trying {k}-NN.")
#   idx_knn = get_knn(x_val, x_train, k = k)
#   y_val_hat = get_classes(y_train, idx_knn)
#   t_acc_val = metrics(y_val, y_val_hat, labels, target_names)
#   if t_acc_val > best_acc:
#     best_k = k
#     best_acc = t_acc_val
#   print("-----------------------------------------------")

# # -- Test using the optimal K.
# k = best_k
# print("-----------------------------------------------")
# print(f"Applying on test set. K={k}")
# idx_knn = get_knn(x_test, x_train_complete, k = k)
# y_test_hat = get_classes(y_train_complete, idx_knn)
# plot_confusion_matrix(y_test, y_test_hat, labels, target_names)
# t_acc_test = metrics(y_test, y_test_hat, labels, target_names)


# u_matrix = f.pca_matrix(x_train, 2)

# -- Plot the first principal component.
####################################################
u_matrix = f.pca_matrix(x_train, 1)

x_train_complete_reduced = f.apply_pca(x_train_complete, u_matrix)
x_test_reduced = f.apply_pca(x_test, u_matrix)

fig, ax = plt.subplots(figsize=(16,8))
group = np.array(labels)

for g in np.unique(group):
    i = np.where(group == g)
    ax.scatter(
        x_train_complete_reduced[:,0][y_train_complete==g],
        np.zeros_like(x_train_complete_reduced[:,0][y_train_complete==g]),
        label=f"{target_names[g]} train")
    ax.scatter(
        x_test_reduced[:,0][y_test==g],
        np.zeros_like(x_test_reduced[:,0][y_test==g]),
        label=f"{target_names[g]} test",
        marker="+")

ax.legend()
plt.title("PCA principal components. (1 component)")
plt.show()

# # -- Plot the two first component.
# ####################################################
# u_matrix = pca_matrix(x_train, 2)

# x_train_complete_reduced = apply_pca(x_train_complete, u_matrix)
# x_test_reduced = apply_pca(x_test, u_matrix)

# fig, ax = plt.subplots(figsize=(16,8))
# group = np.array(labels)

# for g in np.unique(group):
#     i = np.where(group == g)
#     ax.scatter(
#         x_train_complete_reduced[:,0][y_train_complete==g],
#         x_train_complete_reduced[:,1][y_train_complete==g],
#         label=f"{target_names[g]} train")
#     ax.scatter(
#         x_test_reduced[:,0][y_test==g],
#         x_test_reduced[:,1][y_test==g],
#         label=f"{target_names[g]} test",
#         marker="+")

# ax.legend()
# plt.title("PCA principal components. (2 components)")
# plt.show()

# # -- Search the best number of dimentions using the validation set.
# ##############################################################################
# k = best_k
# best_dims = x_train.shape[1]
# best_acc = 0
# for dims in range(1, x_train.shape[1] + 1, 1):
#   u_matrix = pca_matrix(x_train, dims)
#   x_train_reduced = apply_pca(x_train, u_matrix)
#   x_val_reduced = apply_pca(x_val, u_matrix)

#   idx_knn = get_knn(x_val_reduced, x_train_reduced, k = k)
#   y_val_hat = get_classes(y_train, idx_knn)

#   print(f"PCA with {dims} dimensions.")
#   t_acc = metrics(y_val, y_val_hat, labels, target_names)
#   print("-----------------------------------------------")

#   if t_acc > best_acc:
#     best_dims = dims
#     best_acc = t_acc

# # -- Test with that number of dimensions

# dims = best_dims
# print("-----------------------------------------------")
# print(f"Test set, with {dims} dimensiones")
# u_matrix = pca_matrix(x_train, dims)
# x_train_complete_reduced = apply_pca(x_train_complete, u_matrix)
# x_test_reduced = apply_pca(x_test, u_matrix)

# idx_knn = get_knn(x_test_reduced, x_train_complete_reduced, k = k)
# y_test_hat = get_classes(y_train_complete, idx_knn)
# plot_confusion_matrix(y_test, y_test_hat, labels, target_names)
# t_acc_test = metrics(y_test, y_test_hat, labels, target_names)


