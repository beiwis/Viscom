import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patheffects
from scipy.spatial.distance import cdist
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
total_samples = x.shape[0]
print(f"\n1. How many samples of data has the IRIS dataset?\nThe IRIS dataset has {total_samples} samples in total.\n----------------------------------------------------\n")
class_counts = np.bincount(y)
for i, count in enumerate(class_counts):
  print(f"\n2. How many samples for each of each class?\n Class {i}: {count} samples\n----------------------------------------------------\n")


# -- Create a DataFrame with it.
####################################################
data = dict(
    [(n, x[:,i]) for i, n in enumerate(feature_names)] +
    [("flower name", target_names[y]), ("label", y)]
  )
df_iris = pd.DataFrame(data)
# print('\n\ndataset: \n', df_iris)

def shuffle_data(df: pd.DataFrame):
  """
  Shuffles a DataFrame rows randomly.
  """
  return df.sample(frac=1.0, random_state = 0).reset_index(drop = True)

df_iris_shuffle = shuffle_data(df_iris)
df_iris_shuffle

def df_split(
    df: pd.DataFrame,
    frac: float = 0.15
  ):
  """
  Splits the DataFrame in two parts, one with `frac` fraction of the data.
  """
  n_total = len(df)
  n_split_2 = int(n_total * frac)
  n_split_1 = n_total - n_split_2

  assert n_split_1 > 0 and n_split_2 > 0, "Splits must not be empty."

  df_split_1 = df.iloc[:n_split_1].reset_index(drop = True)
  df_split_2 = df.iloc[n_split_1 : n_split_1 + n_split_2].reset_index(drop = True)
  return df_split_1, df_split_2

df_train_complete, df_test = df_split(df_iris_shuffle, 0.3)
df_train, df_val = df_split(df_train_complete, 0.3)

print(f"\n3. How many samples are used for training, validation and test?\nTraining has {len(df_train)} [{100*len(df_train)/total_samples}%], validation has {len(df_val)} [{100*len(df_val)/total_samples}%] and test has {len(df_test)} [{100*len(df_test)/total_samples}%] samples.\n----------------------------------------------------\n")

x_train_complete = df_train_complete.iloc[:,:4].to_numpy()
y_train_complete = df_train_complete.iloc[:,5].to_numpy()
x_train = df_train.iloc[:,:4].to_numpy()
y_train = df_train.iloc[:,5].to_numpy()
x_val = df_val.iloc[:,:4].to_numpy()
y_val = df_val.iloc[:,5].to_numpy()
x_test = df_test.iloc[:,:4].to_numpy()
y_test = df_test.iloc[:,5].to_numpy()
# print(x_test[:,0:1])

def get_knn(
    x: np.ndarray,
    x0: np.ndarray,
    metric: str = "euclidean",
    k: int = 1
  ):
  """
  Get the Nearest Neighbors for every element in x, w.r.t. every element in
  `x0`. Returns an array with the same number of elements than `x`
  containing the index of its nearest neighbor in `x0`.

  Arguments:
    x {np.ndarray} -- Data samples with shape [NxD]
    x0 {np.ndarray} -- Data samples to compute NN w.r.t. [CxD]
    metric {str} -- Metric to be used to compute the distance between to points.
      (default: "euclidean")

  Keyword Arguments:
    k {int} -- Search for the k-nearest neighbors. (default: 1)

  Returns:
    np.ndarray -- Array with the indexes that is nearest neighbor for
      each element in x. Shape: [N,k]
  """
  distances = []
  clusters, d = x0.shape
  distances = cdist(x,x0,metric=metric)
  idx = np.argpartition(distances, k, axis=-1)[...,:k]
  return idx

idx_knn = get_knn(x_test, x_train_complete, k=3)
print(f"\n4. What does the get_knn function do? And the get_classes?\n\nidx_knn (size = {idx_knn.shape}):\n{idx_knn}")

def get_classes(training_labels: np.ndarray, knn_indexes: np.ndarray):
  """
  Given the k nearest neighbors of N samples it returns their labels.

  Arguments:
    training_labels {np.ndarray} -- Training labels Shape: [Mx1]
    knn_indexes {np.ndarray} -- Array with the indexes that is nearest neighbor for
      each element. Shape: [N,k]

  Return:
    {np.ndarray} -- Predicted labels Shape: [Nx1]
  """
  y_hat = np.argmax(
      np.apply_along_axis(
          lambda x: np.bincount(x, minlength=training_labels.size),
          axis = -1,
          arr=training_labels[knn_indexes]
      ),
      axis = -1
    )
  return y_hat

# print(y_train_complete)
y_test_hat = get_classes(y_train_complete, idx_knn)
print(f"\ny_test_hat (size = {y_test_hat.shape}):\n{ y_test_hat}\n----------------------------------------------------\n")

def confusion_matrix(y_true, y_pred, labels):
  """
  Computes the confusion matrix for y_true (GT values) and y_pred,
  predictions given for a model, for the given labels.
  """
  cm = np.zeros((len(labels),len(labels)))
  for predicted_label in labels:
    for true_label in labels:
      cm[true_label,predicted_label] = np.logical_and(    # Bug corrected
          (y_true == true_label), (y_pred == predicted_label)
      ).sum()
  return cm

print(f"""\n5. What represents the value of the row 2, column 1 in the confusion matrix? Assume that the indexes start in 0 to (length − 1)\n\nconfusion_matrix:\n{confusion_matrix(y_test, y_test_hat, labels)}\n
In a confusion matrix, each row represents the instances of an actual class and each column represents the instances of a predicted class.:\n
            |\n
Real values |\n
            |________________\n
             Predicted values  \n
----------------------------------------------------\n""")

def plot_confusion_matrix(y_true, y_hat, labels, label_names = None):
  """
  Plots the confusion matrix.
  """
  cm = confusion_matrix(y_true = y_true, y_pred = y_hat, labels = labels)
  fig, ax = plt.subplots(figsize=(16, 12))
  im = ax.imshow(cm)
  # We want to show all ticks...
  ax.set_xticks(np.arange(len(labels)))
  ax.set_yticks(np.arange(len(labels)))
  plt.xlabel('Predicted label')
  plt.ylabel('True Label')

  # ... and label them with the respective list entries
  if label_names is not None:
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")


  # Loop over data dimensions and create text annotations.
  border = [patheffects.withStroke(linewidth=3, foreground="w")]
  for i in range(len(labels)):
      for j in range(len(labels)):
          text = ax.text(j, i, f"{cm[i, j]:.0f}",
                        ha="center", va="center", color="k",
                        path_effects=border)
  fig.colorbar(im)
  plt.show()

# plot_confusion_matrix(y_test, y_test_hat, labels, target_names)
  
print(f"\n6. What are the metrics that are shown? What represent each one of them?\n\nAccuracy, Precision, Recall, F1 Score\n")

def metrics(y_true, y_hat, labels, label_names = None):
  """
  Compute metrix per class and the average.
  Print their values.

  Note:
  The Average Accuracy is not the average of the accuracies
  but the total accuracy.
  """
  label_names = labels if label_names is None else label_names
  metrics = []
  t_acc = 0
  t_precision = 0
  t_recall = 0
  t_f1 = 0
  for label in labels:
    TP = (label == y_hat[y_true == label]).sum()
    TN = (label != y_hat[y_true != label]).sum()
    FP = (label == y_hat[y_true != label]).sum()
    FN = (label != y_hat[y_true == label]).sum()
    acc = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    t_precision += precision
    t_recall += recall
    t_f1 += f1
    print(
      f"Label {label_names[label]:<10} Acc: {acc:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}, F1 {f1:.3f}."
    )
  t_acc = (y_true == y_hat).sum()/y_true.shape[0]
  print(
    f"\nAverage: Acc: {t_acc:.3f}, Prec: {t_precision/len(labels):.3f}, Rec: {t_recall/len(labels):.3f}, F1 {t_f1/len(labels):.3f}."
  )
  return t_acc

t_acc = metrics(y_test, y_test_hat, labels, target_names)
print(f"\n{t_acc: .3f}\n----------------------------------------------------\n")


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
best_k = 3

def pca_matrix(x: np.ndarray, dims_rescaled: int):
  """
  Computes the matrix to perform the linear tranformation that we will apply
  to reduce the dimensions of the data to `dims_rescaled` dimensions.

  Returns a matrix with of [Dxdims_rescaled] with `dims_rescaled` column
  vectors of D dimensions corresponding to the directions of the space where
  the data samples in `x` have more variance.

  Arguments:
    x {np.ndarray} -- N data samples of D features/dimensions. Shape: [NxD]
    dims_rescaled {int} -- Dimensions that we want to use to represent our data.

  Returns:
    {np.ndarray} -- Principal vectors in as columns of a matrix.
      Shape: [Dxdims_rescaled]
  """
  sigma_x = np.cov(x, rowvar=False)
  u,s,_ = np.linalg.svd(sigma_x)
  u = u[:,:dims_rescaled]
  return u

def apply_pca(x,u):
  """
  Applies the PCA transformation and returns the transformed data to the reduced
  dimensions.

  Arguments:
    x {np.ndarray} -- N data samples of D features/dimensions. Shape: [NxD]
    u {np.ndarray} -- Principal vecotors in as columns of a matrix.
      Shape: [Dxdims_rescaled]

  Returns:
    {np.ndarray} -- N data samples of dims_rescaled features/dimensions.
      Shape: [Nxdims_rescaled]
  """
  return np.dot(x,u)


u_matrix = pca_matrix(x_train, 2)
print(f"\n8. What is PCA doing? What is the direction of the first principal component?\nmatrix: {u_matrix}\nmain direction: {u_matrix[0]}\n----------------------------------------------------\n")

# -- Plot the first principal component.
####################################################
u_matrix = pca_matrix(x_train, 1)

x_train_complete_reduced = apply_pca(x_train_complete, u_matrix)
x_test_reduced = apply_pca(x_test, u_matrix)

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

# -- Plot the two first component.
####################################################
u_matrix = pca_matrix(x_train, 2)

x_train_complete_reduced = apply_pca(x_train_complete, u_matrix)
x_test_reduced = apply_pca(x_test, u_matrix)

fig, ax = plt.subplots(figsize=(16,8))
group = np.array(labels)

for g in np.unique(group):
    i = np.where(group == g)
    ax.scatter(
        x_train_complete_reduced[:,0][y_train_complete==g],
        x_train_complete_reduced[:,1][y_train_complete==g],
        label=f"{target_names[g]} train")
    ax.scatter(
        x_test_reduced[:,0][y_test==g],
        x_test_reduced[:,1][y_test==g],
        label=f"{target_names[g]} test",
        marker="+")

ax.legend()
plt.title("PCA principal components. (2 components)")
plt.show()

# -- Search the best number of dimentions using the validation set.
##############################################################################
k = best_k
best_dims = x_train.shape[1]
best_acc = 0
for dims in range(1, x_train.shape[1] + 1, 1):
  u_matrix = pca_matrix(x_train, dims)
  x_train_reduced = apply_pca(x_train, u_matrix)
  x_val_reduced = apply_pca(x_val, u_matrix)

  idx_knn = get_knn(x_val_reduced, x_train_reduced, k = k)
  y_val_hat = get_classes(y_train, idx_knn)

  print(f"PCA with {dims} dimensions.")
  t_acc = metrics(y_val, y_val_hat, labels, target_names)
  print("-----------------------------------------------")

  if t_acc > best_acc:
    best_dims = dims
    best_acc = t_acc

# -- Test with that number of dimensions

dims = best_dims
print("-----------------------------------------------")
print(f"Test set, with {dims} dimensiones")
u_matrix = pca_matrix(x_train, dims)
x_train_complete_reduced = apply_pca(x_train_complete, u_matrix)
x_test_reduced = apply_pca(x_test, u_matrix)

idx_knn = get_knn(x_test_reduced, x_train_complete_reduced, k = k)
y_test_hat = get_classes(y_train_complete, idx_knn)
plot_confusion_matrix(y_test, y_test_hat, labels, target_names)
t_acc_test = metrics(y_test, y_test_hat, labels, target_names)

