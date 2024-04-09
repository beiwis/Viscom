import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patheffects
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression


def shuffle_data(df: pd.DataFrame):
  """
  Shuffles a DataFrame rows randomly.
  """
  return df.sample(frac=1.0, random_state = 0).reset_index(drop = True)


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
    u {np.ndarray} -- Principal vectors in as columns of a matrix.
      Shape: [Dxdims_rescaled]

  Returns:
    {np.ndarray} -- N data samples of dims_rescaled features/dimensions.
      Shape: [Nxdims_rescaled]
  """
  return np.dot(x,u)

def get_least_dimensions(x_train, y_train, x_val, y_val, target_accuracy):
    dimensions = x_train.shape[1]
    while dimensions > 0:
        model = LogisticRegression()
        model.fit(x_train[:, :dimensions], y_train)
        accuracy = model.score(x_val[:, :dimensions], y_val)
        if accuracy >= target_accuracy:
            return dimensions
        dimensions -= 1
    return 0
