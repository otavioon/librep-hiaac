import sys
import pandas as pd
from pathlib import Path
import os
print('\n',os.getcwd())

sys.path.append("/home/hubert/librep-hiaac")

import tensorflow as tf
from librep.transforms import TSNE
from librep.transforms import UMAP
from librep.datasets.multimodal import TransformMultiModalDataset, ArrayMultiModalDataset
from librep.metrics.dimred_evaluator import DimensionalityReductionQualityReport

dataset = tf.keras.datasets.mnist.load_data(path="mnist.npz")
(train_x, train_y), (test_x, test_y) = dataset

train_x_reordered = train_x.reshape((60000,-1))
print('TRAIN_X_REORDERED', train_x_reordered.shape)

test_x_reordered = test_x.reshape((10000,-1))
print('TEST_X_REORDERED', test_x_reordered.shape)

mnist_dataset_train = ArrayMultiModalDataset(X=train_x_reordered, y=train_y, window_slices=[(0, 28*28)], 
                                             window_names=["px"])
mnist_dataset_test = ArrayMultiModalDataset(X=test_x_reordered, y=test_y, window_slices=[(0, 28*28)], 
                                             window_names=["px"])

transform_tsne = TSNE()
transformer = TransformMultiModalDataset(transforms=[transform_tsne])
train_applied_tsne = transformer(mnist_dataset_train)
test_applied_tsne = transformer(mnist_dataset_test)

metrics_reporter = DimensionalityReductionQualityReport(sampling_threshold=60000)
metrics_train_applied_tsne = metrics_reporter.evaluate([mnist_dataset_train, train_applied_tsne])
print(metrics_train_applied_tsne)