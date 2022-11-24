#!/usr/bin/env python
# coding: utf-8

# # Experiment MNIST - TSNE
# 
# This experiment tries to replicate the reult obtained by the paper https://arxiv.org/pdf/1906.00722.pdf where a process of dimensionality reduction was applied on the mnist dataset, and values of 0.946 for Trustworthiness and 0.938 for continuity were obtained. 

# ## Basic imports

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import tensorflow as tf
import sys
import numpy as np
import pandas as pd
from pathlib import Path
print(sys.path)
# sys.path.append("../../")


# ## Loading the dataset

# In[3]:


from librep.transforms import TSNE
from librep.transforms import UMAP
from librep.datasets.multimodal import TransformMultiModalDataset, ArrayMultiModalDataset, WindowedTransform
from librep.metrics.dimred_evaluator import DimensionalityReductionQualityReport, MultiDimensionalityReductionQualityReport
from librep.datasets.har.loaders import MNISTView


# In[4]:


# loader = MNISTView("../../data/old-views/MNIST/default/", download=False)
# train_val_mnist, test_mnist = loader.load(concat_train_validation=True)


# In[5]:


# train_val_mnist, test_mnist


# In[6]:


# train_val_pd_X = train_val_mnist.data.iloc[:,1:]
# train_val_pd_Y = train_val_mnist.data.iloc[:,0]
# test_pd_X = test_mnist.data.iloc[:,1:]
# test_pd_Y = test_mnist.data.iloc[:,0]


# In[16]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
train_val_pd_X = np.reshape(x_train, (-1, 28*28))
train_val_pd_Y = y_train
test_pd_X = np.reshape(x_test, (-1, 28*28))
test_pd_Y = y_test


# In[17]:


test_pd_X.shape


# In[9]:


# # Code to create new view for mnist
# columns = ['pixel-' + str(val) for val in range(784)]
# columns.insert(0, 'label')
# train_val_mnist.data.columns = columns
# train_val_mnist.data.to_csv('DATA_MNIST.csv', index=False)


# # Reduce with TSNE

# In[10]:


tsne_reducer = TSNE()
train_val_pd_X_reduced = tsne_reducer.fit_transform(train_val_pd_X)
test_pd_X_reduced = tsne_reducer.fit_transform(test_pd_X)


# In[ ]:


# train_x = np.array(train_val_mnist.data.iloc[:,1:])
# train_y = np.array(train_val_mnist.data.iloc[:,0])
# test_x = np.array(test_mnist.data.iloc[:,1:])
# test_y = np.array(test_mnist.data.iloc[:,0])


# In[ ]:


# mnist_dataset_train = ArrayMultiModalDataset(X=train_x, y=train_y, window_slices=[(0, 28*28)], 
#                                              window_names=["px"])
# mnist_dataset_test = ArrayMultiModalDataset(X=test_x, y=test_y, window_slices=[(0, 28*28)], 
#                                              window_names=["px"])


# In[ ]:


# transform_tsne = TSNE()
# transformer = TransformMultiModalDataset(transforms=[transform_tsne])
# train_applied_tsne = transformer(mnist_dataset_train)
# test_applied_tsne = transformer(mnist_dataset_test)


# In[ ]:


metrics_reporter = DimensionalityReductionQualityReport()
metrics_train_applied_tsne = metrics_reporter.evaluate([train_val_pd_X, train_val_pd_X_reduced])
print(metrics_train_applied_tsne)


# In[ ]:


metrics_reporter = DimensionalityReductionQualityReport()
metrics_test_applied_tsne = metrics_reporter.evaluate([test_pd_X, test_pd_X_reduced])
print(metrics_test_applied_tsne)


# # Reduce with UMAP

# In[ ]:


umap_reducer = UMAP()
train_val_pd_X_reduced = umap_reducer.fit_transform(train_val_pd_X)
test_pd_X_reduced = umap_reducer.fit_transform(test_pd_X)


# In[ ]:


metrics_reporter = DimensionalityReductionQualityReport()
metrics_train_applied_tsne = metrics_reporter.evaluate([train_val_pd_X, train_val_pd_X_reduced])
print(metrics_train_applied_tsne)


# In[ ]:


metrics_reporter = DimensionalityReductionQualityReport()
metrics_test_applied_tsne = metrics_reporter.evaluate([test_pd_X, test_pd_X_reduced])
print(metrics_test_applied_tsne)


# In[ ]:


# transform_umap = UMAP()
# transformer = TransformMultiModalDataset(transforms=[transform_umap])
# train_applied_umap = transformer(mnist_dataset_train)
# test_applied_umap = transformer(mnist_dataset_test)


# In[ ]:


# metrics_reporter = DimensionalityReductionQualityReport(sampling_threshold=60000)
# metrics_train_applied_umap = metrics_reporter.evaluate([mnist_dataset_train, train_applied_umap])
# print(metrics_train_applied_umap)


# In[ ]:


# metrics_reporter = DimensionalityReductionQualityReport(sampling_threshold=10000)
# metrics_test_applied_umap = metrics_reporter.evaluate([mnist_dataset_test, test_applied_umap])
# print(metrics_test_applied_umap)


# In[ ]:




