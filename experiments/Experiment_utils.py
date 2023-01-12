# import sys
# sys.path.append("../../..")
# from pathlib import Path
from librep.datasets.multimodal import TransformMultiModalDataset
from librep.transforms.fft import FFT
from librep.utils.dataset import PandasDatasetsIO
from librep.datasets.multimodal import PandasMultiModalDataset, ArrayMultiModalDataset
import pandas as pd

from librep.utils.workflow import SimpleTrainEvalWorkflow, MultiRunWorkflow
from librep.metrics.report import ClassificationReport
from librep.estimators import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def create_PandasMultiModalDataset(X, y):
    X_pd = pd.DataFrame(X)
    # print(len(X), len(y), X_pd)
    X_pd['y'] = y
    
    X_pmd = PandasMultiModalDataset(
        X_pd,
        label_columns="y",
        as_array=True
    )
    return X_pmd

def create_ArrayMultiModalDataset(X, y):
    return ArrayMultiModalDataset(X, y, window_slices=[])

def run_experiments_accuracy_f1(train_X, train_Y, test_X, test_Y):
    # Create PandasMultiModalDataset
    train_pmd = create_PandasMultiModalDataset(train_X, train_Y)
    test_pmd = create_PandasMultiModalDataset(test_X, test_Y)
    # Create ArrayMultiModalDataset
    train_amd = ArrayMultiModalDataset(train_X, train_Y, window_slices=[])
    test_amd = ArrayMultiModalDataset(test_X, test_Y, window_slices=[])
    
    # Set reporter
    reporter = ClassificationReport(
        use_accuracy=True,
        use_f1_score=True,
        use_classification_report=False,
        use_confusion_matrix=False,
        plot_confusion_matrix=False
    )
    # Experiment for Random Forest
    experiment = SimpleTrainEvalWorkflow(
        estimator=RandomForestClassifier,
        estimator_creation_kwags ={'n_estimators':100} ,
        do_not_instantiate=False,
        do_fit=True,
        evaluator=reporter
    )
    # Run experiment
    multi_run_experiment = MultiRunWorkflow(workflow=experiment, num_runs=10, debug=False)
    result = multi_run_experiment(train_pmd, test_pmd)

    # Obtain accuracy and f1
    rf_accuracy = [run['result'][0]['accuracy'] for run in result['runs']]
    rf_f1 = [run['result'][0]['f1 score (weighted)'] for run in result['runs']]

    # Experiment for SVC
    experiment = SimpleTrainEvalWorkflow(
        estimator=SVC,
        estimator_creation_kwags={'C': 3.0, 'kernel': "rbf"},
        do_not_instantiate=False,
        do_fit=True,
        evaluator=reporter
    )

    # Run experiment
    multi_run_experiment = MultiRunWorkflow(workflow=experiment, num_runs=1, debug=False)
    result = multi_run_experiment(train_pmd, test_pmd)
    # Obtain accuracy and f1
    svc_accuracy = [run['result'][0]['accuracy'] for run in result['runs']]
    svc_f1 = [run['result'][0]['f1 score (weighted)'] for run in result['runs']]

    # Experiment for KNN
    experiment = SimpleTrainEvalWorkflow(
        estimator=KNeighborsClassifier,
        estimator_creation_kwags ={'n_neighbors' :1} ,
        do_not_instantiate=False,
        do_fit=True,
        evaluator=reporter
    )
    # Run experiment
    multi_run_experiment = MultiRunWorkflow(workflow=experiment, num_runs=1, debug=False)
    result = multi_run_experiment(train_amd, test_amd)
    
    # Experiment for KNN
    experiment = SimpleTrainEvalWorkflow(
        estimator=KNeighborsClassifier,
        estimator_creation_kwags ={'n_neighbors' :1} ,
        do_not_instantiate=False,
        do_fit=True,
        evaluator=reporter
    )
    knn_accuracy_amd = [run['result'][0]['accuracy'] for run in result['runs']]
    knn_f1_amd = [run['result'][0]['f1 score (weighted)'] for run in result['runs']]
    print('KNN - AMD', knn_accuracy_amd, knn_f1_amd)
    
    # Run experiment
    multi_run_experiment = MultiRunWorkflow(workflow=experiment, num_runs=1, debug=False)
    result = multi_run_experiment(train_pmd, test_pmd)
    # Obtain accuracy and f1
    knn_accuracy = [run['result'][0]['accuracy'] for run in result['runs']]
    knn_f1 = [run['result'][0]['f1 score (weighted)'] for run in result['runs']]
    print('KNN - PMD', knn_accuracy, knn_f1)
    return_object = {
        'RF-ACC': rf_accuracy,
        'RF-F1': rf_f1,
        'SVC-ACC': svc_accuracy,
        'SVC-F1': svc_f1,
        'KNN-ACC': knn_accuracy,
        'KNN-F1': knn_f1
    }
    return return_object

def run_experiments(train_HD, train_LD, train_Y, test_HD, test_LD, test_Y):
    # Create PandasMultiModalDataset
    train_pmd = create_PandasMultiModalDataset(train_LD, train_Y)
    test_pmd = create_PandasMultiModalDataset(test_LD, test_Y)
    # Set reporter
    reporter = ClassificationReport(
        use_accuracy=True,
        use_f1_score=True,
        use_classification_report=False,
        use_confusion_matrix=False,
        plot_confusion_matrix=False
    )
    # Experiment for Random Forest
    experiment = SimpleTrainEvalWorkflow(
        estimator=RandomForestClassifier,
        estimator_creation_kwags ={'n_estimators':100} ,
        do_not_instantiate=False,
        do_fit=True,
        evaluator=reporter)
    # Run experiment
    multi_run_experiment = MultiRunWorkflow(workflow=experiment, num_runs=10, debug=False)
    result = multi_run_experiment(train_pmd, test_pmd)
    # result = multi_run_experiment(train_LD, [test_LD])
    
    # Obtain accuracy and f1
    rf_accuracy = np.mean([run['result'][0]['accuracy'] for run in result['runs']])
    rf_f1 = np.mean([run['result'][0]['f1 score (weighted)'] for run in result['runs']])
    
    # Experiment for SVC
    experiment = SimpleTrainEvalWorkflow(
    estimator=SVC, 
    estimator_creation_kwags ={'C':3.0, 'kernel':"rbf"} ,
    do_not_instantiate=False, 
    do_fit=True,
    evaluator=reporter)

    # Run experiment
    multi_run_experiment = MultiRunWorkflow(workflow=experiment, num_runs=1, debug=False)
    result = multi_run_experiment(train_pmd, test_pmd)
    # Obtain accuracy and f1
    svc_accuracy = np.mean([run['result'][0]['accuracy'] for run in result['runs']])
    svc_f1 = np.mean([run['result'][0]['f1 score (weighted)'] for run in result['runs']])
    
    # Experiment for KNN
    experiment = SimpleTrainEvalWorkflow(
        estimator=KNeighborsClassifier,
        estimator_creation_kwags ={'n_neighbors' :1} ,
        do_not_instantiate=False,
        do_fit=True,
        evaluator=reporter
    )
    # Run experiment
    multi_run_experiment = MultiRunWorkflow(workflow=experiment, num_runs=1, debug=False)
    result = multi_run_experiment(train_pmd, test_pmd)
    # Obtain accuracy and f1
    knn_accuracy = np.mean([run['result'][0]['accuracy'] for run in result['runs']])
    knn_f1 = np.mean([run['result'][0]['f1 score (weighted)'] for run in result['runs']])
    return_object = {
        'RF-ACC': rf_accuracy,
        'RF-F1': rf_f1,
        'SVC-ACC': svc_accuracy,
        'SVC-F1': svc_f1,
        'KNN-ACC': knn_accuracy,
        'KNN-F1': knn_f1
    }
    return return_object