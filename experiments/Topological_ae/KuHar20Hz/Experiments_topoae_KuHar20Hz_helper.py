# ----------
# Imports
# ----------
import sys                # For include librep package
sys.path.append("../../..")
from pathlib import Path
from librep.datasets.multimodal import TransformMultiModalDataset
from librep.transforms.fft import FFT
from librep.utils.dataset import PandasDatasetsIO
from librep.datasets.multimodal import PandasMultiModalDataset
import pandas as pd

from librep.utils.workflow import SimpleTrainEvalWorkflow, MultiRunWorkflow
from librep.metrics.report import ClassificationReport
from librep.estimators import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# ----------
# Function to compute KuHar20Hz
# ----------
def obtainKuHar20Hz():
    # Path of the dataset
    dataset_path = Path("../../../data/old-views/KuHar/resampled_view_20Hz")
    # Loading dataset
    train, validation, test = PandasDatasetsIO(dataset_path).load()
    # Kuhar features to select
    features = [ "accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    # Creating PandasMultiModalDataset
    # Train
    train_dataset = PandasMultiModalDataset(
        train,
        feature_prefixes=features,
        label_columns="activity code",
        as_array=True
    )
    # Validation
    validation_dataset = PandasMultiModalDataset(
        validation,
        feature_prefixes=features,
        label_columns="activity code",
        as_array=True
    )

    # Test
    test_dataset = PandasMultiModalDataset(
        test,
        feature_prefixes=features,
        label_columns="activity code",
        as_array=True
    )
    fft_transform = FFT(centered = True)
    transformer = TransformMultiModalDataset(
        transforms=[fft_transform],
        new_window_name_prefix="fft."
    )
    train_dataset_fft = transformer(train_dataset)
    validation_dataset_fft = transformer(validation_dataset)
    test_dataset_fft = transformer(test_dataset)
    
    kuhar_data = {
        'train_HD': train_dataset_fft.X,
        'train_LD': None,
        'train_Y': train_dataset_fft.y,
        'test_HD': test_dataset_fft.X,
        'test_LD': None,
        'test_Y': test_dataset_fft.y
    }
    return kuhar_data

def create_PandasMultiModalDataset(X, y):
    X_pd = pd.DataFrame(X)
    X_pd['y'] = y
    X_pmd = PandasMultiModalDataset(
        X_pd,
        label_columns="y",
        as_array=True
    )
    return X_pmd

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