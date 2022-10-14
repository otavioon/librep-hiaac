from pathlib import Path
import numpy as np
from typing import List
from pyDRMetrics.pyDRMetrics import DRMetrics
import matplotlib.pyplot as plt

from librep.base.evaluators import CustomMultiEvaluator
from librep.config.type_definitions import ArrayLike, PathLike
from librep.datasets.multimodal import ArrayMultiModalDataset


class DimensionalityReductionQualityReport(CustomMultiEvaluator):
    def __init__(
        self,
        use_residual_variance_pearson: bool = True,
        use_residual_variance_spearman: bool = True,
        use_trustworthiness: bool = True,
        use_continuity: bool = True,
        use_co_k_nearest_neighbor_size: bool = True,
        use_local_continuity_meta_criterion: bool = True,
        use_local_property: bool = True,
        use_global_property: bool = True,
        neighbors_considered = 15,
        output_path: PathLike = None
    ):
        self.use_residual_variance_pearson = use_residual_variance_pearson
        self.use_residual_variance_spearman = use_residual_variance_spearman
        self.use_trustworthiness = use_trustworthiness
        self.use_continuity = use_continuity
        self.use_co_k_nearest_neighbor_size = use_co_k_nearest_neighbor_size
        self.use_local_continuity_meta_criterion = use_local_continuity_meta_criterion
        self.use_local_property = use_local_property
        self.use_global_property = use_global_property
        self.neighbors_considered = neighbors_considered
        self.output_path = Path(output_path) if output_path is not None else None 

        # TODO Save
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)

    def evaluate(
        self, Xs: List[ArrayLike]
    ) -> dict:
        result = {}
        # Assuming X_HD is the first element Xs[0]
        # Assuming X_LD is the first element Xs[1]
        X_highdim = Xs[0]
        X_lowdim = Xs[1]
        # if X_highdim.X:
        #     X_highdim = X_highdim.X
        # if X_lowdim.X:
        #     X_lowdim = X_lowdim.X
        if type(X_highdim) == ArrayMultiModalDataset:
            X_highdim = X_highdim.X
        if type(X_lowdim) == ArrayMultiModalDataset:
            X_lowdim = X_lowdim.X
        # datapoints = X_highdim.shape[0]
        # samples_range = []
        # for i in range(0, datapoints, self.sampling_threshold):
        #     if i + self.sampling_threshold <= datapoints:
        #         samples_range.append((i, i+self.sampling_threshold))
        # print(samples_range)
        X_highdim = X_highdim.astype(np.float32)
        X_lowdim = X_lowdim.astype(np.float32)
        # drms = []
        # for i in samples_range:
        #     # print(i)
        #     drms.append(DRMetrics(X_highdim[i[0]:i[1]], X_lowdim[i[0]:i[1]]))
        drm = DRMetrics(X_highdim, X_lowdim)

        if self.use_residual_variance_pearson:
            # res = np.mean([drm.Vr for drm in drms])
            res = drm.Vr
            result["residual variance (pearson)"] = float(res)

        if self.use_residual_variance_spearman:
            # res = np.mean([drm.Vrs for drm in drms])
            res = drm.Vrs
            result["residual variance (spearman)"] = float(res)

        if self.use_trustworthiness:
            # res = np.mean([drm.T[self.neighbors_considered] for drm in drms])
            res = drm.T[self.neighbors_considered]
            result["trustworthiness"] = float(res)

        if self.use_continuity:
            # res = np.mean([drm.C[self.neighbors_considered] for drm in drms])
            res = drm.C[self.neighbors_considered]
            result["continuity"] = float(res)

        if self.use_co_k_nearest_neighbor_size:
            # res = np.mean([drm.QNN[self.neighbors_considered] for drm in drms])
            res = drm.QNN[self.neighbors_considered]
            result["co k nearest neighbor size"] = float(res)

        if self.use_local_continuity_meta_criterion:
            # res = np.mean([drm.LCMC[self.neighbors_considered] for drm in drms])
            res = drm.LCMC[self.neighbors_considered]
            result["local continuity meta criterion"] = float(res)

        if self.use_local_property:
            # res = np.mean([drm.Qlocal for drm in drms])
            res = drm.Qlocal
            result["local property"] = res

        if self.use_global_property:
            # res = np.mean([drm.Qglobal for drm in drms])
            res = drm.Qglobal
            result["global property"] = res

        return result

class MultiDimensionalityReductionQualityReport(CustomMultiEvaluator):
    def __init__(
        self,
        use_residual_variance_pearson: bool = True,
        use_residual_variance_spearman: bool = True,
        use_trustworthiness: bool = True,
        use_continuity: bool = True,
        use_co_k_nearest_neighbor_size: bool = True,
        use_local_continuity_meta_criterion: bool = True,
        use_local_property: bool = True,
        use_global_property: bool = True,
        neighbors_considered = 15,
        output_path: PathLike = None
    ):
        self.use_residual_variance_pearson = use_residual_variance_pearson
        self.use_residual_variance_spearman = use_residual_variance_spearman
        self.use_trustworthiness = use_trustworthiness
        self.use_continuity = use_continuity
        self.use_co_k_nearest_neighbor_size = use_co_k_nearest_neighbor_size
        self.use_local_continuity_meta_criterion = use_local_continuity_meta_criterion
        self.use_local_property = use_local_property
        self.use_global_property = use_global_property
        self.neighbors_considered = neighbors_considered
        self.output_path = Path(output_path) if output_path is not None else None 

        # TODO Save
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)

    def evaluate(
        self, Xs: List[ArrayLike]
    ) -> ArrayLike:
        
        # Assuming X_HD is the first element Xs[0]
        # Assuming different X_LD are the rest of element Xs[1], Xs[2], Xs[3], ..., Xs[n]
        
        # Preprocessing all in Xs
        for dataset in Xs:
            if type(dataset) == ArrayMultiModalDataset:
                dataset = dataset.X
            dataset = dataset.astype(np.float32)
        # X_highdim = Xs[0]
        # if type(X_highdim) == ArrayMultiModalDataset:
        #     X_highdim = X_highdim.X
        # X_highdim = X_highdim.astype(np.float32)
        
        all_dimensions = []
        
        results = []
        
        X_highdim = Xs[0]
        for i in range(1,len(Xs)):
            result = {}
            X_lowdim = Xs[i]
            result['dim'] = X_lowdim.shape[1]
            all_dimensions.append(X_lowdim.shape[1])
            if type(X_lowdim) == ArrayMultiModalDataset:
                X_lowdim = X_lowdim.X
            X_lowdim = X_lowdim.astype(np.float32)
            drm = DRMetrics(X_highdim, X_lowdim)
            
            if self.use_residual_variance_pearson:
                res = drm.Vr
                result["residual variance (pearson)"] = float(res)
            if self.use_residual_variance_spearman:
                res = drm.Vrs
                result["residual variance (spearman)"] = float(res)
            if self.use_trustworthiness:
                res = drm.T[self.neighbors_considered]
                result["trustworthiness"] = float(res)
            if self.use_continuity:
                res = drm.C[self.neighbors_considered]
                result["continuity"] = float(res)
            if self.use_co_k_nearest_neighbor_size:
                res = drm.QNN[self.neighbors_considered]
                result["co k nearest neighbor size"] = float(res)
            if self.use_local_continuity_meta_criterion:
                res = drm.LCMC[self.neighbors_considered]
                result["local continuity meta criterion"] = float(res)
            if self.use_local_property:
                res = drm.Qlocal
                result["local property"] = res
            if self.use_global_property:
                res = drm.Qglobal
                result["global property"] = res
            results.append(result)
        
        
        # plot trustworthiness
        unique_dims = np.unique(all_dimensions)
        trustworthiness = [[], [], []]
        
        for unique_dim in unique_dims:
            trustworthiness_for_dim = [data['trustworthiness'] for data in results if data['dim']==unique_dim]
            # MIN
            trustworthiness[0].append(np.min(trustworthiness_for_dim))
            # MEAN
            trustworthiness[1].append(np.mean(trustworthiness_for_dim))
            # MAX
            trustworthiness[2].append(np.max(trustworthiness_for_dim))
        
        fig, ax = plt.subplots()
        ax.plot(unique_dims, trustworthiness[1], '-')
        ax.fill_between(unique_dims, trustworthiness[0], trustworthiness[2], alpha=0.2)
        return results