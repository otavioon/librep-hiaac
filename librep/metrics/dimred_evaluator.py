from pathlib import Path

from typing import List
from pyDRMetrics import DRMetrics

from librep.base.evaluators import CustomMultiEvaluator
from librep.config.type_definitions import ArrayLike, PathLike


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

        return result