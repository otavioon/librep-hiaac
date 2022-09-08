from sklearn.manifold import TSNE as reducer
import pandas as pd

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike

class TSNE(Transform):
    """Extract statistical information of a sample.

    Parameters
    ----------
    keep_values : bool
        If true, the statistical information is concatenated with the input
        sample. (the default is False).
    capture_statistical : bool
        If True, extract statistical information about the sample.
    capture_indices : bool
        If True, extract statistical information about the indexes of the sample.

    """
    def __init__(self,
                 # metric: str = 'euclidean',
                  random_state: float = 42):
        # self.metric = metric
        self.random_state = random_state

    def fit(self, X, y):
        data = TSNE(random_state=self.random_state).fit_transform(X)
        tsne_df = pd.DataFrame(data, columns=["X", "Y"])
        tsne_df["class"] = y
        return tsne_df
        
        """Not used.

        """

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Extract statistical information of samples.

        Parameters
        ----------
        X : ArrayLike
            The sample used to extract the information, with shape (n_samples, n_features, ).

        Returns
        -------
        ArrayLike
            An array with the statistical information about the samples. If 
            `keep_values` parameter is set, the statistical information will be
            concatenated along the input sample.

        """
