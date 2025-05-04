import threading
# from typing import Self

import numpy as np
from joblib import Parallel as _Parallel, delayed

from CART import CART
from dataset import Dataset

__all__ = ['RandomForestRegressor']

def _aspositive(n: int) -> int:
    if n >= 0:
        return n
    else:
        return (1 << 15) + n

# out is passed through a list
def _regresor_predict(callback, X, out, lock):
    prediction = callback(X)
    with lock:
        out += prediction

def _regressor_fit(callback, dataset, sample_weights):
    callback(dataset, sample_weights)

def Parallel(n_jobs):
    return _Parallel(n_jobs=n_jobs, backend='threading', prefer='threads')

class RandomForestRegressor:
    def __init__(
            self,
            nb_trees: int = 1000,
            *,
            interaction_depth: int = -1,
            max_depth: int = -1,
            minobs: int = 1,
            min_nb_new_instances: int = 1,
            exact_categorical_splits: bool = False,
            loss: str = 'poisson',
            epsilon: float = 1.,
            margin: str = 'absolute',
            split: str = 'best',
            pruning: bool = False,
            name: str = None,
            n_jobs: int = None,
            nb_cov: int = -1,
            prop_sample: float = 0.5,
            verbose: bool = False,
            # **kwargs
            # prop_root_p0=-1.,
            # id=0,
            # replacement=False,
            # prop_sample=1.0,
            # frac_valid=0.2,
            # parallel="Yes",
            # bootstrap="No",
    ):
        self.nb_trees_ = nb_trees
        self.n_jobs_ = n_jobs
        self.interaction_depth_ = _aspositive(interaction_depth)
        self.max_depth_ = _aspositive(max_depth)
        self.minobs_ = minobs
        self.min_nb_new_instances_ = min_nb_new_instances
        self.exact_categorical_splits_ = exact_categorical_splits
        self.loss_ = loss
        self.epsilon_ = epsilon
        self.margin_ = margin
        self.split_ = split
        self.pruning_ = pruning
        self.name_ = name
        self.nb_cov_ = _aspositive(nb_cov)
        self.prop_sample_ = prop_sample

        self.fitted_ = False
        self.trees_ = Parallel(n_jobs=self.n_jobs_)(
            delayed(CART)(
                max_interaction_depth=self.interaction_depth_,
                max_depth=self.max_depth_,
                minobs=self.minobs_,
                min_nb_new_instances=self.min_nb_new_instances_,
                exact_categorical_splits=self.exact_categorical_splits_,
                loss=self.loss_,
                pruning=self.pruning_,
                epsilon=self.epsilon_,
                nb_cov=self.nb_cov_,
                prop_sample=self.prop_sample_,
                bootstrap=True,
                id=id_,
                verbose=verbose
                # TODO: name
            )
            for id_ in range(self.nb_trees_)
        )

    def fit(self, dataset: Dataset, sample_weights=None) -> 'Self':
        Parallel(n_jobs=self.n_jobs_)(
            delayed(_regressor_fit)(self.trees_[i].fit, dataset, sample_weights)
            for i in range(self.nb_trees_)
        )
        self.fitted_ = True

    def predict(self, X):
        if not self.fitted_:
            raise ValueError('Unable to predict on Forest before training')
        out = np.zeros(X.shape[0], dtype=np.float64)
        lock = threading.Lock()
        Parallel(n_jobs=self.n_jobs_)(
            delayed(_regresor_predict)(self.trees_[i].predict, X, out, lock)
            for i in range(self.nb_trees_)
        )
        out /= self.nb_trees_
        return out


