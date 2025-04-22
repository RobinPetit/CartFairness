# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: language_level=3
# cython: linetrace=True

cimport cython
cimport numpy as np
from cython.parallel import prange

import numpy as np
import pandas as pd

cdef void _extract_mean_ys(
        np.float64_t[:] X, np.float64_t[:] y,
        np.float64_t[:] sums, np.int32_t[:] sizes) noexcept nogil:
    cdef int i
    cdef int value
    for i in prange(y.shape[0], nogil=True, schedule='runtime'):
        value = <int>(X[i])
        sums[value] += y[i]
        sizes[value] += 1


@cython.final
cdef class Dataset:
    def __init__(self, np.ndarray[object, ndim=2] X,
                  np.ndarray[np.float64_t, ndim=1] y,
                  np.ndarray[np.float64_t, ndim=1] p,
                  np.ndarray[object, ndim=1] dtypes,
                  np.ndarray[np.float64_t, ndim=1] w=None):
        self._y = y
        self._p = p
        self._w = w
        if self._w is None:
            self._w = np.ones(y.shape[0], dtype=np.float64)
        self._is_categorical = np.zeros(X.shape[1], dtype=np.uint8)
        self._X = np.empty_like(X, dtype=np.float64)
        self._indexed_X = None
        self._indexed_y = None
        self._indexed_p = None
        self._indexed_w = None
        self._size = X.shape[0]
        self._indices = np.arange(self._size, dtype=int)
        self._reverse_mapping = [[] for _ in range(X.shape[1])]
        cdef size_t nb_cols = X.shape[1]
        cdef bint categorical, is_pd_categorical, is_np_categorical
        cdef int col_idx
        for col_idx in range(len(dtypes)):
            dtype = dtypes[col_idx]
            is_pd_categorical = isinstance(dtype, pd.core.dtypes.dtypes.CategoricalDtype)
            is_np_categorical = isinstance(dtype, np.dtype) and dtype.kind == 'U'
            categorical = is_pd_categorical or is_np_categorical
            if categorical:
                self._is_categorical[col_idx] = True
                self._labelize(X, self._X, col_idx, True)
            else:
                assert isinstance(dtype, np.dtype) and dtype.kind in 'fiu'
                self._is_categorical[col_idx] = False
                np.asarray(self._X)[:, col_idx] = X[:, col_idx]

    def __getitem__(self, indices):
        assert isinstance(indices, (slice, np.ndarray))
        cdef Dataset ret = Dataset.__new__(Dataset)
        ret._X = self._X
        ret._y = self._y
        ret._p = self._p
        ret._w = self._w
        ret._is_categorical = self._is_categorical
        ret._indices = np.asarray(self._indices)[indices]
        ret._size = ret._indices.shape[0]
        ret._reverse_mapping = self._reverse_mapping
        ret._indexed_X = None
        ret._indexed_y = None
        ret._indexed_p = None
        ret._indexed_w = None
        return ret

    property nb_features:
        def __get__(self):
            return self._X.shape[1]

    cpdef bint is_categorical(self, int feature_idx):
        return self._is_categorical[feature_idx]

    cdef void _labelize(
            self, np.ndarray[object, ndim=2] X, np.float64_t[:, :] out,
            int col_idx, bint compute_mapping=True):
        cdef int counter = 0
        if compute_mapping:
            self._reverse_mapping[col_idx] = np.unique(X[:, col_idx])
        cdef np.ndarray filled_indices = np.zeros(X.shape[0], dtype=bool)
        cdef np.ndarray indices
        for value in self._reverse_mapping[col_idx]:
            indices = np.where(X[:, col_idx] == value)[0]
            filled_indices[indices] = True
            np.asarray(out)[indices, col_idx] = counter
            counter += 1
        np.asarray(out)[~filled_indices, col_idx] = -1

    cdef Dataset sample(self, double prop_sample, bint replacement):
        cdef np.ndarray sampled_indices = np.random.choice(
            self._size, size=int(self._size * prop_sample), replace=replacement
        )
        return self[sampled_indices]

    cdef bint not_all_equal(self, int col_idx) noexcept nogil:
        cdef size_t i = 0
        cdef np.float64_t val = self._X[self._indices[i], col_idx]
        while i < self._size:
            if self._X[self._indices[i], col_idx] != val:
                return True
            i += 1
        return False

    cdef size_t get_length(self) noexcept nogil:
        return self._size
    cdef size_t size(self) noexcept nogil:
        return self._size

    @property
    def X(self):
        if self._indexed_X is None:
            self._indexed_X = np.asarray(self._X)[np.asarray(self._indices), :]
        return self._indexed_X

    @property
    def y(self):
        if self._indexed_y is None:
            self._indexed_y = np.asarray(self._y)[np.asarray(self._indices)]
        return self._indexed_y

    @property
    def p(self):
        if self._indexed_p is None:
            self._indexed_p = np.asarray(self._p)[np.asarray(self._indices)]
        return self._indexed_p

    @property
    def w(self):
        if self._indexed_w is None:
            self._indexed_w = np.asarray(self._w)[np.asarray(self._indices)]
        return self._indexed_w

    cdef np.ndarray transform(self, np.ndarray X):
        cdef np.ndarray ret = np.empty_like(X, dtype=np.float64)
        cdef int idx
        for feature_idx in range(X.shape[1]):
            if self._is_categorical[feature_idx]:
                self._labelize(X, ret, feature_idx, False)
            else:
                ret[:, feature_idx] = X[:, feature_idx]
        return ret

    cdef np.ndarray order_categorical(self, int feature_idx):
        cdef int size = len(self._reverse_mapping[feature_idx])
        cdef np.float64_t[:] ysums = np.zeros(size, dtype=np.float64)
        cdef np.int32_t[:] ysizes = np.zeros(size, dtype=np.int32)
        _extract_mean_ys(self.X[:, feature_idx], self.y[:], ysums, ysizes)
        cdef int i
        for i in range(ysums.shape[0]):
            if ysums[i] > 0:
                ysums[i] /= ysizes[i]
        cdef np.ndarray ret = np.argsort(ysums)
        return ret[np.asarray(ysizes)[ret] > 0]

    cpdef int nb_modalities_of(self, int feature_idx):
        return len(self._reverse_mapping[feature_idx])

