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
        self.sum_of_weights_p0 = self._w[p == 0].sum()
        self.sum_of_weights_p1 = self._w[p == 1].sum()

        cdef int col_idx
        for col_idx in range(len(dtypes)):
            dtype = dtypes[col_idx]
            is_pd_categorical = isinstance(
                dtype,
                pd.core.dtypes.dtypes.CategoricalDtype
            )
            is_np_categorical = isinstance(dtype, np.dtype) \
                    and dtype.kind == 'U'
            categorical = is_pd_categorical or is_np_categorical
            if categorical:
                self._is_categorical[col_idx] = True
                self._labelize(X, self._X, col_idx, True)
            else:
                assert isinstance(dtype, np.dtype) \
                        and dtype.kind in 'fiu'
                self._is_categorical[col_idx] = False
                self._X[:, col_idx] = X[:, col_idx]

    def __getstate__(self):
        return (
            self._X,
            self._y,
            self._p,
            self._w,
            self._is_categorical,
            self._indices,
            self._size,
            self._reverse_mapping,
            self.sum_of_weights_p0,
            self.sum_of_weights_p1,
        )

    def __setstate__(self, data):
        self._X = data[0]
        self._y = data[1]
        self._p = data[2]
        self._w = data[3]
        self._is_categorical = data[4]
        self._indices = data[5]
        self._size = data[6]
        self._reverse_mapping = data[7]
        self.sum_of_weights_p0 = data[8]
        self.sum_of_weights_p1 = data[9]
        self._indexed_X = None
        self._indexed_y = None
        self._indexed_p = None
        self._indexed_w = None

    def __len__(self) -> int:
        return self.size()

    def __getitem__(self, indices):
        assert isinstance(indices, (slice, np.ndarray))
        cdef Dataset ret = Dataset.__new__(Dataset)
        ret._X = self._X
        ret._y = self._y
        ret._p = self._p
        ret._w = self._w
        ret._is_categorical = self._is_categorical
        ret._indices = self._indices[indices]
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
        cdef tuple uniques
        if compute_mapping:
            uniques = np.unique(X[:, col_idx], return_index=True)
            # Slight twist for reproducibility
            self._reverse_mapping[col_idx] = uniques[0][np.argsort(uniques[1])]
        cdef np.ndarray filled_indices = np.zeros(X.shape[0], dtype=bool)
        cdef np.ndarray indices
        for value in self._reverse_mapping[col_idx]:
            indices = np.where(X[:, col_idx] == value)[0]
            filled_indices[indices] = True
            np.asarray(out)[indices, col_idx] = counter
            counter += 1
        np.asarray(out)[~filled_indices, col_idx] = -1

    cdef Dataset sample(self, double prop_sample,
                        bint replacement):
        cdef np.ndarray where_p_is_0 = np.where(
            np.asarray(self.p) == 0
        )[0]
        cdef np.ndarray where_p_is_1 = np.where(
            np.asarray(self.p) == 1
        )[0]
        cdef np.ndarray indices_0 = np.random.choice(
            where_p_is_0, where_p_is_0.shape[0],
            replace=replacement
        )
        cdef np.ndarray indices_1 = np.random.choice(
            where_p_is_1, where_p_is_1.shape[0],
            replace=replacement
        )
        cdef np.ndarray cumsum_on_0 = np.cumsum(
            np.asarray(self.w)[indices_0]
        )
        cdef np.ndarray cumsum_on_1 = np.cumsum(
            np.asarray(self.w)[indices_1]
        )
        cdef int i = np.searchsorted(
            cumsum_on_0, prop_sample*self.sum_of_weights_p0, 'right'
        )
        cdef int j = np.searchsorted(
            cumsum_on_1, prop_sample*self.sum_of_weights_p1, 'right'
        )
        cdef np.ndarray indices = np.empty(i+j, dtype=int)
        indices[:i] = indices_0[:i]
        indices[i:] = indices_1[:j]
        return self[indices]

    cdef bint not_all_equal(self, int col_idx) noexcept nogil:
        cdef size_t i = 1
        cdef np.float64_t[:, ::1] X
        with gil:
            X = self._get_X()
        cdef np.float64_t val = X[0, col_idx]
        while i < self._size:
            if X[i, col_idx] != val:
                return True
            i += 1
        return False

    cdef size_t get_length(self) noexcept nogil:
        return self._size
    cdef size_t size(self) noexcept nogil:
        return self._size

    @property
    def X(self):
        return self._get_X()

    @property
    def y(self):
        return self._get_y()

    @property
    def p(self):
        return self._get_p()

    @property
    def w(self):
        return self._get_w()

    cdef np.float64_t[:, ::1] _get_X(self) noexcept:
        if self._indexed_X is None:
            self._indexed_X = self._X[self._indices, :]
        return self._indexed_X

    cdef np.float64_t[::1] _get_y(self) noexcept:
        if self._indexed_y is None:
            self._indexed_y = self._y[self._indices]
        return self._indexed_y

    cdef np.float64_t[::1] _get_w(self) noexcept:
        if self._indexed_w is None:
            self._indexed_w = self._w[self._indices]
        return self._indexed_w

    cdef np.float64_t[::1] _get_p(self) noexcept:
        if self._indexed_p is None:
            self._indexed_p = self._p[self._indices]
        return self._indexed_p

    cdef np.ndarray transform(self, np.ndarray X):
        cdef np.ndarray[np.float64_t, ndim=2] ret = np.empty_like(
            X, dtype=np.float64, order='C'
        )
        cdef int idx
        for feature_idx in range(X.shape[1]):
            if self._is_categorical[feature_idx]:
                self._labelize(X, ret, feature_idx, False)
            else:
                ret[:, feature_idx] = X[:, feature_idx]
        return ret

    cpdef int nb_modalities_of(self, int feature_idx):
        return len(self._reverse_mapping[feature_idx])

    cdef np.float64_t get_prop_p0(self):
        if self._w is None:
            return np.mean(1-np.asarray(self.p))
        else:
            return np.average(1-np.asarray(self.p), weights=self.w)

    cdef str _reverse(self, int feature_idx, int id_):
        return self._reverse_mapping[feature_idx][id_]

    cdef void _clear(self) noexcept:
        self._indexed_X = None
        self._indexed_y = None
        self._indexed_w = None
        self._indexed_p = None
