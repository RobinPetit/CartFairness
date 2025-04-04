# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: language_level=3
# cython: linetrace=True

cimport numpy as np
from libc.math cimport log, fabs

ctypedef size_t Pyssize_t

cdef extern from "_CART.h" nogil:
    cdef struct _Node:
        _Node* left_child
        _Node* right_child
        _Node* parent
        np.float64_t avg_value
        size_t nb_samples
        size_t depth
        int feature_idx
        np.float64_t threshold
        np.float64_t loss
    _Node* new_node(size_t)
    void clear_node(_Node*)
    void _set_ys(_Node*, double, double, size_t)
    void _set_left_child(_Node*, _Node*)
    void _set_right_child(_Node*, _Node*)
    bint _is_root(_Node*)
    bint _is_leaf(_Node*)

cdef class Node:
    cdef _Node* node

    @staticmethod
    cdef Node from_ptr(_Node* ptr):
        cdef Node ret = Node.__new__(Node)
        ret.node = ptr
        return ret

import cython
from cython.operator cimport dereference
from cython.parallel import prange

import numpy as np
import pandas as pd

from time import time

PROBE = 0

class TODOError(ValueError):
    pass

ctypedef enum LossFunction:
    MSE,
    POISSON

cdef int indirect_binary_search(np.float64_t[:] values, np.float64_t threshold,
                                int beg, int end,
                                np.int64_t[:] sorted_indices) noexcept nogil:
    cdef int mid
    while beg < end:
        mid = (beg + end) // 2
        if values[sorted_indices[mid]] <= threshold:
            beg = mid+1
        else:
            end = mid
    return beg

cdef int _masks(np.float64_t[:] values, np.float64_t threshold,
                int base_idx, np.int64_t[:] sorted_indices) noexcept nogil:
    return indirect_binary_search(
        values, threshold, base_idx, values.shape[0], sorted_indices
    )

cdef np.float64_t mse(np.float64_t[:] ys) noexcept nogil:
    cdef np.float64_t mu = 0.
    cdef np.float64_t sum_square = 0.
    cdef size_t n = ys.shape[0]
    cdef size_t i = 0
    for i in prange(n, nogil=True, schedule='runtime'):
        mu += ys[i]
        sum_square += ys[i]*ys[i]
    mu /= n
    return sum_square / n + mu * mu

cdef np.float64_t poisson(np.float64_t[:] ys) noexcept nogil:
    cdef np.float64_t epsilon = 1e-18
    cdef np.float64_t mu = 0.
    cdef np.float64_t ret = 0
    cdef size_t i = 0
    cdef size_t n = ys.shape[0]
    for i in prange(n, nogil=True, schedule='runtime'):
        mu += ys[i]
    mu /= n
    for i in prange(n, nogil=True, schedule='runtime'):
        if ys[i] > epsilon and mu > epsilon:
            ret += ys[i] * log((ys[i] + epsilon) / (mu + epsilon)) + (mu - ys[i])
    return 2 * ret / n


@cython.final
cdef class Dataset:
    cdef np.float64_t[:, :] _X
    cdef np.float64_t[:] _y
    cdef np.float64_t[:] _p
    cdef np.uint8_t[:] _is_categorical
    cdef np.int_t[:] _indices
    cdef size_t _size
    cdef list _reverse_mapping

    def __init__(self, np.ndarray[object, ndim=2] X,
                  np.ndarray[np.float64_t, ndim=1] y,
                  np.ndarray[np.float64_t, ndim=1] p,
                  np.ndarray[object, ndim=1] dtypes):
        self._y = y
        self._p = p
        self._is_categorical = np.zeros(X.shape[1], dtype=np.uint8)
        self._X = np.empty_like(X, dtype=np.float64)
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
        ret._is_categorical = self._is_categorical
        ret._indices = np.asarray(self._indices)[indices]
        ret._size = ret._indices.shape[0]
        return ret

    cdef void _labelize(self, np.ndarray[object, ndim=2] X,
                        np.float64_t[:, :] out,
                        int col_idx, bint compute_mapping=True):
        cdef int counter = 0
        if compute_mapping:
            self._reverse_mapping[col_idx] = np.unique(X[:, col_idx])
        cdef np.ndarray indices
        for value in self._reverse_mapping[col_idx]:
            indices = np.where(X[:, col_idx] == value)[0]
            np.asarray(out)[indices, col_idx] = counter
            counter += 1

    cdef Dataset sample(self, float prop_sample, bint replacement):
        cdef np.ndarray sampled_indices = np.random.choice(
            self._size, size=int(self._size * prop_sample), replace=replacement
        )
        return self[sampled_indices]

    cdef bint not_all_equal(self, int col_idx) nogil:
        cdef size_t i = 0
        cdef np.float64_t val = self._X[self._indices[i], col_idx]
        while i < self._size:
            if self._X[self._indices[i], col_idx] != val:
                return True
            i += 1
        return False

    cdef size_t get_length(self) noexcept nogil:
        return self._size

    @property
    def X(self):
        return np.asarray(self._X)[np.asarray(self._indices), :]

    @property
    def y(self):
        return np.asarray(self._y)[np.asarray(self._indices)]

    @property
    def p(self):
        return np.asarray(self._p)[np.asarray(self._indices)]

    cdef np.ndarray transform(self, np.ndarray X):
        cdef np.ndarray ret = np.empty_like(X, dtype=np.float64)
        cdef int idx
        for feature_idx in range(X.shape[1]):
            if self._is_categorical[feature_idx]:
                self._labelize(X, ret, feature_idx, False)
            else:
                ret[:, feature_idx] = X[:, feature_idx]
        return ret

@cython.final
cdef class SplitChoice:
    cdef size_t feature_idx
    cdef np.float64_t threshold
    cdef np.float64_t loss, dloss
    cdef np.float64_t loss_left, loss_right
    cdef Dataset left_data, right_data

    def __cinit__(self, size_t feature_idx, np.float64_t threshold,
                  np.float64_t loss, np.float64_t dloss,
                  np.float64_t left_loss, np.float64_t right_loss,
                  Dataset left_data, Dataset right_data):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.loss = loss
        self.dloss = dloss
        self.left_data = left_data
        self.right_data = right_data
        self.loss_left = left_loss
        self.loss_right = right_loss

@cython.final
cdef class CART:
    cdef bint bootstrap
    cdef bint replacement
    cdef LossFunction loss_fct
    cdef np.float64_t epsilon
    cdef int nb_cov
    cdef int id
    cdef np.float64_t prop_sample
    cdef np.float64_t delta_loss
    cdef size_t minobs
    cdef size_t max_depth
    cdef int max_interaction_depth
    cdef int nb_nodes
    cdef bint pruning
    cdef Dataset data
    cdef _Node* root

    cdef list nodes

    def __cinit__(self, epsilon=0., prop_root_p0=1.0, id=0, nb_cov=1,
                  replacement=False, prop_sample=1.0, frac_valid=0.2,
                  max_interaction_depth=0, max_depth=0, margin="absolute",
                  minobs=1, delta_loss=0, loss="MSE", name=None,
                  parallel="Yes", pruning="No", bootstrap="No"):
        self.bootstrap = (bootstrap == 'Yes')
        self.pruning = (pruning == 'Yes')
        self.replacement = replacement
        self.epsilon = epsilon
        self.nb_cov = nb_cov
        self.id = id
        self.prop_sample = prop_sample
        self.delta_loss = delta_loss
        self.minobs = minobs
        self.max_depth = 0
        self.nb_nodes = 0
        self.max_interaction_depth = max_interaction_depth
        loss = loss.lower()
        LOSS_MAPPING = {
            'mse': MSE,
            'poisson': POISSON
        }
        assert loss in LOSS_MAPPING.keys()
        self.loss_fct = LOSS_MAPPING[loss]

    def __dealloc__(self):
        clear_node(self.root)

    cdef np.float64_t _loss(self, np.float64_t[:] ys) nogil:
        cdef np.float64_t ret
        if self.loss_fct == MSE:
            ret = mse(ys)
        else:
            ret = poisson(ys)
        return ret

    def fit(self, dataset: Dataset):
        global PROBE
        PROBE = 0
        start = time()
        self.data = dataset
        if self.bootstrap:
            print('Bootstrapping...')
            self.data = self.data.sample(self.prop_sample, self.replacement)
        # split train vs test ?!
        self.root = self._build_tree(self.data)
        self._retrieve_all_nodes()
        if self.pruning:
            raise TODOError()
        time_elapsed = time() - start
        print("\n")
        print('*******************************')
        print(f"Tree {self.id}: Params(id={self.max_interaction_depth}, cov={self.nb_cov})")
        print(f"Time elapsed: {time_elapsed}")
        print(f"Tree depth:{self.max_depth}")
        print(f"Nb nodes: {len(self.nodes)}")
        print('*******************************')
        print(f'\t\t{100 * PROBE / time_elapsed:3.2f}%')

    cdef void _retrieve_all_nodes(self):
        self.nodes = list()
        cdef Node node
        cdef list stack = []
        stack.append(Node.from_ptr(self.root))
        while len(stack) > 0:
            node = stack.pop()
            self.nodes.append(node)
            if dereference(node.node).left_child != NULL:
                stack.append(Node.from_ptr(dereference(node.node).left_child))
            if dereference(node.node).right_child != NULL:
                stack.append(Node.from_ptr(dereference(node.node).right_child))

    cdef _Node* _build_tree(self, Dataset data, size_t depth=0, np.float64_t loss=np.inf):
        # Should use a PQ to expand the nodes in decreasing order of H/Gini
        cdef SplitChoice split = self._find_best_split(data, loss)
        cdef _Node* ret = self._create_node(data.y, depth)
        if split is None:
            return ret
        if split.left_data.get_length() <= self.minobs or \
                split.right_data.get_length() <= self.minobs or \
                split.dloss < self.delta_loss or split.loss <= 0 or \
                self.nb_nodes > self.max_interaction_depth:
            return ret
        self.nb_nodes += 1
        ret.feature_idx = split.feature_idx
        ret.threshold = split.threshold
        _set_left_child(
            ret, self._build_tree(split.left_data, depth+1, split.loss_left)
        )
        _set_right_child(
            ret, self._build_tree(split.right_data, depth+1, split.loss_right)
        )
        cdef size_t _depth = dereference(ret).depth
        cdef bint   _kind = _is_root(ret)
        cdef int    _idx = dereference(ret).feature_idx
        cdef np.float64_t _threshold = dereference(ret).threshold
        cdef np.float64_t _loss = dereference(ret).loss
        cdef np.float64_t _avg = dereference(ret).avg_value
        cdef str kind = 'Node' if _kind else 'Leaf'
        print(f"{'  ' * _depth} {kind}, Depth: {_depth}, "
              f"Feature: {_idx}, Threshold: {_threshold}, Loss: {_loss}"
              f", Mean_value: {_avg}")
        return ret

    cdef _Node* _create_node(self, np.float64_t[:] ys, size_t depth):
        self.max_depth = max(depth, self.max_depth)
        cdef _Node* node = new_node(depth)
        _set_ys(node, np.mean(ys), self._loss(ys), ys.shape[0])
        return node

    cdef SplitChoice _find_best_split(self, Dataset data, np.float64_t precomputed_loss=np.inf):
        global PROBE
        cdef np.uint8_t[:] usable = np.ones(data.X.shape[1], dtype=np.uint8)
        cdef int j
        for j in range(usable.shape[0]):
            if usable[j] and not data.not_all_equal(j):
                usable[j] = False
        cdef np.ndarray covariates = np.where(usable)[0]
        cdef np.ndarray indices
        if covariates.shape[0] > self.nb_cov:
            indices = np.random.choice(covariates.shape[0], self.nb_cov, replace=False)
            covariates = covariates[indices]

        cdef np.float64_t current_loss
        if precomputed_loss == np.inf:
            current_loss = self._loss(data.y)
        else:
            current_loss = precomputed_loss
        cdef np.float64_t prop_p0 = np.mean(np.asarray(data.p) == 0)

        cdef np.ndarray values
        cdef Dataset left_data
        cdef Dataset right_data
        cdef np.float64_t loss_left
        cdef np.float64_t loss_right
        cdef np.float64_t threshold
        cdef np.float64_t prop_left_p0
        cdef np.float64_t prop_right_p0
        cdef int threshold_idx
        cdef int feature_idx

        cdef np.float64_t loss, dloss
        cdef np.float64_t best_dloss = 0.
        cdef size_t best_feature_idx = -1
        cdef np.float64_t best_threshold = 0

        cdef SplitChoice ret = None
        cdef np.ndarray sorted_indices
        cdef int base_idx

        for j in range(covariates.shape[0]):
            feature_idx = covariates[j]
            values = np.unique(data.X[:, feature_idx])
            base_idx = 0
            sorted_indices = np.argsort(data.X[:, feature_idx])
            # À faire: choisir le split des variables catégorielles
            # sur base des valeurs moyennes de y par modalité
            for threshold_idx in range(values.shape[0]-1):
                # extract this into a C function
                threshold = (values[threshold_idx] + values[threshold_idx+1]) / 2.
                start = time()
                base_idx = _masks(data.X[:, feature_idx], threshold, base_idx, sorted_indices)
                PROBE += time() - start
                left_data = data[:base_idx]
                right_data = data[base_idx:]

                if min(left_data.get_length(), right_data.get_length()) <= self.minobs:
                    loss = dloss = 0.
                    prop_left_p0 = prop_right_p0 = 0
                else:
                    loss_left = self._loss(left_data.y)
                    loss_right = self._loss(right_data.y)

                    prop_left_p0 = np.mean(np.asarray(left_data.p) == 0)
                    prop_right_p0 = np.mean(np.asarray(right_data.p) == 0)

                    loss = loss_left * left_data.get_length()
                    loss += loss_right * right_data.get_length()
                    dloss = current_loss - loss / data.get_length()

                    if dloss > best_dloss and \
                            fabs(prop_left_p0 - prop_p0) <= self.epsilon*prop_p0 and \
                            fabs(prop_right_p0 - prop_p0) <= self.epsilon*prop_p0:
                        best_dloss = dloss
                        ret = SplitChoice(
                            feature_idx, threshold, current_loss,
                            dloss, loss_left, loss_right,
                            left_data, right_data
                        )
        return ret

    def predict(self, X):
        cdef np.float64_t[:, :] data = self.data.transform(X)
        cdef int n = X.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1] ret = np.empty(n, dtype=np.float64)
        cdef int i
        for i in prange(n, nogil=True, schedule='runtime'):
            # Careful: stop search in the tree on unknown modality for category var.
            ret[i] = self._predict_instance(data[i, :])
        return ret

    cdef np.float64_t _predict_instance(self, np.float64_t[:] x) noexcept nogil:
        cdef _Node* node = self.root
        cdef np.float64_t val
        while _is_root(node):
            val = x[node.feature_idx]
            if val <= dereference(node).threshold:
                node = dereference(node).left_child
            else:
                node = dereference(node).right_child
        return dereference(node).avg_value

