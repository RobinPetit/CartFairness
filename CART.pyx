# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: language_level=3
# cython: linetrace=True

cimport numpy as np
from libc.math cimport log, fabs

import cython
from cython.operator cimport dereference
from cython.parallel import prange


ctypedef size_t Pyssize_t

cdef extern from "_CART.h" nogil:
    cdef struct Vector:
        void* _base
        size_t allocated
        size_t n
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
        np.float64_t dloss
        bint is_categorical
        Vector categorical_values_left
        Vector categorical_values_right
        int idx

    _Node* new_node(size_t)
    void clear_node(_Node*)
    void _set_ys(_Node*, size_t, double, double, size_t)
    void _set_categorical_node_left_right_values(_Node*, np.int32_t*, size_t, size_t)
    void _set_left_child(_Node*, _Node*)
    void _set_right_child(_Node*, _Node*)
    bint _is_root(_Node*)
    bint _is_leaf(_Node*)
    bint vector_contains_int32(Vector*, np.int32_t)


cdef class Node:
    cdef _Node* node

    @staticmethod
    cdef Node from_ptr(_Node* ptr):
        cdef Node ret = Node.__new__(Node)
        ret.node = ptr
        return ret

    def __eq__(self, Node other) -> bool:
        return self.node == other.node

    property feature_idx:
        def __get__(self):
            return dereference(self.node).feature_idx

    property threshold:
        def __get__(self):
            return dereference(self.node).threshold

    property loss:
        def __get__(self):
            return dereference(self.node).loss

    property dloss:
        def __get__(self):
            return self.node.dloss

    property avg_value:
        def __get__(self):
            return dereference(self.node).avg_value

    property depth:
        def __get__(self):
            return dereference(self.node).depth

    property parent:
        def __get__(self):
            return Node.from_ptr(dereference(self.node).parent)

    property left_child:
        def __get__(self):
            return Node.from_ptr(dereference(self.node).left_child)
        def __set__(self, Node value):
            dereference(self.node).left_child = value.node

    property right_child:
        def __get__(self):
            return Node.from_ptr(dereference(self.node).right_child)
        def __set__(self, Node value):
            dereference(self.node).right_child = value.node

    property kind:
        def __get__(self):
            if _is_leaf(self.node):
                return 'Leaf'
            elif _is_root(self.node):
                return 'Root'
            else:
                return 'Node'

    property nb_samples:
        def __get__(self):
            return dereference(self.node).nb_samples

    property is_categorical:
        def __get__(self):
            return self.node.is_categorical

    property index:
        def __get__(self):
            return self.node.idx

    property position:
        def __get__(self):
            if _is_root(self.node):
                return None
            elif self.node == self.node.parent.left_child:
                return 'left'
            else:
                return 'right'

    cpdef list get_left_modalities(self, Dataset data):
        if not self.node.is_categorical:
            raise ValueError('Not a categorical split')
        cdef list ret = []
        cdef size_t i
        cdef Vector* vec = &self.node.categorical_values_left
        for i in range(vec.n):
            ret.append(data._reverse_mapping[self.node.feature_idx][(<np.int32_t*>(vec._base))[i]])
        return ret

    cpdef list get_right_modalities(self, Dataset data):
        if not self.node.is_categorical:
            raise ValueError('Not a categorical split')
        cdef list ret = []
        cdef size_t i
        cdef Vector* vec = &self.node.categorical_values_right
        for i in range(vec.n):
            ret.append(data._reverse_mapping[self.node.feature_idx][(<np.int32_t*>(vec._base))[i]])
        return ret

    def __str__(self):
        return f'Node(idx={self.node.idx}, ptr={<long>(self.node):x})'
    def __repr__(self):
        return str(self)

import numpy as np
import pandas as pd

from time import time

PROBE = 0

class TODOError(ValueError):
    pass

ctypedef enum LossFunction:
    MSE,
    POISSON

cdef int indirect_binary_search(
        np.float64_t[:] values, np.float64_t threshold, int beg, int end,
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

cdef void _mask_categorical(
        np.float64_t[:] values, int value, np.uint8_t[:] out) noexcept nogil:
    cdef int i
    for i in prange(values.shape[0], nogil=True, schedule='runtime'):
        if <int>(values[i]) == value:
            out[i] = True

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

cdef void _extract_mean_ys(np.float64_t[:] X, np.float64_t[:] y,
                      np.float64_t[:] sums, np.int32_t[:] sizes) noexcept nogil:
    cdef int i
    cdef int value
    for i in prange(y.shape[0], nogil=True, schedule='runtime'):
        value = <int>(X[i])
        sums[value] += y[i]
        sizes[value] += 1

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
        ret._reverse_mapping = self._reverse_mapping
        return ret

    cpdef bint is_categorical(self, int feature_idx):
        return self._is_categorical[feature_idx]

    cdef void _labelize(
            self, np.ndarray[object, ndim=2] X, np.float64_t[:, :] out,
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

    cdef np.ndarray order_categorical(self, int feature_idx):
        cdef int size = len(self._reverse_mapping[feature_idx])
        cdef np.ndarray ysums = np.zeros(size, dtype=np.float64)
        cdef np.ndarray ysizes = np.zeros(size, dtype=np.int32)
        _extract_mean_ys(self.X[:, feature_idx], self.y[:], ysums, ysizes)
        cdef np.ndarray mask = np.where(ysizes > 0)[0]
        ysums[mask] /= ysizes[mask]
        cdef np.ndarray ret = np.argsort(ysums)
        ret = ret[ysizes[ret] > 0]  # TODO: optimize me
        return ret

@cython.final
cdef class SplitChoice:
    cdef size_t feature_idx
    cdef np.float64_t threshold
    cdef np.float64_t loss, dloss
    cdef np.float64_t loss_left, loss_right
    cdef Dataset left_data, right_data
    cdef bint is_categorical
    cdef size_t threshold_idx
    cdef np.int32_t[::1] labels  # Needs to be contiguous in memory!

    def __cinit__(self, size_t feature_idx, bint is_categorical,
                  np.float64_t loss, np.float64_t dloss,
                  np.float64_t left_loss, np.float64_t right_loss,
                  Dataset left_data, Dataset right_data,
                  np.float64_t threshold=np.inf,
                  int threshold_idx=0, np.int32_t[::1] ordered_labels=None):
        self.feature_idx = feature_idx
        self.loss = loss
        self.dloss = dloss
        self.left_data = left_data
        self.right_data = right_data
        self.loss_left = left_loss
        self.loss_right = right_loss
        self.is_categorical = is_categorical
        if self.is_categorical:
            self.threshold_idx = threshold_idx
            self.labels = ordered_labels
        else:
            self.threshold = threshold

    cdef bint is_better_than(self, SplitChoice other):
        return other is None or other.dloss < self.dloss

ctypedef enum _SplitType:
    BEST,
    DEPTH

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
    cdef int nb_splitting_nodes
    cdef bint pruning
    cdef Dataset data
    cdef _Node* root
    cdef _SplitType split_type

    cdef list all_nodes

    cdef int idx_nodes

    def __cinit__(self, epsilon=0., prop_root_p0=1.0, id=0, nb_cov=1,
                  replacement=False, prop_sample=1.0, frac_valid=0.2,
                  max_interaction_depth=0, max_depth=0, margin="absolute",
                  minobs=1, delta_loss=0, loss="MSE", name=None,
                  parallel="Yes", pruning="No", bootstrap="No",
                  split='depth'):
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
        self.nb_splitting_nodes = 0
        self.idx_nodes = 0
        self.max_interaction_depth = max_interaction_depth
        loss = loss.lower()
        LOSS_MAPPING = {
            'mse': MSE,
            'poisson': POISSON
        }
        assert loss in LOSS_MAPPING.keys()
        self.loss_fct = LOSS_MAPPING[loss]
        split = split.lower()
        if split == 'best':
            self.split_type = _SplitType.BEST
        elif split == 'depth':
            self.split_type = _SplitType.DEPTH
        else:
            raise ValueError('Unknown split type: ' + str(split))

    property nodes:
        def __get__(self):
            return self.all_nodes

    property max_interaction_depth:
        def __get__(self):
            return self.max_interaction_depth

    property minobs:
        def __get__(self):
            return self.minobs

    property margin:
        def __get__(self):
            return self.margin

    property epsilon:
        def __get__(self):
            return self.epsilon

    property delta_loss:
        def __get__(self):
            return self.delta_loss

    property loss:
        def __get__(self):
            return self.loss

    property nb_cov:
        def __get__(self):
            return self.nb_cov

    property idx_nodes:
        def __get__(self):
            return self.idx_nodes

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
        return self.nodes

    cdef void _retrieve_all_nodes(self):
        self.all_nodes = list()
        cdef Node node
        cdef list stack = []
        stack.append(Node.from_ptr(self.root))
        while len(stack) > 0:
            node = stack.pop()
            self.all_nodes.append(node)
            if dereference(node.node).left_child != NULL:
                stack.append(Node.from_ptr(dereference(node.node).left_child))
            if dereference(node.node).right_child != NULL:
                stack.append(Node.from_ptr(dereference(node.node).right_child))
        self.all_nodes.sort(key=lambda n: n.index)

    cdef _Node* _build_tree(self, Dataset data):
        if self.split_type == _SplitType.DEPTH:
            return self._build_tree_depth_first(data)
        elif self.split_type == _SplitType.BEST:
            return self._build_tree_best_first(data)

    cdef _Node* _build_tree_depth_first(self, Dataset data, size_t depth=0, np.float64_t loss=np.inf):
        # Should use a PQ to expand the nodes in decreasing order of H/Gini
        cdef SplitChoice split = self._find_best_split(data, loss)
        cdef _Node* ret = self._create_node(data.y, depth)
        ret.threshold = -1
        ret.feature_idx = -1
        self.nb_nodes += 1
        self.idx_nodes +=1
        ret.dloss = 0.
        if split is None or split.left_data.get_length() <= self.minobs or \
                split.right_data.get_length() <= self.minobs or \
                split.dloss < self.delta_loss or split.loss <= 0 or \
                self.nb_splitting_nodes > self.max_interaction_depth:
            return ret
        ret.feature_idx = split.feature_idx
        self.nb_splitting_nodes += 1
        ret.dloss = split.dloss
        if split.is_categorical:
            _set_categorical_node_left_right_values(
                ret, &split.labels[0], split.labels.shape[0], split.threshold_idx
            )
            ret.threshold = split.threshold_idx + .5
        else:
            ret.threshold = split.threshold
        cdef size_t _depth = dereference(ret).depth
        cdef bint   _kind = _is_root(ret)
        cdef int    _idx = dereference(ret).feature_idx
        cdef np.float64_t _threshold = dereference(ret).threshold
        cdef np.float64_t _loss = dereference(ret).loss
        cdef np.float64_t _avg = dereference(ret).avg_value
        cdef str kind = 'Node' if _kind else 'Leaf'
        print(f"{'  ' * _depth} {kind}, Depth: {_depth}, "
              f"Feature: {_idx}, Threshold: {_threshold}, DLoss: {ret.dloss}"
              f", Mean_value: {_avg}")
        _set_left_child(
            ret, self._build_tree_depth_first(split.left_data, depth+1, split.loss_left)
        )
        _set_right_child(
            ret, self._build_tree_depth_first(split.right_data, depth+1, split.loss_right)
        )
        return ret

    cdef _Node* _build_tree_best_first(self, Dataset data):
        pass

    cdef _Node* _create_node(self, np.float64_t[:] ys, size_t depth):
        self.max_depth = max(depth, self.max_depth)
        cdef _Node* node = new_node(depth)
        _set_ys(node, self.idx_nodes, np.mean(ys), self._loss(ys) * ys.shape[0], ys.shape[0])
        return node

    cdef SplitChoice _find_best_split(self, Dataset data, np.float64_t precomputed_loss=np.inf):
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
            current_loss = self._loss(data.y) * data.get_length()
        else:
            current_loss = precomputed_loss
        cdef np.float64_t prop_p0 = np.mean(np.asarray(data.p) == 0)

        cdef SplitChoice ret = None
        cdef SplitChoice best_split

        cdef int feature_idx = 0

        for j in range(covariates.shape[0]):
            feature_idx = covariates[j]
            best_split = self._find_best_threshold(
                data, feature_idx, current_loss, prop_p0
            )
            if best_split.is_better_than(ret):
                ret = best_split
        return ret

    cdef SplitChoice _find_best_threshold(
            self, Dataset data, int feature_idx,
            np.float64_t current_loss, np.float64_t prop_p0):
        if data.is_categorical(feature_idx):
            return self._find_best_threshold_categorical(
                data, feature_idx, current_loss, prop_p0
            )
        else:
            return self._find_best_threshold_numerical(
                data, feature_idx, current_loss, prop_p0
            )

    cdef SplitChoice _find_best_threshold_numerical(
            self, Dataset data, int feature_idx,
            np.float64_t current_loss, np.float64_t prop_p0):
        global PROBE
        cdef np.ndarray values = np.unique(data.X[:, feature_idx])
        cdef int base_idx = 0
        cdef np.ndarray sorted_indices = np.argsort(data.X[:, feature_idx])

        cdef Dataset left_data
        cdef Dataset right_data

        cdef np.float64_t loss_left
        cdef np.float64_t loss_right
        cdef np.float64_t threshold
        cdef np.float64_t prop_left_p0
        cdef np.float64_t prop_right_p0
        cdef np.float64_t loss, dloss

        cdef SplitChoice ret  = None
        cdef SplitChoice split

        for threshold_idx in range(values.shape[0]-1):
            # extract this into a C function
            threshold = (values[threshold_idx] + values[threshold_idx+1]) / 2.
            start = time()
            base_idx = _masks(data.X[:, feature_idx], threshold, base_idx, sorted_indices)
            PROBE += time() - start
            left_data = data[sorted_indices[:base_idx]]
            right_data = data[sorted_indices[base_idx:]]

            if min(left_data.get_length(), right_data.get_length()) <= self.minobs:
                loss = dloss = 0.
                prop_left_p0 = prop_right_p0 = 0
            else:
                loss_left = self._loss(left_data.y) * left_data.get_length()
                loss_right = self._loss(right_data.y) * right_data.get_length()

                prop_left_p0 = np.mean(np.asarray(left_data.p) == 0)
                prop_right_p0 = np.mean(np.asarray(right_data.p) == 0)

                loss = loss_left + loss_right
                dloss = current_loss - loss #/ data.get_length()

                if fabs(prop_left_p0 - prop_p0) > self.epsilon*prop_p0 or \
                        fabs(prop_right_p0 - prop_p0) > self.epsilon*prop_p0:
                    continue
                split = SplitChoice(
                    feature_idx, False, current_loss,
                    dloss, loss_left, loss_right,
                    left_data, right_data,
                    threshold=threshold
                )
                if split.is_better_than(ret):
                    ret = split
        return ret

    cdef _find_best_threshold_categorical(
            self, Dataset data, int feature_idx,
            np.float64_t current_loss, np.float64_t prop_p0):
        global PROBE
        cdef np.ndarray ordered = data.order_categorical(feature_idx).astype(np.int32)
        cdef np.ndarray goes_left = np.zeros(data.get_length(), dtype=bool)
        cdef threshold_idx
        cdef np.float64_t[:] values = data.X[:, feature_idx]

        cdef Dataset left_data
        cdef Dataset right_data

        cdef np.float64_t loss_left
        cdef np.float64_t loss_right
        cdef np.float64_t threshold
        cdef np.float64_t prop_left_p0
        cdef np.float64_t prop_right_p0
        cdef np.float64_t loss, dloss

        cdef SplitChoice ret  = None
        cdef SplitChoice split
        cdef int threshold_value
        for threshold_idx in range(ordered.shape[0]-1):
            threshold_value = ordered[threshold_idx]
            _mask_categorical(values, threshold_value, goes_left)
            left_data = data[goes_left]
            right_data = data[~goes_left]

            if min(left_data.get_length(), right_data.get_length()) <= self.minobs:
                loss = dloss = 0.
                prop_left_p0 = prop_right_p0 = 0
            else:
                loss_left = self._loss(left_data.y) * left_data.get_length()
                loss_right = self._loss(right_data.y) * right_data.get_length()

                prop_left_p0 = np.mean(np.asarray(left_data.p) == 0)
                prop_right_p0 = np.mean(np.asarray(right_data.p) == 0)

                loss = loss_left + loss_right
                dloss = current_loss - loss #/ data.get_length()

                if fabs(prop_left_p0 - prop_p0) > self.epsilon*prop_p0 or \
                        fabs(prop_right_p0 - prop_p0) > self.epsilon*prop_p0:
                    continue
                split = SplitChoice(
                    feature_idx, True, current_loss,
                    dloss, loss_left, loss_right,
                    left_data, right_data,
                    threshold_idx=threshold_idx,
                    ordered_labels=ordered
                )
                if split.is_better_than(ret):
                    ret = split
        return ret


    def predict(self, X):
        cdef np.float64_t[:, :] data = self.data.transform(X)
        cdef int n = X.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1] ret = np.empty(n, dtype=np.float64)
        cdef int i
        for i in prange(n, nogil=True, schedule='runtime'):
            ret[i] = self._predict_instance(data[i, :])
        return ret

    cdef np.float64_t _predict_instance(self, np.float64_t[:] x) noexcept nogil:
        cdef _Node* node = self.root
        cdef np.float64_t val
        cdef int categorical_label
        while not _is_leaf(node):
            if node.is_categorical:
                categorical_label = <int>(x[node.feature_idx])
                if vector_contains_int32(&node.categorical_values_left, categorical_label):
                    node = node.left_child
                elif vector_contains_int32(&node.categorical_values_right, categorical_label):
                    node = node.right_child
                else:
                    # Stop search if modality of categorical value has never been
                    # seen at this point
                    break
            else:
                val = x[node.feature_idx]
                if val <= node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
        return node.avg_value

    def compute_importance(self):
        # TODO
        pass

