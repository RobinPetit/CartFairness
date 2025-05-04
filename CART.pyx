# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: language_level=3
# cython: linetrace=True

cimport numpy as np
from libc.math cimport fabs, INFINITY

import cython
from cython.parallel import prange
from cpython.ref cimport PyObject,  Py_XINCREF, Py_XDECREF

from loss cimport Loss, LossFunction
from dataset cimport Dataset

# include "loss.pyx"
# include "dataset.pyx"

@cython.final
cdef class Node:
    @staticmethod
    cdef Node from_ptr(_Node* ptr):
        cdef Node ret = Node.__new__(Node)
        ret.node = ptr
        return ret

    def __eq__(self, Node other) -> bool:
        return self.node == other.node

    property feature_idx:
        def __get__(self):
            cdef int ret
            if self.node.feature_idx == <size_t>-1:
                ret = -1
            else:
                ret = <int>self.node.feature_idx
            return ret

    property threshold:
        def __get__(self):
            return self.node.threshold

    property loss:
        def __get__(self):
            return self.node.loss

    property dloss:
        def __get__(self):
            return self.node.dloss

    property prop_p0:
        def __get__(self):
            return self.node.prop_p0

    property avg_value:
        def __get__(self):
            return self.node.avg_value

    property depth:
        def __get__(self):
            return self.node.depth

    property parent:
        def __get__(self):
            return Node.from_ptr(self.node.parent)

    property left_child:
        def __get__(self):
            return Node.from_ptr(self.node.left_child)

    property right_child:
        def __get__(self):
            return Node.from_ptr(self.node.right_child)

    property is_leaf:
        def __get__(self):
            return _is_leaf(self.node)

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
            return self.node.nb_samples

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
            idx = (<np.int32_t*>(vec._base))[i]
            ret.append(data._reverse_mapping[self.node.feature_idx][idx])
        return ret

    cpdef list get_right_modalities(self, Dataset data):
        if not self.node.is_categorical:
            raise ValueError('Not a categorical split')
        cdef list ret = []
        cdef size_t i
        cdef Vector* vec = &self.node.categorical_values_right
        cdef int idx
        for i in range(vec.n):
            idx = (<np.int32_t*>(vec._base))[i]
            ret.append(data._reverse_mapping[self.node.feature_idx][idx])
        return ret

    cpdef list _get_left_indices(self):
        if not self.node.is_categorical:
            raise ValueError('Not a categorical split')
        cdef list ret = []
        cdef size_t i
        cdef Vector* vec = &self.node.categorical_values_left
        for i in range(vec.n):
            ret.append((<np.int32_t*>(vec._base))[i])
        return ret

    cpdef list _get_right_indices(self):
        if not self.node.is_categorical:
            raise ValueError('Not a categorical split')
        cdef list ret = []
        cdef size_t i
        cdef Vector* vec = &self.node.categorical_values_right
        for i in range(vec.n):
            ret.append((<np.int32_t*>(vec._base))[i])
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

cdef int _masks(
        np.float64_t[::1] values, np.float64_t threshold,
        int base_idx) noexcept nogil:
    cdef int beg = base_idx
    cdef int mid
    cdef int end = values.shape[0]
    while beg < end:
        mid = (beg + end) // 2
        if values[mid] <= threshold:
            beg = mid+1
        else:
            end = mid
    return beg

cdef void augment_p0_counts(
        double[::1] p, double[::1] w, double* sum_left, double* sum_right,
        double* sum_weights_left, double* sum_weights_right
        ) noexcept nogil:
    cdef double _sum_left = sum_left[0]
    cdef double _sum_right = sum_right[0]
    cdef double _sum_weights_left = sum_weights_left[0]
    cdef double _sum_weights_right = sum_weights_right[0]
    cdef int i
    for i in range(p.shape[0]):
        if p[i] == 0:
            _sum_left += w[i]
            _sum_right -= w[i]
        _sum_weights_left += w[i]
        _sum_weights_right -= w[i]
    sum_left[0] = _sum_left
    sum_right[0] = _sum_right
    sum_weights_left[0] = _sum_weights_left
    sum_weights_right[0] = _sum_weights_right

@cython.final
cdef class SplitChoice:
    def __cinit__(self, size_t feature_idx, bint is_categorical,
                  np.float64_t loss, np.float64_t dloss,
                  np.float64_t left_loss, np.float64_t right_loss,
                  Dataset left_data, Dataset right_data,
                  np.float64_t threshold=np.inf,
                  # for categorical splits
                  int threshold_idx=0, np.int32_t[::1] ordered_labels=None,
                  np.int32_t[::1] first_index=None):
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
            self.first_index = first_index
        else:
            self.threshold = threshold

    cdef bint is_better_than(self, SplitChoice other):
        return other is None or self.dloss - other.dloss > 1e-10

cdef void _compute_first_idx(
        np.float64_t[::1] Xs, np.float64_t[::1] ys,  # in
        np.int32_t[::1] indices, np.float64_t[::1] mean_ys  # out
        ) noexcept nogil:
    cdef int i
    for i in range(ys.shape[0]):
        indices[<int>(Xs[i])+1] += 1
        mean_ys[<int>(Xs[i])] += ys[i]
    for i in range(indices.shape[0]-1):
        mean_ys[i] /= indices[i+1]
    for i in range(2, indices.shape[0]):
        indices[i] += indices[i-1]

cdef void _select_indices(
        np.int32_t[::1] sorted_indices, np.int32_t[::1] ordered, size_t max_idx,
        np.int32_t[::1] first_idx, np.npy_bool[::1] selected) noexcept nogil:
    cdef size_t i
    cdef np.int32_t j
    for i in range(max_idx+1):
        for j in range(first_idx[ordered[i]], first_idx[ordered[i]+1]):
            selected[sorted_indices[j]] = True

@cython.final
cdef class __SortedFeatureData:
    def __cinit__(self, Dataset data, int feature_idx):
        self.indices = np.argsort(data.X[:, feature_idx]).astype(np.int32)
        self.X = np.asarray(data.X)[self.indices, feature_idx]
        self.y = np.asarray(data.y)[self.indices]
        self.w = np.asarray(data.w)[self.indices]
        self.p = np.asarray(data.p)[self.indices]

    cdef inline np.float64_t get_max(self) noexcept nogil:
        return self.X[self.X.shape[0]-1]

MAX_NB_MODALITIES = 38

cdef inline bint _as_bool(x: bool | str, name: str):
    if isinstance(x, str):
        x = x.lower()
        if x not in ('yes', 'no'):
            raise ValueError(f'Unknown value "{x} for attribute {name}')
        return x == 'yes'
    else:
        return bool(x)

@cython.final
cdef class CART:
    def __init__(
            self,
            *,
            epsilon=1.,
            prop_root_p0=-1.,
            id=0,
            nb_cov=1,  # TODO: handle a proportion rather than an integer
            replacement=False,  # when could this be False?
            prop_sample=1.0,
            frac_valid=0.2,
            max_interaction_depth=0,
            max_depth=<size_t>(-1),
            margin="absolute",
            minobs=1,
            loss="MSE",
            name=None,
            parallel="Yes",
            pruning="No",
            bootstrap="No",  # TODO: use only prop_sample
            split='best',
            min_nb_new_instances=1,
            normalized_loss=False,  # TODO: do better
            exact_categorical_splits=False,
            verbose=True
    ):
        self.prop_p0 = prop_root_p0
        self.bootstrap = _as_bool(bootstrap, 'bootstrap')
        self.pruning = _as_bool(pruning, 'pruning')
        self.relative_margin = (margin == 'relative')
        self.normalized_loss = normalized_loss
        self.replacement = replacement
        self.epsilon = epsilon
        self.nb_cov = nb_cov
        self.id = id
        self.prop_sample = prop_sample
        self.minobs = minobs
        self.max_depth = max_depth
        self.depth = 0
        self.nb_nodes = 0
        self.nb_splitting_nodes = 0
        self.idx_nodes = 0
        self.max_interaction_depth = max_interaction_depth
        loss = loss.lower()
        LOSS_MAPPING = {
            'mse': LossFunction.MSE,
            'poisson': LossFunction.POISSON,
            'gamma': LossFunction.GAMMA
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
        self.min_nb_new_instances = min_nb_new_instances
        self.exact_categorical_splits = exact_categorical_splits
        self.fitted = False
        self._verbose = verbose

    def __getstate__(self):
        return {
            'bootstrap':                self.bootstrap,
            'replacement':              self.replacement,
            'normalized_loss':          self.normalized_loss,
            'pruning':                  self.pruning,
            'relative_margin':          self.relative_margin,
            'exact_categorical_splits': self.exact_categorical_splits,
            'fitted':                   self.fitted,
            'verbose':                 self._verbose,
            'epsilon':                  self.epsilon,
            'prop_sample':              self.prop_sample,
            'prop_margin':              self.prop_margin,
            'prop_p0':                  self.prop_p0,
            'prop_root_p0':             self.prop_root_p0,
            'nb_cov':                   self.nb_cov,
            'id':                       self.id,
            'minobs':                   self.minobs,
            'depth':                    self.depth,
            'max_depth':                self.max_depth,
            'interaction_depth':        self.max_interaction_depth,
            # 'nb_nodes': nb_nodes,
            'nb_splitting_nodes':       self.nb_splitting_nodes,
            'min_nb_new_instances':     self.min_nb_new_instances,
            # 'idx_nodes': idx_nodes,
            'root':                     <unsigned long long>(<void*>self.root),
            'data':                     self.data,
            'split_type':               self.split_type,
            'loss_fct':                 self.loss_fct,
            # 'all_nodes': all_nodes
        }

    def __setstate__(self, data):
        self.bootstrap = data['bootstrap']
        self.replacement = data['replacement']
        self.normalized_loss = data['normalized_loss']
        self.pruning = data['pruning']
        self.relative_margin = data['relative_margin']
        self.exact_categorical_splits = data['exact_categorical_splits']
        self.fitted = data['fitted']
        self._verbose = data['verbose']
        self.epsilon = data['epsilon']
        self.prop_sample = data['prop_sample']
        self.prop_margin = data['prop_margin']
        self.prop_p0 = data['prop_p0']
        self.prop_root_p0 = data['prop_root_p0']
        self.nb_cov = data['nb_cov']
        self.id = data['id']
        self.minobs = data['minobs']
        self.depth = data['depth']
        self.max_depth = data['max_depth']
        self.max_interaction_depth = data['interaction_depth']
        self.nb_splitting_nodes = data['nb_splitting_nodes']
        self.min_nb_new_instances = data['min_nb_new_instances']
        self.root = <_Node*>(<unsigned long long>(data['root']))
        self.data = data['data']
        self.split_type = data['split_type']
        self.loss_fct = data['loss_fct']

    def fit(self, dataset: Dataset,
            np.ndarray[np.float64_t, ndim=1] samples_weights=None):
        global PROBE
        PROBE = 0
        start = time()
        self.data = dataset
        if self.bootstrap:
            if self._verbose:
                print('Bootstrapping...')
            self.data = self.data.sample(self.prop_sample, self.replacement)
        # split train vs test ?!
        if self.prop_p0 < 0:
            self.prop_root_p0 = np.average(
                1-np.asarray(self.data.p),
                weights=np.asarray(self.data.w)
            )
        self.prop_margin = self.epsilon
        if self.relative_margin:
            self.prop_margin *= self.prop_root_p0
        if self.exact_categorical_splits:
            for j in range(dataset.nb_features):
                if not dataset.is_categorical(j):
                    continue
                if dataset.nb_modalities_of(j) > MAX_NB_MODALITIES:
                    raise ValueError(
                        'Unable to perform exact categorical split on '
                        f'covariate {j} that has {dataset.nb_modalities_of(j)} '
                        f'> {MAX_NB_MODALITIES} modalities'
                    )
        self.all_nodes = []
        self.root = self._build_tree(self.data)
        self.fitted = True
        if self.pruning:
            raise TODOError()
        time_elapsed = time() - start
        if self._verbose:
            print("\n")
            print('*******************************')
            print(f"Tree {self.id}: Params(id={self.max_interaction_depth}, cov={self.nb_cov})")
            print(f"Time elapsed: {time_elapsed}")
            print(f"Tree depth:{self.depth}")
            print(f"Nb nodes: {len(self.nodes)}")
            print('*******************************')
            print(f'\t\t{100 * PROBE / time_elapsed:3.2f}%')
        return self.nodes

    def predict(self, X):
        if not self.fitted:
            raise ValueError('Predicting on non-trained DecisionTree')
        cdef np.float64_t[:, ::1] data
        if X.dtype.kind.lower()== 'o':
            data = self.data.transform(X)
        else:
            data = np.ascontiguousarray(X, dtype=np.float64)
        cdef int n = X.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1] ret = np.empty(n, dtype=np.float64)
        cdef int i
        for i in prange(n, nogil=True, schedule='runtime'):
            ret[i] = self._predict_instance(data[i, :])
        return ret

    ########## Cython

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

    property loss:
        def __get__(self):
            return self.loss

    property nb_cov:
        def __get__(self):
            return self.nb_cov

    property idx_nodes:
        def __get__(self):
            return self.idx_nodes

    property depth:
        def __get__(self):
            return self.depth

    def __dealloc__(self):
        if self.root != NULL:
            clear_node(self.root)

    cdef np.float64_t _loss(self, np.float64_t[::1] ys, np.float64_t[::1] ws):
        cdef Loss loss = Loss(self.loss_fct, self.normalized_loss)
        loss.augment(ys, ws)
        return loss.get()

    cdef _Node* _build_tree(self, Dataset data):
        if self.split_type == _SplitType.DEPTH:
            return self._build_tree_depth_first(data)
        elif self.split_type == _SplitType.BEST:
            return self._build_tree_best_first(data)
        cdef Node node
        for node in self.all_nodes:
            Py_XDECREF(<PyObject*>node.node.extra_data)
            node.node.extra_data = NULL

    cdef _Node* _build_tree_depth_first(
            self, Dataset data, size_t depth=0, np.float64_t loss=np.inf):
        cdef SplitChoice split = self._find_best_split(data, loss)
        cdef _Node* ret = self._create_node(data, depth, split.loss)
        if (
            split is None or
            split.left_data.size() <= self.minobs or
            split.right_data.size() <= self.minobs or
            split.dloss <= 0 or split.loss <= 0 or
            self.nb_splitting_nodes > self.max_interaction_depth
        ):
            if np.isinf(loss):
                ret.loss = self._loss(data.y, data.w)
            else:
                ret.loss = loss
            ret.dloss = 0.
            ret.feature_idx = -1
            return ret
        ret.feature_idx = split.feature_idx
        self.nb_splitting_nodes += 1
        ret.dloss = split.dloss
        if split.is_categorical:
            _set_categorical_node_left_right_values(
                ret, &split.labels[0], split.labels.shape[0],
                split.threshold_idx, &split.first_index[0]
            )
            ret.threshold = split.threshold_idx + .5
        else:
            ret.threshold = split.threshold
        if self._verbose:
            print(f"{'  ' * ret.depth} Node ({ret.idx}), "
                  f"Depth: {ret.depth}, "
                  f"Feature: {ret.feature_idx}, "
                  f"Threshold: {ret.threshold}, DLoss: {ret.dloss}"
                  f", Mean_value: {ret.avg_value},  N={ret.nb_samples}")
        if depth < self.max_depth:
            _set_left_child(
                ret, self._build_tree_depth_first(
                    split.left_data, depth+1, split.loss_left
                )
            )
            _set_right_child(
                ret, self._build_tree_depth_first(
                    split.right_data, depth+1, split.loss_right
                )
            )
        return ret

    cdef _Node* _build_tree_best_first(self, Dataset data):
        cdef _Node* node = self._create_node(data, 0)
        cdef _Node* ret = node
        cdef _Node* left
        cdef _Node* right
        cdef NodePq_t pq
        init_node_pq(&pq, self.max_interaction_depth)
        pq_insert(&pq, node)
        cdef SplitChoice
        while (
                self.nb_splitting_nodes <= self.max_interaction_depth and
                not pq_empty(&pq)
        ):
            node = pq_pop(&pq)
            if node.depth >= self.max_depth:
                continue
            split = self._find_best_split(
                <Dataset>(<object>node.extra_data), node.loss
            )
            if split is None:
                continue
            node.feature_idx = split.feature_idx
            node.dloss = split.dloss
            if split.is_categorical:
                _set_categorical_node_left_right_values(
                    node, &split.labels[0], split.labels.shape[0],
                    split.threshold_idx, &split.first_index[0]
                )
                node.threshold = split.threshold_idx + .5
            else:
                node.threshold = split.threshold
            Py_XDECREF(<PyObject*>node.extra_data)
            node.extra_data = NULL
            self.nb_splitting_nodes += 1
            left = self._create_node(
                split.left_data, node.depth+1, split.loss_left
            )
            right = self._create_node(
                split.right_data, node.depth+1, split.loss_right
            )
            _set_left_child(node, left)
            _set_right_child(node, right)
            pq_insert(&pq, left)
            pq_insert(&pq, right)
            if self._verbose:
                print(f"{'  ' * node.depth} {'Leaf' if _is_leaf(node) else 'Node'} "
                      f"({node.idx}), Depth: {node.depth}, "
                      f"Feature: {node.feature_idx}, "
                      f"Threshold: {node.threshold}, DLoss: {node.dloss}"
                      f", Mean_value: {node.avg_value},  N={node.nb_samples}")
        while not pq_empty(&pq):
            node = pq_pop(&pq)
            Py_XDECREF(<PyObject*>node.extra_data)
            node.extra_data = NULL
        return ret

    cdef _Node* _create_node(
            self, Dataset data, size_t depth, np.float64_t loss=INFINITY):
        self.depth = max(depth, self.depth)
        cdef _Node* node = new_node(depth)
        if loss == np.inf:
            loss = self._loss(data.y, data.w)
        _set_ys_ps(
            node, self.idx_nodes, np.mean(data.y), loss,
            data.size(), data.get_prop_p0()
        )
        node.dloss = 0
        node.threshold = -1
        node.feature_idx = -1
        node.extra_data = <PyObject*>data
        Py_XINCREF(<PyObject*>node.extra_data)
        self.all_nodes.append(Node.from_ptr(node))
        self.nb_nodes += 1
        self.idx_nodes += 1
        return node

    cdef SplitChoice _find_best_split(
            self, Dataset data, np.float64_t precomputed_loss=np.inf):
        cdef np.uint8_t[::1] usable = np.ones(data.X.shape[1], dtype=bool)
        cdef int j
        for j in range(usable.shape[0]):
            if usable[j] and not data.not_all_equal(j):
                usable[j] = False
        cdef np.ndarray covariates = np.where(usable)[0]
        cdef np.ndarray indices
        if <size_t>covariates.shape[0] > self.nb_cov:
            indices = np.random.choice(covariates.shape[0], self.nb_cov, replace=False)
            covariates = covariates[indices]

        cdef np.float64_t current_loss
        if precomputed_loss == np.inf:
            current_loss = self._loss(data.y, data.w)
        else:
            current_loss = precomputed_loss
        cdef np.float64_t prop_p0 = self.prop_root_p0

        cdef SplitChoice ret = None
        cdef SplitChoice best_split

        cdef int feature_idx = 0

        for j in range(covariates.shape[0]):
            feature_idx = covariates[j]
            best_split = self._find_best_threshold(
                data, feature_idx, current_loss, prop_p0
            )
            if best_split is not None and best_split.is_better_than(ret):
                ret = best_split
        return ret

    cdef SplitChoice _find_best_threshold(
            self, Dataset data, int feature_idx,
            np.float64_t current_loss, np.float64_t prop_p0):
        global PROBE
        cdef SplitChoice ret
        if data.is_categorical(feature_idx):
            ret = self._find_best_threshold_categorical(
                data, feature_idx, current_loss, prop_p0
            )
        else:
            start = time()
            ret = self._find_best_threshold_numerical(
                data, feature_idx, current_loss, prop_p0
            )
            PROBE += time() - start
        return ret

    cdef inline void _update_losses(
            self, Loss loss_left, Loss loss_right,
            np.float64_t[::1] ys, np.float64_t[::1] ws) noexcept nogil:
        loss_left.augment(ys, ws)
        loss_right.diminish(ys, ws)

    cdef SplitChoice _find_best_threshold_numerical(
            self, Dataset data, int feature_idx,
            np.float64_t current_loss, np.float64_t prop_p0):
        global PROBE
        cdef np.float64_t[::1] values = np.unique(data.X[:, feature_idx])
        cdef size_t base_idx = 0
        cdef size_t prev_base_idx = 0
        cdef int threshold_idx
        cdef __SortedFeatureData sorted_data = __SortedFeatureData(
            data, feature_idx
        )
        cdef size_t nb_samples = data.size()
        cdef double sum_of_weights = np.sum(sorted_data.w)

        cdef np.float64_t threshold
        cdef np.float64_t prop_left_p0
        cdef np.float64_t prop_right_p0
        cdef np.float64_t dloss
        cdef Loss loss_left = Loss(self.loss_fct, self.normalized_loss)
        cdef Loss loss_right = Loss(self.loss_fct, self.normalized_loss)
        loss_right.augment(data.y, data.w)

        cdef np.float64_t sum_p0_left = 0
        cdef np.float64_t sum_p0_right = nb_samples - np.sum(data.p)
        cdef np.float64_t sum_weights_p0_left = 0.
        cdef np.float64_t sum_weights_p0_right = sum_of_weights

        cdef SplitChoice ret  = None

        cdef double best_loss_left = 0
        cdef double best_loss_right = 0
        cdef double best_dloss = 0
        cdef size_t best_base_idx = 0
        cdef double best_threshold = 0

        with nogil:  # Yeepee! It is all nogil!
            for threshold_idx in range(values.shape[0]-1):
                threshold = (values[threshold_idx] + values[threshold_idx+1])/2
                base_idx = _masks(sorted_data.X, threshold, base_idx)
                if base_idx <= self.minobs:
                    continue
                if nb_samples-base_idx <= self.minobs:
                    break
                if base_idx - prev_base_idx < self.min_nb_new_instances:
                    continue
                augment_p0_counts(
                    sorted_data.p[prev_base_idx:base_idx],
                    sorted_data.w[prev_base_idx:base_idx],
                    &sum_p0_left, &sum_p0_right,
                    &sum_weights_p0_left, &sum_weights_p0_right
                )
                prop_left_p0 = sum_p0_left / sum_weights_p0_left
                prop_right_p0 = sum_p0_right / sum_weights_p0_right
                self._update_losses(
                    loss_left, loss_right,
                    sorted_data.y[prev_base_idx:base_idx],
                    sorted_data.w[prev_base_idx:base_idx]
                )
                prev_base_idx = base_idx
                dloss = current_loss - (loss_left.get() + loss_right.get())

                if fabs(prop_left_p0 - prop_p0) > self.prop_margin or \
                        fabs(prop_right_p0 - prop_p0) > self.prop_margin:
                    continue
                if dloss > best_dloss:
                    best_dloss = dloss
                    best_loss_left = loss_left.get()
                    best_loss_right = loss_right.get()
                    best_base_idx = base_idx
                    best_threshold = threshold
        if best_dloss == 0:
            return None
        return SplitChoice(
            feature_idx, False, current_loss,
            best_dloss, best_loss_left, best_loss_right,
            data[sorted_data.indices[:best_base_idx]],
            data[sorted_data.indices[best_base_idx:]],
            threshold=best_threshold,
            threshold_idx=best_threshold-.5
        )

    cdef SplitChoice _find_best_threshold_categorical(
            self, Dataset data, int feature_idx,
            np.float64_t current_loss, np.float64_t prop_p0):
        if self.exact_categorical_splits:
            return self._find_best_threshold_categorical_exact(
                data, feature_idx, current_loss, prop_p0
            )
        else:
            return self._find_best_threshold_categorical_sorted_by_mean_ys(
                data, feature_idx, current_loss, prop_p0
            )

    cdef SplitChoice _find_best_threshold_categorical_exact(
            self, Dataset data, int feature_idx,
            np.float64_t current_loss, np.float64_t prop_p0):
        cdef __SortedFeatureData sorted_data = __SortedFeatureData(
            data, feature_idx
        )
        cdef size_t nb_samples = data.size()
        cdef int max_modality = <int>sorted_data.get_max()
        cdef np.int32_t[::1] first_idx = np.zeros(max_modality+2, dtype=np.int32)
        cdef np.float64_t[::1]mean_ys = np.zeros(max_modality+1, np.float64)
        _compute_first_idx(sorted_data.X, sorted_data.y, first_idx, mean_ys)
        cdef size_t nb_modalities = 0
        cdef np.int32_t[::1] mapping = np.empty(max_modality+1, dtype=np.int32)
        cdef int j = 0
        while j+1 < first_idx.shape[0]:
            while j+1 < first_idx.shape[0] and first_idx[j] == first_idx[j+1]:
                j += 1
            mapping[nb_modalities] = j
            nb_modalities += 1
            j += 1
        mapping = mapping[:nb_modalities]
        if nb_modalities <= 1:
            return None
        if nb_modalities > MAX_NB_MODALITIES:
            raise ValueError(
                f'Unable to perform exact split on {nb_modalities} modalities'
            )
        cdef PartitionResult_t result
        result = find_best_partition(
            self.loss_fct, nb_modalities,
            &sorted_data.y[0], &sorted_data.w[0], &sorted_data.p[0],
            &first_idx[0], &mapping[0],
            self.prop_root_p0, self.prop_margin, self.minobs
        )
        if np.isinf(result.total_loss):
            return None
        cdef np.ndarray goes_left  = np.zeros(nb_samples, dtype=bool)
        cdef np.ndarray goes_right = np.zeros(nb_samples, dtype=bool)
        cdef size_t i
        cdef int beg, end
        cdef int nb_modalities_left = 0
        cdef int l = 0
        cdef int r = nb_modalities-1
        cdef np.int32_t[::1] ordered = np.zeros(nb_modalities, dtype=np.int32)
        for i in range(nb_modalities):
            beg = first_idx[mapping[i]]
            end = first_idx[mapping[i]+1]
            if result.mask&1:
                nb_modalities_left += 1
                goes_left[sorted_data.indices[beg:end]]  = True
                ordered[l] = mapping[i]
                l += 1
            else:
                goes_right[sorted_data.indices[beg:end]] = True
                ordered[r] = mapping[i]
                r -= 1
            result.mask >>= 1
        cdef Dataset left_data  = data[goes_left]
        cdef Dataset right_data = data[goes_right]
        return SplitChoice(
            feature_idx, True, current_loss,
            current_loss - result.total_loss,
            result.loss_left, result.loss_right,
            left_data, right_data,
            threshold_idx=nb_modalities_left,
            ordered_labels=ordered,
            first_index=first_idx
        )

    cdef SplitChoice _find_best_threshold_categorical_sorted_by_mean_ys(
            self, Dataset data, int feature_idx,
            np.float64_t current_loss, np.float64_t prop_p0):
        cdef __SortedFeatureData sorted_data = __SortedFeatureData(
            data, feature_idx
        )
        cdef int max_modality = <int>sorted_data.get_max()
        cdef int idx
        cdef np.int32_t[::1] first_idx = np.zeros(max_modality+2, dtype=np.int32)
        cdef np.float64_t[::1] mean_ys = np.zeros(max_modality+1, np.float64)
        _compute_first_idx(sorted_data.X, sorted_data.y, first_idx, mean_ys)
        cdef np.int32_t[::1] ordered = np.argsort(mean_ys).astype(np.int32)
        cdef size_t nb_samples = data.size()
        cdef double sum_of_weights = np.sum(sorted_data.w)

        cdef np.float64_t prop_left_p0
        cdef np.float64_t prop_right_p0
        cdef np.float64_t dloss = 0
        cdef Loss loss_left = Loss(self.loss_fct, self.normalized_loss)
        cdef Loss loss_right = Loss(self.loss_fct, self.normalized_loss)
        loss_right.augment(data.y, data.w)

        cdef np.float64_t sum_p0_left = 0
        cdef np.float64_t sum_p0_right = nb_samples - np.sum(data.p)
        cdef np.float64_t sum_weights_p0_left = 0.
        cdef np.float64_t sum_weights_p0_right = sum_of_weights
        cdef double best_loss_left = 0
        cdef double best_loss_right = 0
        cdef double best_dloss = 0
        cdef size_t best_idx = 0
        cdef int threshold_idx
        cdef size_t nb_left = 0
        cdef size_t nb_added_left = 0
        cdef size_t beg_idx, end_idx
        with nogil:  # Hell yeah baby!
            for idx in range(max_modality):
                threshold_idx = ordered[idx]
                beg_idx = first_idx[threshold_idx]
                end_idx = first_idx[threshold_idx+1]
                nb_added_left = end_idx - beg_idx
                nb_left += nb_added_left
                augment_p0_counts(
                    sorted_data.p[beg_idx:end_idx],
                    sorted_data.w[beg_idx:end_idx],
                    &sum_p0_left, &sum_p0_right,
                    &sum_weights_p0_left, &sum_weights_p0_right
                )
                self._update_losses(
                    loss_left, loss_right,
                    sorted_data.y[beg_idx:end_idx],
                    sorted_data.w[beg_idx:end_idx]
                )
                if nb_added_left < self.min_nb_new_instances:
                    continue
                if nb_samples - nb_left <= self.minobs:
                    break
                dloss = current_loss - (loss_left.get() + loss_right.get())
                prop_left_p0 = sum_p0_left / sum_weights_p0_left
                prop_right_p0 = sum_p0_right / sum_weights_p0_right
                if fabs(prop_left_p0 - prop_p0) > self.prop_margin or \
                        fabs(prop_right_p0 - prop_p0) > self.prop_margin:
                    continue
                if nb_left <= self.minobs:
                    continue
                if dloss > best_dloss:
                    best_dloss = dloss
                    best_loss_left = loss_left.get()
                    best_loss_right = loss_right.get()
                    best_idx = idx
        if dloss == 0:
            return None
        cdef np.ndarray[np.npy_bool, ndim=1] selected_indices = np.zeros(
            sorted_data.indices.shape[0], dtype=bool
        )
        _select_indices(
            sorted_data.indices, ordered, best_idx, first_idx, selected_indices
        )
        return SplitChoice(
            feature_idx, True, current_loss,
            best_dloss, best_loss_left, best_loss_right,
            data[selected_indices],
            data[~selected_indices],
            threshold_idx=best_idx,
            ordered_labels=ordered,
            first_index=first_idx
        )

    cdef np.float64_t _predict_instance(self, np.float64_t[::1] x) noexcept nogil:
        cdef _Node* node = self.root
        cdef np.float64_t val
        cdef int categorical_label
        while not _is_leaf(node):
            if node.is_categorical:
                categorical_label = <int>(x[node.feature_idx])
                if not vector_contains_int32(
                        &node.valid_modalities, categorical_label):
                    # Stop search if modality of categorical value has never
                    # been seen at this point
                    break
                elif vector_contains_int32(
                        &node.categorical_values_left, categorical_label):
                    node = node.left_child
                else:
                    node = node.right_child
            else:
                val = x[node.feature_idx]
                if val <= node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
        return node.avg_value

    def compute_importance2(self):
        return self.get_node_importances()

    cpdef np.ndarray get_node_importances(self):
        # Attention si on subsample les covariables!
        cdef np.ndarray importances = np.zeros(
            self.data.nb_features, dtype=np.float64
        )
        cdef int feature_idx
        cdef Node node
        for node in self.all_nodes:
            if node.is_leaf:
                continue
            feature_idx = node.feature_idx
            importances[feature_idx] += node.dloss
        importances *= 100. / importances.sum()
        return importances

