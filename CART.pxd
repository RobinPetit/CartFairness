# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: language_level=3
# cython: linetrace=True

cimport cython
cimport numpy as np

ctypedef size_t Pyssize_t

from loss cimport Loss, LossFunction

ctypedef enum _SplitType:
    BEST,
    DEPTH

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
        Vector valid_modalities

    _Node* new_node(size_t)
    void clear_node(_Node*)
    void _set_ys(_Node*, size_t, double, double, size_t)
    void _set_categorical_node_left_right_values(
            _Node*, const np.int32_t*, size_t, size_t, const np.int32_t*)
    void _set_left_child(_Node*, _Node*)
    void _set_right_child(_Node*, _Node*)
    bint _is_root(_Node*)
    bint _is_leaf(_Node*)
    bint vector_contains_int32(Vector*, np.int32_t)

    cdef struct PartitionResult_t:
        np.uint32_t mask
        double total_loss
        double loss_left
        double loss_right
    PartitionResult_t find_best_partition(
        LossFunction, size_t, const double*, const double*, const double*,
        const np.int32_t*, const np.int32_t*,
        double, double, size_t)
    # PartitionResult_t find_best_partition_mse(
    #     size_t, const double*, const double*, const double*,
    #     const np.int32_t*, const np.int32_t*,
    #     double, size_t)
    # PartitionResult_t find_best_partition_poisson_deviance(
    #     size_t, const double*, const double*, const double*,
    #     const np.int32_t*, const np.int32_t*,
    #     double, size_t)

from dataset cimport Dataset

cdef class Node:
    cdef _Node* node

    @staticmethod
    cdef Node from_ptr(_Node* ptr)

    cpdef list get_left_modalities(self, Dataset data)
    cpdef list get_right_modalities(self, Dataset data)
    cpdef list _get_left_indices(self)
    cpdef list _get_right_indices(self)

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
    cdef np.int32_t[::1] first_index

    cdef bint is_better_than(self, SplitChoice other)

@cython.final
cdef class __SortedFeatureData:
    cdef np.ndarray indices
    cdef np.float64_t[::1] X
    cdef np.float64_t[::1] y
    cdef np.float64_t[::1] w
    cdef np.float64_t[::1] p

    cdef inline np.float64_t get_max(self) noexcept nogil

@cython.final
cdef class CART:
    cdef bint bootstrap
    cdef bint replacement
    cdef LossFunction loss_fct
    cdef np.float64_t _epsilon
    cdef int nb_cov
    cdef int id
    cdef np.float64_t prop_sample
    cdef np.float64_t delta_loss
    cdef size_t _minobs
    cdef size_t depth
    cdef size_t max_depth
    cdef int max_interaction_depth
    cdef int nb_nodes
    cdef int nb_splitting_nodes
    cdef bint normalized_loss
    cdef bint pruning
    cdef bint relative_margin
    cdef np.float64_t prop_margin
    cdef Dataset data
    cdef _Node* root
    cdef _SplitType split_type
    cdef size_t min_nb_new_instances
    cdef np.float64_t prop_root_p0
    cdef bint exact_categorical_splits
    cdef int idx_nodes

    cdef list all_nodes

    cdef np.float64_t _loss(self, np.float64_t[::1] ys, np.float64_t[::1] ws)
    cdef _Node* _build_tree(self, Dataset data)
    cdef _Node* _build_tree_depth_first(
            self, Dataset data, size_t depth=*, np.float64_t loss=*)
    cdef _Node* _build_tree_best_first(self, Dataset data)
    cdef _Node* _create_node(
        self, np.float64_t[::1] ys,
        np.float64_t[::1] weights, size_t depth
    )
    cdef inline void _update_losses(
        self, Loss loss_left, Loss loss_right,
        np.float64_t[::1] ys, np.float64_t[::1] ws
    ) noexcept nogil
    cdef SplitChoice _find_best_split(
        self, Dataset data, np.float64_t precomputed_loss=*
    )
    cdef SplitChoice _find_best_threshold(
        self, Dataset data, int feature_idx,
        np.float64_t current_loss, np.float64_t prop_p0
    )
    cdef SplitChoice _find_best_threshold_numerical(
        self, Dataset data, int feature_idx,
        np.float64_t current_loss, np.float64_t prop_p0
    )
    cdef SplitChoice _find_best_threshold_categorical(
        self, Dataset data, int feature_idx,
        np.float64_t current_loss, np.float64_t prop_p0
    )
    cdef SplitChoice _find_best_threshold_categorical_exact(
        self, Dataset data, int feature_idx,
        np.float64_t current_loss, np.float64_t prop_p0
    )
    cdef SplitChoice _find_best_threshold_categorical_sorted_by_mean_ys(
        self, Dataset data, int feature_idx,
        np.float64_t current_loss, np.float64_t prop_p0
    )
    cdef np.float64_t _predict_instance(
        self, np.float64_t[::1] x
    ) noexcept nogil
    cpdef np.ndarray get_node_importances(self)
