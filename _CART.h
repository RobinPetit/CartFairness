#ifndef __CART_HEADER__
#define __CART_HEADER__

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <Python.h>
#include "_loss.h"

#define __EPSILON 1e-10
#define CLOSE_ENOUGH(a, b) (fabs(a-b) < __EPSILON)

typedef struct Vector {
    void* _base;
    size_t allocated;
    size_t n;
} Vector;

typedef struct NodePq_t {
    Vector data;
} NodePq_t;

struct _Node {
    struct _Node* parent;
    struct _Node* left_child;
    struct _Node* right_child;

    double     prop_p0, avg_value, threshold, loss, dloss;
    size_t     nb_samples, depth, feature_idx, idx;
    bool       is_categorical;
    Vector     categorical_values_left;
    Vector     categorical_values_right;
    Vector     valid_modalities;
    PyObject*  extra_data;
};

#ifndef __max
static inline size_t __max(size_t a, size_t b) {
    return (a > b) ? a : b;
}
#endif

/********** Vector **********/

#define __ARRAY_ELEM_SIZE __max(sizeof(void*), sizeof(double))

static inline void init_vector(Vector* vector, size_t n) {
    vector->n = 0;
    vector->allocated = n;
    if(n > 0)
        vector->_base = malloc(__ARRAY_ELEM_SIZE * n);
    else
        vector->_base = NULL;
}

static inline void free_vector(Vector* vector) {
    RELEASE_PTR(vector->_base);
    vector->allocated = vector->n = 0;
}

static inline bool vector_contains_int32(const Vector* vector, int32_t x) {
    size_t i;
    for(i = 0; i < vector->n; ++i)
        if(((int32_t*)(vector->_base))[i] == x)
            return true;
    return false;
}

static inline bool vector_contains_double(const Vector* vector, double x) {
    size_t i;
    for(i = 0; i < vector->n; ++i)
        if(CLOSE_ENOUGH(((double*)(vector->_base))[i], x))
            return true;
    return false;
}

static inline void _ensure_sufficient_size(Vector* vector) {
    if(vector->n == vector->allocated) {
        vector->allocated <<= 1;
        vector->_base = realloc(
            vector->_base, __ARRAY_ELEM_SIZE*vector->allocated
        );
    }
}

#define __LET_INSERT_IN_VECTOR(_name, T) \
static inline void insert_ ## _name ## _in_vector(Vector* vector, T entry) { \
    _ensure_sufficient_size(vector); \
    ((T*)vector->_base)[vector->n] = entry; \
    ++vector->n; \
}

__LET_INSERT_IN_VECTOR(ptr, void*)
__LET_INSERT_IN_VECTOR(int32, int32_t)
__LET_INSERT_IN_VECTOR(double, double)

#undef __LET_INSERT_IN_VECTOR

static inline void init_vector_from_int_ptr(
        Vector* vector, const int32_t* const data, size_t n) {
    free_vector(vector);
    init_vector(vector, n);
    size_t i;
    for(i = 0; i < n; ++i)
        insert_int32_in_vector(vector, data[i]);
}

static inline double Vector_double_at(const Vector* vec, size_t i) {
    return ((double*)(vec->_base))[i];
}

static inline int32_t Vector_int32_at(const Vector* vec, size_t i) {
    return ((int32_t*)(vec->_base))[i];
}

/********** PQ **********/

static inline void init_node_pq(NodePq_t* pq, size_t n) {
    init_vector(&pq->data, n+1);
    pq->data.n = 1;  // start indexing at 1
}

static inline void destroy_pq_node(NodePq_t* pq) {
    free_vector(&pq->data);
}

static inline double __get_pq_loss(NodePq_t* pq, size_t i) {
    return ((struct _Node**)pq->data._base)[i]->loss;
}

static inline double __get_pq_dloss(NodePq_t* pq, size_t i) {
    return ((struct _Node**)pq->data._base)[i]->dloss;
}

static inline bool _pq_less_loss(NodePq_t* pq, size_t i, size_t j) {
    return __get_pq_loss(pq, i) < __get_pq_loss(pq, j);
}

static inline bool _pq_less_dloss(NodePq_t* pq, size_t i, size_t j) {
    return __get_pq_dloss(pq, i) < __get_pq_dloss(pq, j);
}

static inline void _pq_swap(NodePq_t* pq, size_t i, size_t j) {
    void* ptr_i = ((void**)pq->data._base)[i];
    ((void**)pq->data._base)[i] = ((void**)pq->data._base)[j];
    ((void**)pq->data._base)[j] = ptr_i;
}

#define PQ_LESS _pq_less_loss

static inline void _pq_swim(NodePq_t* pq, size_t i) {
    while(i > 1 && PQ_LESS(pq, i/2, i)) {
        _pq_swap(pq, i, i/2);
        i /= 2;
    }
}

static inline void _pq_sink(NodePq_t* pq, size_t i) {
    while(2*i < pq->data.n) {
        size_t left_child = 2*i;
        size_t right_child = left_child+1;
        size_t biggest_child = left_child;
        if(right_child < pq->data.n && PQ_LESS(pq, left_child, right_child))
            biggest_child = right_child;
        if(PQ_LESS(pq, biggest_child, i))
            break;
        _pq_swap(pq, i, biggest_child);
        i = biggest_child;
    }
}

static inline void pq_insert(NodePq_t* pq, struct _Node* node) {
    insert_ptr_in_vector(&pq->data, node);
    _pq_swim(pq, pq->data.n-1);
}

static inline struct _Node* pq_top(NodePq_t* pq) {
    return ((struct _Node**)pq->data._base)[1];
}

static inline struct _Node* pq_pop(NodePq_t* pq) {
    --pq->data.n;
    _pq_swap(pq, 1, pq->data.n);
    struct _Node* ret = ((struct _Node**)pq->data._base)[pq->data.n];
    _pq_sink(pq, 1);
    return ret;
}

static inline bool pq_empty(const NodePq_t* pq) {
    return pq->data.n <= 1;
}

/********** Node **********/

static inline struct _Node* new_node(size_t depth) {
    struct _Node* ret = (struct _Node*)malloc(sizeof(struct _Node));
    ret->left_child = ret->right_child = ret->parent = NULL;
    ret->depth = depth;
    ret->is_categorical = false;
    init_vector(&ret->categorical_values_left, 0);
    init_vector(&ret->categorical_values_right, 0);
    init_vector(&ret->valid_modalities, 0);
    return ret;
}

static inline void clear_node(struct _Node* root) {
    if(root->left_child != NULL)
        clear_node(root->left_child);
    if(root->right_child != NULL)
        clear_node(root->right_child);
    free_vector(&root->categorical_values_left);
    free_vector(&root->categorical_values_right);
    free_vector(&root->valid_modalities);
    free(root);
}

static inline void _set_ys_ps(
        struct _Node* node, size_t idx, double avg, double loss, size_t size, double prop_p0) {
    node->idx = idx;
    node->avg_value = avg;
    node->loss = loss;
    node->nb_samples = size;
    node->prop_p0 = prop_p0;
}

static inline void _set_categorical_node_left_right_values(
        struct _Node* node, const int32_t* const labels, size_t n,
        size_t threshold_idx, const int32_t* first_idx) {
    node->is_categorical = true;
    init_vector_from_int_ptr(
        &node->categorical_values_left,
        labels, threshold_idx+1
    );
    init_vector_from_int_ptr(
        &node->categorical_values_right,
        labels+threshold_idx+1, n-threshold_idx-1
    );
    init_vector(&node->valid_modalities, n);
    for(size_t i=0; i < n; ++i)
        if(first_idx[i] < first_idx[i+1])
            insert_int32_in_vector(&node->valid_modalities, (int32_t)i);
}

static inline void _set_left_child(struct _Node* root, struct _Node* child) {
    root->left_child = child;
    child->parent = root;
}

static inline void _set_right_child(struct _Node* root, struct _Node* child) {
    root->right_child = child;
    child->parent = root;
}

static inline bool _is_root(struct _Node* root) {
    return root->parent == NULL;
}

static inline bool _is_leaf(struct _Node* root) {
    return root->left_child == NULL || root->right_child == NULL;
}

/********** Partitions **********/

typedef struct PartitionResult_t {
    uint32_t mask;
    double total_loss;
    double loss_left;
    double loss_right;
} PartitionResult_t;

static inline PartitionResult_t find_best_partition(
        LossFunction_e loss_type, size_t nb_modalities,
        const double* ys, const double* ws, const double* ps,
        const int32_t* first_idx, const int32_t* mapping,
        double prop_root_p0, double prop_margin, size_t minobs) {
    AnyLoss_t losses = _create_any_loss_array(loss_type, nb_modalities);
    AnyLoss_t loss_left = _create_any_loss_array(loss_type, 1);
    AnyLoss_t loss_right= _create_any_loss_array(loss_type, 1);
    size_t* sum_p0 = calloc(nb_modalities, sizeof(*sum_p0));
    for(size_t i = 0; i < nb_modalities; ++i) {
        int32_t beg = first_idx[mapping[i]];
        int32_t end = first_idx[mapping[i]+1];
        _augment_any_loss(&losses, i, &ys[beg], &ws[beg], end-beg);
        for(int32_t j = beg; j < end; ++j)
            if(ps[i] == 0.)
                ++sum_p0[i];
    }
    double loss;
    PartitionResult_t ret = {0, INFINITY, INFINITY, INFINITY};
    for(uint32_t mask = 1; mask < (1u<<(nb_modalities-1)); ++mask) {
        _init_any_loss(&loss_left);
        _init_any_loss(&loss_right);
        uint32_t copy = mask;
        size_t sum_p0_left = 0;
        size_t sum_p0_right = 0;
        for(size_t idx = 0; idx < nb_modalities; ++idx) {
            void* other = _get_any_loss(&losses, idx);
            if(copy & 1) {
                _join_any_loss(&loss_left, other);
                sum_p0_left += sum_p0[idx];
            } else {
                _join_any_loss(&loss_right, other);
                sum_p0_right += sum_p0[idx];
            }
            copy >>= 1;
        }
        size_t count_left = loss_left.base->n;
        size_t count_right = loss_right.base->n;
        if(count_left <= minobs || count_right <= minobs)
            continue;
        double prop_p0_left = sum_p0_left / (double)loss_left.base->sum_of_weights;
        double prop_p0_right = sum_p0_right / (double)loss_right.base->sum_of_weights;
        if(fabs(prop_p0_left - prop_root_p0) > prop_margin)
            continue;
        if(fabs(prop_p0_right - prop_root_p0) > prop_margin)
            continue;
        loss  = _evaluate_any_loss(&loss_left);
        loss += _evaluate_any_loss(&loss_right);
        if(loss < ret.total_loss) {
            ret.mask = mask;
            ret.total_loss = loss;
            ret.loss_left = loss_left.base->value;
            ret.loss_right = loss_right.base->value;
        }
    }
    _destroy_any_loss_array(&losses, nb_modalities);
    _destroy_any_loss_array(&loss_left, 1);
    _destroy_any_loss_array(&loss_right, 1);
    RELEASE_PTR(sum_p0);
    return ret;
}

#endif
