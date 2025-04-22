#ifndef __CART_HEADER__
#define __CART_HEADER__

#include "_loss.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define __EPSILON 1e-10
#define CLOSE_ENOUGH(a, b) (fabs(a-b) < __EPSILON)

typedef struct Vector {
    void* _base;
    size_t allocated;
    size_t n;
} Vector;

typedef struct PQ {
    Vector priorities;
    Vector pq;
    size_t n;
} PQ;

struct _Node {
    struct _Node* parent;
    struct _Node* left_child;
    struct _Node* right_child;

    double avg_value, threshold, loss, dloss;
    size_t nb_samples, depth, feature_idx, idx;
    bool is_categorical;
    Vector categorical_values_left;
    Vector categorical_values_right;
    Vector valid_modalities;
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
    if(vector->_base != NULL) {
        free(vector->_base);
        vector->allocated = vector->n = 0;
        vector->_base = NULL;
    }
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

static inline void insert_ptr_in_vector(Vector* vector, void* entry) {
    _ensure_sufficient_size(vector);
    ((void**)(vector->_base))[vector->n] = entry;
    ++vector->n;
}

static inline void insert_int32_in_vector(Vector* vector, int32_t entry) {
    _ensure_sufficient_size(vector);
    ((int32_t*)(vector->_base))[vector->n] = entry;
    ++vector->n;
}

static inline void insert_double_in_vector(Vector* vector, double entry) {
    _ensure_sufficient_size(vector);
    ((double*)(vector->_base))[vector->n] = entry;
    ++vector->n;
}

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

/*static inline void init_PQ(PQ* pq, size_t n) {
    pq->n = n;
    init_vector(&pq->pq, n);
    init_vector(&pq->priorities, n);
}

static inline void free_PQ(PQ* pq) {
    free_vector(&pq->priorities);
    free_vector(&pq->pq);
    pq->n = 0;
}

static inline bool PQ_less(PQ* pq, size_t i, size_t j) {
    double pi = Vector_double_at(&pq->priorities, Vector_int32_at(&pq->pq, i));
    double pj = Vector_double_at(&pq->priorities, Vector_int32_at(&pq->pq, j));
    return pi < pj;
}

static inline void PQ_swap(PQ* pq, size_t i, size_t j) {
    // pq->
}

static inline void PQ_swim(PQ* pq, size_t i) {
    while(i > 0 && PQ_less(pq, i, (i-1)/2)) {
        PQ_swap(pq, i, (i-1)/2);
        i = (i-1) / 2;
    }
}*/

/********** Node **********/

static inline struct _Node* new_node(size_t depth) {
    struct _Node* ret = (struct _Node*)malloc(sizeof(struct _Node));
    ret->left_child = ret->right_child = ret->parent = NULL;
    ret->depth = depth;
    ret->is_categorical = false;
    init_vector(&ret->categorical_values_left, 0);
    init_vector(&ret->categorical_values_right, 0);
    return ret;
}

static inline void clear_node(struct _Node* root) {
    if(root->left_child != NULL)
        clear_node(root->left_child);
    if(root->right_child != NULL)
        clear_node(root->right_child);
    free_vector(&root->categorical_values_left);
    free_vector(&root->categorical_values_right);
    free(root);
}

static inline void _set_ys(
        struct _Node* node, size_t idx, double avg, double loss, size_t size) {
    node->idx = idx;
    node->avg_value = avg;
    node->loss = loss;
    node->nb_samples = size;
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

static inline PartitionResult_t find_best_partition_mse(
        size_t nb_modalities,
        const double* ys, const double* ws, const double* ps,
        const int32_t* first_idx, const int32_t* mapping,
        double prop_root_p0, double epsilon, size_t minobs) {
    MSE_t* losses = calloc(nb_modalities, sizeof(*losses));
    size_t* sum_p0 = calloc(nb_modalities, sizeof(*sum_p0));
    MSE_t loss_left;
    MSE_t loss_right;
    for(size_t i = 0; i < nb_modalities; ++i) {
        _init_mse(&losses[i]);
        int32_t beg = first_idx[mapping[i]];
        int32_t end = first_idx[mapping[i]+1];
        augment_mse(&losses[i], &ys[beg], &ws[beg], end-beg);
        for(int32_t j = beg; j < end; ++j)
            if(ps[i] == 0.)
                ++sum_p0[i];
    }
    double loss;
    PartitionResult_t ret = {0, INFINITY, INFINITY, INFINITY};
    for(uint32_t mask = 1; mask <= (1<<(nb_modalities-2)); ++mask) {
        _init_mse(&loss_left);
        _init_mse(&loss_right);
        uint32_t copy = mask;
        size_t sum_p0_left = 0;
        size_t sum_p0_right = 0;
        for(size_t idx = 0; idx < nb_modalities; ++idx) {
            if(copy & 1) {
                join_mse(&loss_left, &losses[idx]);
                sum_p0_left += sum_p0[idx];
            } else {
                join_mse(&loss_right, &losses[idx]);
                sum_p0_right += sum_p0[idx];
            }
            copy >>= 1;
        }
        size_t count_left = loss_left.n;
        size_t count_right = loss_right.n;
        if(count_left <= minobs || count_right <= minobs)
            continue;
        double prop_p0_left = sum_p0_left / (double)count_left;
        double prop_p0_right = sum_p0_right / (double)count_right;
        if(fabs(prop_p0_left - prop_root_p0) > epsilon*prop_root_p0)
            continue;
        if(fabs(prop_p0_right - prop_root_p0) > epsilon*prop_root_p0)
            continue;
        loss  = evaluate_mse(&loss_left);
        loss += evaluate_mse(&loss_right);
        if(loss < ret.total_loss) {
            ret.mask = mask;
            ret.total_loss = loss;
            ret.loss_left = loss_left.value;
            ret.loss_right = loss_right.value;
        }
    }
    free(losses);
    free(sum_p0);
    return ret;
}

static inline PartitionResult_t find_best_partition_poisson_deviance(
        size_t nb_modalities,
        const double* ys, const double* ws, const double* ps,
        const int32_t* first_idx, const int32_t* mapping,
        double prop_root_p0, double epsilon, size_t minobs) {
    PoissonDeviance_t* losses = calloc(nb_modalities, sizeof(*losses));
    size_t* sum_p0 = calloc(nb_modalities,  sizeof(*sum_p0));
    PoissonDeviance_t loss_left;
    PoissonDeviance_t loss_right;
    for(size_t i = 0; i < nb_modalities; ++i) {
        _init_poisson_deviance(&losses[i]);
        int32_t beg = first_idx[mapping[i]];
        int32_t end = first_idx[mapping[i]+1];
        augment_poisson_deviance(&losses[i], &ys[beg], &ws[beg], end-beg);
        for(int32_t j = beg; j < end; ++j)
            if(ps[j] == 0.)
                ++sum_p0[i];
    }
    double loss = 0;
    PartitionResult_t ret = {0, INFINITY, INFINITY, INFINITY};
    for(uint32_t mask = 1; mask <= (1<<(nb_modalities-1)); ++mask) {
        _init_poisson_deviance(&loss_left);
        _init_poisson_deviance(&loss_right);
        uint32_t copy = mask;
        size_t sum_p0_left = 0;
        size_t sum_p0_right = 0;
        for(size_t idx = 0; idx < nb_modalities; ++idx) {
            if(copy & 1) {
                join_poisson_deviance(&loss_left, &losses[idx]);
                sum_p0_left += sum_p0[idx];
            } else {
                join_poisson_deviance(&loss_right, &losses[idx]);
                sum_p0_right += sum_p0[idx];
            }
            copy >>= 1;
        }
        size_t count_left = loss_left.n;
        size_t count_right = loss_right.n;
        if(count_left <= minobs || count_right <= minobs)
            continue;
        double prop_p0_left = sum_p0_left / (double)count_left;
        double prop_p0_right = sum_p0_right / (double)count_right;
        if(fabs(prop_p0_left - prop_root_p0) > epsilon*prop_root_p0)
            continue;
        if(fabs(prop_p0_right - prop_root_p0) > epsilon*prop_root_p0)
            continue;
        loss  = evaluate_poisson_deviance(&loss_left);
        loss += evaluate_poisson_deviance(&loss_right);
        if(loss < ret.total_loss) {
            ret.mask = mask;
            ret.total_loss = loss;
            ret.loss_left = loss_left.value;
            ret.loss_right = loss_right.value;
        }
    }
    for(size_t i = 0; i < nb_modalities; ++i) {
        RELEASE_PTR(losses[i].sum_of_weights);
        RELEASE_PTR(losses[i].ylogys);
    }
    RELEASE_PTR(loss_left.sum_of_weights);
    RELEASE_PTR(loss_left.ylogys);
    RELEASE_PTR(loss_right.sum_of_weights);
    RELEASE_PTR(loss_right.ylogys);
    free(losses);
    free(sum_p0);
    return ret;
}

#endif
