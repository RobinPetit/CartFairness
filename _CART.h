#ifndef __CART_HEADER__
#define __CART_HEADER__

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
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
    Vector data;
    size_t n;
} PQ;

struct _Node {
    struct _Node* parent;
    struct _Node* left_child;
    struct _Node* right_child;

    double avg_value, threshold, loss;
    size_t nb_samples, depth, feature_idx;
    bool is_categorical;
    Vector categorical_values_left;
    Vector categorical_values_right;
};

#ifndef __max
static inline size_t __max(size_t a, size_t b) {
    return (a > b) ? a : b;
}
#endif

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
        vector->_base = realloc(vector->_base, __ARRAY_ELEM_SIZE*vector->allocated);
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


static inline void init_PQ(PQ* pq, size_t n) {
    pq->n = n;
    init_vector(&pq->data, n);
    init_vector(&pq->priorities, n);
}

static inline void free_PQ(PQ* pq) {
    free_vector(&pq->priorities);
    free_vector(&pq->data);
    pq->n = 0;
}

static inline struct _Node* new_node(size_t depth) {
    struct _Node* ret = (struct _Node*)malloc(sizeof(struct _Node));
    ret->left_child = ret->right_child = ret->parent = NULL;
    ret->depth = depth;
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

static inline void _set_ys(struct _Node* node, double avg, double loss, size_t size) {
    node->avg_value = avg;
    node->loss = loss;
    node->nb_samples = size;
}

static inline void _set_categorical_node_left_right_values(
        struct _Node* node, const int32_t* const labels, size_t n, size_t threshold_idx) {
    init_vector_from_int_ptr(&node->categorical_values_left, labels, threshold_idx);
    init_vector_from_int_ptr(&node->categorical_values_right, labels+threshold_idx, n-threshold_idx);
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
#endif
