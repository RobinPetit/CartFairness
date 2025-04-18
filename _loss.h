#ifndef LOSSES_H
#define LOSSES_H

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#define MALLOC_ONE(T) (T*)(malloc(sizeof(T)))
#define REALLOC(T, ptr, n) ptr = (T*)realloc(ptr, n*sizeof(T))
#define RELEASE_PTR(ptr) do { \
    if((ptr) == NULL) { \
        free(ptr); \
        ptr = NULL; \
    } \
    } while(0);

#define BASE_STRUCT_LOSS \
    double value; \
    size_t n; \
    bool precomputed;

struct __BasicLoss_t {
    BASE_STRUCT_LOSS
};

typedef struct MSE_t {
    BASE_STRUCT_LOSS
    double sum;
    double weighted_sum;
    double weighted_sum_squares;
    double sum_of_weights;
} MSE_t;

typedef struct PoissonDeviance_t {
    BASE_STRUCT_LOSS
    double* sum_of_weights;
    double* ylogys;
    size_t max_y;
    double weighted_sum;
    double sum;
} PoissonDeviance_t;


static inline MSE_t* create_mse() {
    MSE_t* ret = MALLOC_ONE(MSE_t);
    ret->sum = ret->weighted_sum = ret->weighted_sum_squares= ret->value = 0;
    ret->sum_of_weights = 0;
    ret->n = 0;
    ret->precomputed = true;
    return ret;
}

static inline void destroy_mse(MSE_t** mse) {
    if(*mse == NULL)
        return;
    free(*mse);
    *mse = NULL;
}

static inline void _compute_mse(MSE_t* mse) {
    double mu = mse->sum / mse->n;
    mse->precomputed = true;
    mse->value = mse->weighted_sum_squares
        + mu*mu * mse->sum_of_weights
        - 2*mu*mse->weighted_sum;
}

static inline double evaluate_mse(const MSE_t* mse) {
    if(!mse->precomputed)
        _compute_mse((MSE_t*)(mse));
    return mse->value;
}

static inline void augment_mse(
        MSE_t* mse, const double* ys, const double* ws, size_t n) {
    for(size_t i = 0; i < n; ++i) {
        mse->sum += ys[i];
        mse->weighted_sum += ys[i]*ws[i];
        mse->weighted_sum_squares += ys[i]*ys[i]*ws[i];
        mse->sum_of_weights += ws[i];
    }
    mse->precomputed = false;
}

static inline void diminish_mse(
        MSE_t* mse, const double* ys, const double* ws, size_t n) {
    for(size_t i = 0; i < n; ++i) {
        mse->sum -= ys[i];
        mse->weighted_sum -= ys[i]*ws[i];
        mse->weighted_sum_squares -= ys[i]*ys[i]*ws[i];
        mse->sum_of_weights -= ws[i];
    }
    mse->precomputed = false;
}

static inline void _reallocate_poisson_deviance(PoissonDeviance_t* pd, size_t k) {
    size_t current_size = pd->max_y;
    while(pd->max_y <= k)
        pd->max_y *= 2;
    REALLOC(double, pd->sum_of_weights, pd->max_y);
    REALLOC(double, pd->ylogys, pd->max_y);
    for(size_t i=current_size; i < pd->max_y; ++i) {
        pd->sum_of_weights[i] = 0;
        pd->ylogys[i] = i * log(i);
    }
}

static inline PoissonDeviance_t* create_poisson_deviance() {
    PoissonDeviance_t* ret = MALLOC_ONE(PoissonDeviance_t);
    ret->sum_of_weights = NULL;
    ret->ylogys = NULL;
    ret->max_y = 1;
    _reallocate_poisson_deviance(ret, 2);
    ret->sum = 0;
    ret->ylogys[0] = ret->sum_of_weights[0] = 0.;
    ret->precomputed = true;
    ret->value = 0;
    ret->n = 0;
    return ret;
}

static inline void destroy_poisson_deviance(PoissonDeviance_t** pd) {
    if(*pd == NULL)
        return;
    RELEASE_PTR((*pd)->sum_of_weights);
    RELEASE_PTR((*pd)->ylogys);
    *pd = NULL;
}

static inline void _compute_poisson(PoissonDeviance_t* pd) {
    double mu = pd->sum / pd->n;
    double logmu = (mu > 1e-18) ? log(mu) : 0.;
    pd->value = pd->sum_of_weights[0] * mu;
    for(size_t k=1; k < pd->max_y; ++k)
        pd->value += pd->sum_of_weights[k] *
            (pd->ylogys[k] - k*(1.+logmu) + mu);
    pd->value *= 2.;
    pd->precomputed = true;
}

static inline double evaluate_poisson_deviance(const PoissonDeviance_t* pd) {
    if(!pd->precomputed)
        _compute_poisson((PoissonDeviance_t*)(pd));
    return pd->value;
}

static inline void augment_poisson_deviance(
        PoissonDeviance_t* pd, const double* ys, const double* ws, size_t n) {
    for(size_t i=0; i < n; ++i) {
        if((size_t)(ys[i]) >= pd->max_y)
            _reallocate_poisson_deviance(pd, ys[i]);
        pd->sum_of_weights[(size_t)ys[i]] += ws[i];
        pd->sum += ys[i];
    }
    pd->n += n;
    pd->precomputed = false;
}

static inline void diminish_poisson_deviance(
        PoissonDeviance_t* pd, const double* ys, const double* ws, size_t n) {
    for(size_t i=0; i < n; ++i) {
        pd->sum_of_weights[(size_t)ys[i]] -= ws[i];
        pd->sum -= ys[i];
    }
    pd->n -= n;
    pd->precomputed = false;
}

/*typedef struct PoissonDeviance_t {
    size_t* counts;
    size_t n;
    bool precomputed;
    double value;
} PoissonDeviance_t;

static inline PoissonDeviance_t* create_poisson_deviance() {
    PoissonDeviance_t* ret = (PoissonDeviance_t*)malloc(sizeof(*ret));
    ret->n = 2;
    ret->counts = (size_t*)calloc(ret->n, sizeof(size_t));
    ret->precomputed = true;
    ret->value = 0;
    return ret;
};
static inline void destroy_poisson_deviance(PoissonDeviance_t** pd) {
    if(*pd == NULL)
        return;
    if((*pd)->counts != NULL) {
        free((*pd)->counts);
        (*pd)->counts = NULL;
    }
    *pd = NULL;
}

static inline void _compute_poisson(PoissonDeviance_t* pd) {
    double mu = 0;
    size_t total = 0;
    for(size_t k=0; k < pd->n; ++k) {
        mu += k * pd->counts[k];
        total += pd->counts[k];
    }
    mu /= total;
    pd->value = pd->counts[0] * mu;
    if(mu > 1e-18) {
        for(size_t k=1; k < pd->n; ++k)
            pd->value += pd->counts[k] * (k * log(k / mu) + (mu - k));
    } else {
        for(size_t k=1; k < pd->n; ++k)
            pd->value += pd->counts[k] * (mu - k);
    }
    pd->value *= 2. / total;
    pd->precomputed = true;
}

static inline double evaluate_poisson_deviance(const PoissonDeviance_t* pd) {
    if(!pd->precomputed)
        _compute_poisson((PoissonDeviance_t*)(pd));
    return pd->value;
}

static inline void _reallocate_poisson_deviance(PoissonDeviance_t* pd, size_t k) {
    size_t current_size = pd->n;
    while(pd->n <= k)
        pd->n *= 2;
    pd->counts = (size_t*)realloc(pd->counts, pd->n * sizeof(*pd->counts));
    for(size_t i=current_size; i < pd->n; ++i)
        pd->counts[i] = 0;
}

static inline void augment_poisson_deviance(
        PoissonDeviance_t* pd, double* ys, double* ws, size_t n) {
    for(size_t i=0; i < n; ++i) {
        if(pd->n <= (size_t)ys[i])
            _reallocate_poisson_deviance(pd, ys[i]);
        ++pd->counts[(size_t)(ys[i])];
    }
    pd->precomputed = false;
}

static inline void diminish_poisson_deviance(
        PoissonDeviance_t* pd, double* ys, double* ws, size_t n) {
    for(size_t i=0; i < n; ++i) {
        --pd->counts[(int)(ys[i])];
    }
    pd->precomputed = false;
}
*/

#endif
