#ifndef LOSSES_H
#define LOSSES_H

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct MSE_t {
    double sum;
    double sum_squares;
    bool precomputed;
    double value;
    size_t n;
} MSE_t;

static inline MSE_t* create_mse() {
    MSE_t* ret = (MSE_t*)malloc(sizeof(*ret));
    ret->sum = ret->sum_squares = ret->value = 0;
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
    mse->value = mse->sum_squares / mse->n - mu*mu;
}

static inline double evaluate_mse(const MSE_t* mse) {
    if(!mse->precomputed)
        _compute_mse((MSE_t*)(mse));
    return mse->value;
}

static inline void augment_mse(MSE_t* mse, double* ys, size_t n) {
    for(size_t i = 0; i < n; ++i) {
        mse->sum += ys[i];
        mse->sum_squares += ys[i]*ys[i];
    }
    mse->precomputed = false;
}

static inline void diminish_mse(MSE_t* mse, double* ys, size_t n) {
    for(size_t i = 0; i < n; ++i) {
        mse->sum -= ys[i];
        mse->sum_squares -= ys[i]*ys[i];
    }
    mse->precomputed = false;
}

typedef struct PoissonDeviance_t {
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
        PoissonDeviance_t* pd, double* ys, size_t n) {
    for(size_t i=0; i < n; ++i) {
        if(pd->n <= (size_t)ys[i])
            _reallocate_poisson_deviance(pd, ys[i]);
        ++pd->counts[(size_t)(ys[i])];
    }
    pd->precomputed = false;
}

static inline void diminish_poisson_deviance(
        PoissonDeviance_t* pd, double* ys, size_t n) {
    for(size_t i=0; i < n; ++i) {
        --pd->counts[(int)(ys[i])];
    }
    pd->precomputed = false;
}

#endif
