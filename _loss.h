#ifndef LOSSES_H
#define LOSSES_H

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

typedef enum LossFunction_e {
    MSE,
    POISSON,
    GAMMA
} LossFunction_e;

#define MALLOC_ONE(T) (T*)(malloc(sizeof(T)))
#define REALLOC(T, ptr, n) ptr = (T*)realloc(ptr, n*sizeof(T))
#define RELEASE_PTR(ptr) do { \
    if((ptr) == NULL) { \
        free(ptr); \
        ptr = NULL; \
    } \
} while(0);

// Let's get C-messy!
#define BASE_STRUCT_LOSS \
    double value; \
    size_t n; \
    bool precomputed; \
    double sum_of_weights;

typedef struct __BasicLoss_t {
    BASE_STRUCT_LOSS
} __BasicLoss_t;

typedef struct MSE_t {
    BASE_STRUCT_LOSS
    double sum;
    double weighted_sum;
    double weighted_sum_squares;
} MSE_t;

typedef struct PoissonDeviance_t {
    BASE_STRUCT_LOSS
    double* ylogys;
    size_t max_y;
    double weighted_sum;
    double sum;
    double* sum_of_weights_ys;
} PoissonDeviance_t;

typedef struct GammaDeviance_t {
    BASE_STRUCT_LOSS
    double sum;
    double weighted_sum;
    double weighted_sum_log;
} GammaDeviance_t;

static inline void _destroy_basic_loss(void** ptr) {
    if(*ptr == NULL)
        return;
    RELEASE_PTR(*ptr);
}

static inline void _init_mse(MSE_t* mse) {
    mse->sum = mse->weighted_sum = mse->weighted_sum_squares = mse->value = 0;
    mse->sum_of_weights = 0;
    mse->n = 0;
    mse->precomputed = true;
}

static inline MSE_t* create_mse() {
    MSE_t* ret = MALLOC_ONE(MSE_t);
    _init_mse(ret);
    return ret;
}

static inline void destroy_mse(void** mse) {
    _destroy_basic_loss(mse);
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

static inline void join_mse(MSE_t* mse, const MSE_t* other) {
    mse->sum += other->sum;
    mse->weighted_sum += other->weighted_sum;
    mse->weighted_sum_squares += other->weighted_sum_squares;
    mse->sum_of_weights += other->sum_of_weights;
    mse->n += other->n;
    mse->precomputed = false;
}

static inline void unjoin_mse(MSE_t* mse, const MSE_t* other) {
    mse->sum -= other->sum;
    mse->weighted_sum -= other->weighted_sum;
    mse->weighted_sum_squares -= other->weighted_sum_squares;
    mse->sum_of_weights -= other->sum_of_weights;
    mse->n -= other->n;
    mse->precomputed = false;
}

static inline void _reallocate_poisson_deviance(PoissonDeviance_t* pd, size_t k) {
    size_t current_size = pd->max_y;
    while(pd->max_y <= k)
        pd->max_y *= 2;
    REALLOC(double, pd->sum_of_weights_ys, pd->max_y);
    REALLOC(double, pd->ylogys, pd->max_y);
    for(size_t i=current_size; i < pd->max_y; ++i) {
        pd->sum_of_weights_ys[i] = 0;
        pd->ylogys[i] = i * log(i);
    }
}

static inline void _init_poisson_deviance(PoissonDeviance_t* pd) {
    pd->sum_of_weights = 0;
    pd->sum_of_weights_ys = NULL;
    pd->ylogys = NULL;
    pd->max_y = 1;
    _reallocate_poisson_deviance(pd, 2);
    pd->sum = 0;
    pd->ylogys[0] = pd->sum_of_weights_ys[0] = 0.;
    pd->precomputed = true;
    pd->value = 0;
    pd->n = 0;
}

static inline PoissonDeviance_t* create_poisson_deviance() {
    PoissonDeviance_t* ret = MALLOC_ONE(PoissonDeviance_t);
    _init_poisson_deviance(ret);
    return ret;
}

static inline void destroy_poisson_deviance(void** ptr) {
    if(*ptr == NULL)
        return;
    PoissonDeviance_t* pd = *ptr;
    RELEASE_PTR(pd->sum_of_weights_ys);
    RELEASE_PTR(pd->ylogys);
    RELEASE_PTR(pd);
}

static inline void _compute_poisson(PoissonDeviance_t* pd) {
    double mu = pd->sum / pd->n;
    double logmu = (mu > 1e-18) ? log(mu) : 0.;
    pd->value = pd->sum_of_weights_ys[0] * mu;
    for(size_t k=1; k < pd->max_y; ++k)
        pd->value += pd->sum_of_weights_ys[k] *
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
    for(size_t i = 0; i < n; ++i) {
        if((size_t)(ys[i]) >= pd->max_y)
            _reallocate_poisson_deviance(pd, ys[i]);
        pd->sum_of_weights += ws[i];
        pd->sum_of_weights_ys[(size_t)ys[i]] += ws[i];
        pd->sum += ys[i];
    }
    pd->n += n;
    pd->precomputed = false;
}

static inline void diminish_poisson_deviance(
        PoissonDeviance_t* pd, const double* ys, const double* ws, size_t n) {
    for(size_t i=0; i < n; ++i) {
        pd->sum_of_weights -= ws[i];
        pd->sum_of_weights_ys[(size_t)ys[i]] -= ws[i];
        pd->sum -= ys[i];
    }
    pd->n -= n;
    pd->precomputed = false;
}

static inline void join_poisson_deviance(
        PoissonDeviance_t* pd, const PoissonDeviance_t* other) {
    if(pd->max_y < other->max_y)
        _reallocate_poisson_deviance(pd, other->max_y);
    for(size_t k=0; k < other->max_y; ++k)
        pd->sum_of_weights_ys[k] += other->sum_of_weights_ys[k];
    pd->sum_of_weights += other->sum_of_weights;
    pd->weighted_sum += other->weighted_sum;
    pd->sum += other->sum;
    pd->n += other->n;
    pd->precomputed = false;
}

static inline void unjoin_poisson_deviance(
        PoissonDeviance_t* pd, const PoissonDeviance_t* other) {
    for(size_t k=0; k < other->max_y; ++k)
        pd->sum_of_weights_ys[k] -= other->sum_of_weights_ys[k];
    pd->sum_of_weights -= other->sum_of_weights;
    pd->weighted_sum -= other->weighted_sum;
    pd->sum -= other->sum;
    pd->n -= other->n;
    pd->precomputed = false;
}

static inline void _init_gamma_deviance(GammaDeviance_t* gd) {
    gd->value = gd->n = 0;
    gd->precomputed = false;
    gd->sum = gd->weighted_sum = gd->weighted_sum_log = gd->sum_of_weights = 0;
}

static inline GammaDeviance_t* create_gamma_deviance() {
    GammaDeviance_t* ret = MALLOC_ONE(GammaDeviance_t);
    _init_gamma_deviance(ret);
    return ret;
}

static inline void destroy_gamma_deviance(void** gd) {
    _destroy_basic_loss(gd);
}

static inline void _compute_gamma_deviance(GammaDeviance_t* gd) {
    double mu = gd->sum / gd->n;
    gd->precomputed = true;
    gd->value = gd->weighted_sum_log
        - (log(mu)+1) * gd->sum_of_weights
        + gd->weighted_sum / mu;
    gd->value *= 2;
}

static inline double evaluate_gamma_deviance(const GammaDeviance_t* gd) {
    if(!gd->precomputed)
        _compute_gamma_deviance((GammaDeviance_t*)gd);
    return gd->value;
}

static inline void augment_gamma_deviance(
        GammaDeviance_t* gd, const double* ys, const double* ws, size_t n) {
    for(size_t i = 0; i < n; ++i) {
        gd->weighted_sum += ws[i] * ys[i];
        gd->sum_of_weights += ws[i];
        gd->weighted_sum_log += ws[i] * log(ys[i]);
        gd->sum += ys[i];
    }
    gd->n += n;
    gd->precomputed = false;
}

static inline void diminish_gamma_deviance(
        GammaDeviance_t* gd, const double* ys, const double* ws, size_t n) {
    for(size_t i = 0; i < n; ++i) {
        gd->weighted_sum -= ws[i] * ys[i];
        gd->sum_of_weights -= ws[i];
        gd->weighted_sum_log -= ws[i] * log(ys[i]);
        gd->sum -= ys[i];
    }
    gd->n -= n;
    gd->precomputed = false;
}

static inline void join_gamma_deviance(
        GammaDeviance_t* gd, const GammaDeviance_t* other) {
    gd->weighted_sum += other->weighted_sum;
    gd->sum_of_weights += other->sum_of_weights;
    gd->weighted_sum_log += other->weighted_sum_log;
    gd->sum += other->sum;
    gd->n += other->n;
    gd->precomputed = false;
}

static inline void unjoin_gamma_deviance(
        GammaDeviance_t* gd, const GammaDeviance_t* other) {
    gd->weighted_sum -= other->weighted_sum;
    gd->sum_of_weights -= other->sum_of_weights;
    gd->weighted_sum_log -= other->weighted_sum_log;
    gd->sum -= other->sum;
    gd->n -= other->n;
    gd->precomputed = false;
}

typedef struct AnyLoss_t {
    union {
        __BasicLoss_t* base;
        MSE_t* mse;
        PoissonDeviance_t* pd;
        GammaDeviance_t* gd;
        void* any;
    };
    LossFunction_e type;
} AnyLoss_t;

static inline void _init_any_loss(AnyLoss_t* loss) {
    switch(loss->type) {
    case MSE:
        _init_mse(loss->mse);
        break;
    case POISSON:
        _init_poisson_deviance(loss->pd);
        break;
    case GAMMA:
        _init_gamma_deviance(loss->gd);
        break;
    default:
        break;
    }
}

static inline AnyLoss_t _create_any_loss_array(LossFunction_e type, size_t n) {
    AnyLoss_t ret = {.any=NULL, .type=type};
    switch(type) {
    case MSE:
        ret.mse = calloc(n, sizeof(MSE_t));
        for(size_t i = 0; i < n; ++i)
            _init_mse(&ret.mse[i]);
        break;
    case POISSON:
        ret.pd = calloc(n, sizeof(PoissonDeviance_t));
        for(size_t i = 0; i < n; ++i)
            _init_poisson_deviance(&ret.pd[i]);
        break;
    case GAMMA:
        ret.gd = calloc(n, sizeof(GammaDeviance_t));
        for(size_t i = 0; i < n; ++i)
            _init_gamma_deviance(&ret.gd[i]);
        break;
    default: break;
    }
    return ret;
}

static inline void _destroy_any_loss_array(AnyLoss_t* losses, size_t n) {
    if(losses->type == POISSON) {
        for(size_t i = 0; i < n; ++i) {
            RELEASE_PTR(losses->pd[i].sum_of_weights_ys);
            RELEASE_PTR(losses->pd[i].ylogys);
        }
    }
    RELEASE_PTR(losses->any);
}

static inline void* _get_any_loss(AnyLoss_t* losses, size_t i) {
    switch(losses->type) {
    case MSE:
        return &losses->mse[i];
    case POISSON:
        return &losses->pd[i];
    case GAMMA:
        return &losses->gd[i];
    default:
        return 0;
    }
}

static inline void _augment_any_loss(
        AnyLoss_t* losses, size_t i,
        const double* ys, const double* ws, size_t n) {
    switch(losses->type) {
    case MSE:
        augment_mse(losses->mse+i, ys, ws, n);
        break;
    case POISSON:
        augment_poisson_deviance(losses->pd+i, ys, ws, n);
        break;
    case GAMMA:
        augment_gamma_deviance(losses->gd+i, ys, ws, n);
        break;
    default:
        break;
    }
}

static inline void _join_any_loss(AnyLoss_t* loss, const void* other) {
    switch(loss->type) {
    case MSE:
        join_mse(loss->mse, other);
        break;
    case POISSON:
        join_poisson_deviance(loss->pd, other);
        break;
    case GAMMA:
        join_gamma_deviance(loss->gd, other);
        break;
    }
}

static inline double _evaluate_any_loss(const AnyLoss_t* loss) {
    switch(loss->type) {
    case MSE:
        return evaluate_mse(loss->mse);
    case POISSON:
        return evaluate_poisson_deviance(loss->pd);
    case GAMMA:
        return evaluate_gamma_deviance(loss->gd);
    default: return 0;
    }
}

#endif
