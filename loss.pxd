# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: language_level=3
# cython: linetrace=True

cimport cython
cimport numpy as np

cdef extern from "_loss.h" nogil:
    cdef struct __BasicLoss_t:
        double value
        size_t n
        bint precomputed
    cdef struct MSE_t:
        pass
    cdef struct PoissonDeviance_t:
        pass
    cdef struct GammaDeviance_t:
        pass
    void _init_mse(MSE_t*)
    MSE_t* create_mse()
    double evaluate_mse(MSE_t*)
    void destroy_mse(void**)
    void augment_mse(MSE_t*, double*, double*, size_t)
    void diminish_mse(MSE_t*, double*, double*, size_t)
    void join_mse(MSE_t*, const MSE_t*)
    void unjoin_mse(MSE_t*, const MSE_t*)
    void _init_poisson_deviance(PoissonDeviance_t*)
    PoissonDeviance_t* create_poisson_deviance()
    void destroy_poisson_deviance(void**)
    double evaluate_poisson_deviance(PoissonDeviance_t*)
    void augment_poisson_deviance(PoissonDeviance_t*, double*, double*, size_t)
    void diminish_poisson_deviance(PoissonDeviance_t*, double*, double*, size_t)
    void join_poisson_deviance(PoissonDeviance_t*, const PoissonDeviance_t*)
    void unjoin_poisson_deviance(PoissonDeviance_t*, const PoissonDeviance_t*)
    GammaDeviance_t* create_gamma_deviance()
    void destroy_gamma_deviance(void**)
    double evaluate_gamma_deviance(GammaDeviance_t*)
    void augment_gamma_deviance(GammaDeviance_t*, double*, double*, size_t)
    void diminish_gamma_deviance(GammaDeviance_t*, double*, double*, size_t)
    void join_gamma_deviance(GammaDeviance_t*, const GammaDeviance_t*)
    void unjoin_gamma_deviance(GammaDeviance_t*, const GammaDeviance_t*)

    ctypedef enum LossFunction "LossFunction_e":
        MSE = 1,
        POISSON = 2,
        GAMMA = 3

@cython.final
cdef class Loss:
    cdef LossFunction loss_type
    cdef bint normalized
    cdef void* loss_ptr

    cdef inline double get(self) noexcept nogil
    cdef inline void augment(self, double[::1] ys, double[::1] ws) noexcept nogil
    cdef inline void diminish(self, double[::1] ys, double[::1] ws) noexcept nogil
    cdef inline size_t get_size(self) noexcept nogil

    cdef inline void join(self, Loss other) noexcept nogil
    cdef inline void unjoin(self, Loss other) noexcept nogil

cpdef float poisson_deviance(np.ndarray y1, np.ndarray y2)
