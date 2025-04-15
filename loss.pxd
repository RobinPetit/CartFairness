# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: language_level=3
# cython: linetrace=True

cdef extern from "_loss.h" nogil:
    cdef struct MSE_t:
        pass
    cdef struct PoissonDeviance_t:
        pass
    MSE_t* create_mse()
    double evaluate_mse(MSE_t*)
    void destroy_mse(MSE_t**)
    void augment_mse(MSE_t*, double*, size_t)
    void diminish_mse(MSE_t*, double*, size_t)
    PoissonDeviance_t* create_poisson_deviance()
    void destroy_poisson_deviance(PoissonDeviance_t**)
    double evaluate_poisson_deviance(PoissonDeviance_t*)
    void augment_poisson_deviance(PoissonDeviance_t*, double*, size_t)
    void diminish_poisson_deviance(PoissonDeviance_t*, double*, size_t)

ctypedef enum LossFunction:
    MSE,
    POISSON

cdef class Loss:
    cdef LossFunction loss_type
    cdef void* loss_ptr
    cdef double get(self) noexcept nogil
    cdef void augment(self, double[::1] ys) noexcept nogil
    cdef void diminish(self, double[::1] ys) noexcept nogil
