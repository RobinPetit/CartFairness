# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: language_level=3
# cython: linetrace=True

cimport cython

@cython.final
cdef class Loss:
    def __cinit__(self, LossFunction loss_type, bint normalized):
        self.loss_type = loss_type
        self.normalized = normalized
        if loss_type == LossFunction.MSE:
            self.loss_ptr = create_mse()
        elif loss_type == LossFunction.POISSON:
            self.loss_ptr = create_poisson_deviance()
        elif loss_type == LossFunction.GAMMA:
            self.loss_ptr = create_gamma_deviance()
        else:
            raise ValueError('Unknown loss type!')
        if self.loss_ptr == NULL:
            raise ValueError('NULL POINTER!')

    def __dealloc__(self):
        if self.loss_type == LossFunction.MSE:
            destroy_mse(&self.loss_ptr)
        elif self.loss_type == LossFunction.POISSON:
            destroy_poisson_deviance(&self.loss_ptr)
        elif self.loss_type == LossFunction.GAMMA:
            destroy_gamma_deviance(&self.loss_ptr)

    cdef double get(self) noexcept nogil:
        cdef double ret = 0
        if self.loss_type == LossFunction.MSE:
            ret = evaluate_mse(<MSE_t*>self.loss_ptr)
        elif self.loss_type == LossFunction.POISSON:
            ret = evaluate_poisson_deviance(<PoissonDeviance_t*>self.loss_ptr)
        elif self.loss_type == LossFunction.GAMMA:
            ret = evaluate_gamma_deviance(<GammaDeviance_t*>self.loss_ptr)
        if self.normalized:
            ret /= (<__BasicLoss_t*>self.loss_ptr).n
        return ret

    cdef void augment(self, double[::1] ys, double[::1] ws) noexcept nogil:
        if self.loss_type == LossFunction.MSE:
            augment_mse(<MSE_t*>self.loss_ptr, &ys[0], &ws[0], ys.shape[0])
        elif self.loss_type == LossFunction.POISSON:
            augment_poisson_deviance(
                <PoissonDeviance_t*>self.loss_ptr, &ys[0], &ws[0], ys.shape[0]
            )
        elif self.loss_type == LossFunction.GAMMA:
            augment_gamma_deviance(
                <GammaDeviance_t*>self.loss_ptr, &ys[0], &ws[0], ys.shape[0]
            )

    cdef void diminish(self, double[::1] ys, double[::1] ws) noexcept nogil:
        if self.loss_type == LossFunction.MSE:
            diminish_mse(<MSE_t*>self.loss_ptr, &ys[0], &ws[0], ys.shape[0])
        elif self.loss_type == LossFunction.POISSON:
            diminish_poisson_deviance(
                <PoissonDeviance_t*>self.loss_ptr, &ys[0], &ws[0], ys.shape[0]
            )
        elif self.loss_type == LossFunction.GAMMA:
            diminish_gamma_deviance(
                <GammaDeviance_t*>self.loss_type, &ys[0], &ws[0], ys.shape[0]
            )

    cdef size_t get_size(self) noexcept nogil:
        return (<__BasicLoss_t*>self.loss_ptr).n

    cdef void join(self, Loss other) noexcept nogil:
        if self.loss_type == LossFunction.MSE:
            join_mse(<MSE_t*>self.loss_ptr, <MSE_t*>other.loss_ptr)
        elif self.loss_type == LossFunction.POISSON:
            join_poisson_deviance(
                <PoissonDeviance_t*>self.loss_ptr,
                <PoissonDeviance_t*>other.loss_ptr
            )
        elif self.loss_type == LossFunction.GAMMA:
            join_gamma_deviance(
                <GammaDeviance_t*>self.loss_ptr,
                <GammaDeviance_t*>other.loss_ptr
            )

    cdef void unjoin(self, Loss other) noexcept nogil:
        if self.loss_type == LossFunction.MSE:
            unjoin_mse(<MSE_t*>self.loss_ptr, <MSE_t*>other.loss_ptr)
        elif self.loss_type == LossFunction.POISSON:
            unjoin_poisson_deviance(
                <PoissonDeviance_t*>self.loss_ptr,
                <PoissonDeviance_t*>other.loss_ptr
            )
        elif self.loss_type == LossFunction.GAMMA:
            unjoin_gamma_deviance(
                <GammaDeviance_t*>self.loss_ptr,
                <GammaDeviance_t*>other.loss_ptr
            )

