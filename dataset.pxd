# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: language_level=3
# cython: linetrace=True

cimport cython
cimport numpy as np

cdef extern from "_dataset.h" nogil:
    pass

cdef void _extract_mean_ys(
        np.float64_t[:] X, np.float64_t[:] y,
        np.float64_t[:] sums, np.int32_t[:] sizes) noexcept nogil

@cython.final
cdef class Dataset:
    cdef np.ndarray _X
    cdef np.ndarray _y
    cdef np.ndarray _p
    cdef np.ndarray _w
    cdef np.ndarray _is_categorical
    cdef np.ndarray _indices
    cdef size_t _size
    cdef list _reverse_mapping

    cdef np.float64_t[:, ::1] _indexed_X
    cdef np.float64_t[::1] _indexed_y
    cdef np.float64_t[::1] _indexed_p
    cdef np.float64_t[::1] _indexed_w

    cdef np.float64_t sum_of_weights_p0
    cdef np.float64_t sum_of_weights_p1

    cpdef bint is_categorical(self, int feature_idx)
    cdef void _labelize(
            self, np.ndarray[object, ndim=2] X,
            np.float64_t[:, :] out, int col_idx,
            bint compute_mapping=*)
    cdef Dataset sample(
            self, double prop_sample, bint replacement)
    cdef bint not_all_equal(self, int col_idx) noexcept nogil
    cdef size_t get_length(self) noexcept nogil
    cdef size_t size(self) noexcept nogil
    cdef np.ndarray transform(self, np.ndarray X)
    cpdef int nb_modalities_of(self, int feature_idx)
    cdef np.float64_t get_prop_p0(self)
    cdef str _reverse(self, int, int)

    cdef np.float64_t[:, ::1] _get_X(self) noexcept
    cdef np.float64_t[::1] _get_y(self) noexcept
    cdef np.float64_t[::1] _get_w(self) noexcept
    cdef np.float64_t[::1] _get_p(self) noexcept

    cdef void _clear(self) noexcept
