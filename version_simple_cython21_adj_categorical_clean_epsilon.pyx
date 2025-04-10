# cython: language_level=3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import mean_poisson_deviance
import time
cimport numpy as np
from cython cimport boundscheck, wraparound
from libc.math cimport log
from numpy.random import randint
from random import sample
import matplotlib.patches as patches

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# Definition of the usual loss functions
cdef double _mse(double[:] y):
    # compute mean square error loss function if chosen in _loss
    cdef double mean = compute_mean(y)
    cdef double mse = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t n = y.shape[0]

    with boundscheck(False), wraparound(False), nogil:
        for i in range(n):
                mse += (y[i] - mean) ** 2
    return mse/n

cpdef double _poisson_deviance_simp2(double[:] y, double[:] y_pred):
    # compute poisson deviance loss function if chosen in _loss

    cdef double epsilon = 10**-18
    cdef double dev = 0.0
    cdef Py_ssize_t n = y.shape[0]
    cdef Py_ssize_t i

    # Check if the array is empty
    if n == 0:
        raise ValueError("Input arrays cannot be empty.")

    # Iterate over the arrays and calculate the deviance
    with boundscheck(False), wraparound(False), nogil:
        for i in range(n):
            # Ensure that neither y[i] nor y_pred[i] is too small or zero
            if y[i] > epsilon and y_pred[i] > epsilon:
                dev += 2*(y[i] * log((y[i] + epsilon) / (y_pred[i] + epsilon)) + (y_pred[i]-y[i]))
            else:
                # Handle the case where either y[i] or y_pred[i] is 0 or too small
                dev += 2*(y_pred[i]-y[i])  # Optionally: handle as zero contribution, or use a default small value

    # Return the deviance divided by the number of elements
    return dev / n

cpdef double compute_mean(double[:] arr):
    # function for computing the mean to not manipulate numpy vectors.

    cdef double sum_values = 0.0
    cdef Py_ssize_t n = arr.shape[0]
    cdef Py_ssize_t i

    with boundscheck(False), wraparound(False), nogil:
        for i in range(n):
            sum_values += arr[i]

    return sum_values / n

cdef double _pairwise_sum(double[:] arr, int start, int end) nogil:
    # function for computing the sum pairwise to avoid approximation (not used finally I think)

    cdef Py_ssize_t length = end - start

    if length == 1:
        return arr[start]
    elif length == 2:
        return arr[start] + arr[start + 1]

    cdef Py_ssize_t mid = start + length // 2
    return _pairwise_sum(arr, start, mid) + _pairwise_sum(arr, mid, end)


cpdef double _loss(double[:] y):
    # compute loss function chosen

    cdef double[:] y_pred = y.copy()
    cdef double y_mean = np.mean(y)#compute_mean2(y)
    y_pred[:] = y_mean  # This fills the entire array with the mean
    return _poisson_deviance_simp2(y, y_pred)


cpdef list my_random_choice(list population, int k=1, bint replace=False):
    """
    Return a k-length list of elements chosen from the population sequence.
    If replace is False, the elements are sampled without replacement.
    """
    cdef list result

    if not replace and k > len(population):
        raise ValueError("Cannot sample without replacement if k exceeds population size")

    if replace:
        # Generate k random indices with replacement
        result = [population[randint(0, len(population))] for _ in range(k)]
    else:
        # Use Python's sample for sampling without replacement
        result = sample(population, k)

    return result

# Define a helper function to sort
cpdef double sort_key(tuple cat_mean):
    """
    Small function to get the key of the dictionary (mean value associated with the modality of the categorical variable).
    """
    return cat_mean[1]

cpdef np.ndarray[object, ndim=2] preprocess_data(np.ndarray[object, ndim=2] X, dict dic_cov):
    """
    Transform numerical data into float64 dtypes (preprocessing step) based on dictionary dic_cov containing
    mapping between each variable and its dtype
    """

    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]

    # Iterate through columns and update data types
    for i in range(n_cols):
        feature_type = dic_cov[list(dic_cov.keys())[i]]
        #print(feature_type)
        if feature_type == 'category':
            # Leave as category (assuming 'category' columns are passed with their type information)
            pass
        else:
            # Convert other columns (like int64) to float64
            for j in range(n_rows):
                X[j, i] = float(X[j, i])

    return X


cdef class Node:
    """
    Node class being elements of CARTRegressor_cython. The characteristics of the nodes are its kind (Root for the first node
    (on top of the tree), Node as splitting node or Leaf as terminal node), feature index, threshold, and associated loss.
    The average value denotes the prediction associated with this node as the mean response observed on the training observations.
    Mapping categorical denotes the mapping (dictionary) between modalities of the categorical variable and integer value used for splitting.
    Integer values are determined in ordering the mean response for each modality in ascending order. Note that The mapping
    or a same variable could be different at each node of the tree depending on the modalities present (depending on observations)
    in the nodes as well as mean response observed that could be different based on the observations. Other attributes are
    saved as the number of observations in the node, the depth of the node in the tree, the parent node and children nodes
    of the current node.
    """

    cdef str _kind
    cdef int _depth
    cdef int _feature_index
    cdef double _threshold
    cdef double _loss
    cdef double _average_value

    cdef Node _parent_node
    cdef Node _left_child
    cdef Node _right_child
    cdef str _position
    cdef int _index
    cdef double _loss_decrease
    cdef dict mapping_categorical
    cdef int nb_samples

    def __init__(self, kind="Root", depth=0, feature_index=-1, threshold=-1, loss=0, average_value=-1, loss_decrease=0, mapping=None, nb_samples=0):
        # Main characteristic of the node (kind=Node/Leaf) split feature, split threshold, split mse, split samples)
        self._kind = kind
        self._feature_index = feature_index
        self._threshold = threshold
        self._loss = loss
        self._average_value = average_value
        self.mapping_categorical = mapping
        self.nb_samples = nb_samples

        # Secondary characteristic allowing to identify the position of the node in the tree
        self._depth = depth
        self._parent_node = None
        self._left_child = None
        self._right_child = None
        self._position = None
        self._index = -1
        self._loss_decrease = loss_decrease#0

    # access attributes outside of the class
    property kind:
        def __get__(self):
            return self._kind
        def __set__(self, value):
            self._kind = value

    property feature_index:
        def __get__(self):
            return self._feature_index
        def __set__(self, value):
            self._feature_index = value

    property threshold:
        def __get__(self):
            return self._threshold
        def __set__(self, value):
            self._threshold = value

    property loss:
        def __get__(self):
            return self._loss
        def __set__(self, value):
            self._loss = value

    property average_value:
        def __get__(self):
            return self._average_value
        def __set__(self, value):
            self._average_value = value

    property depth:
        def __get__(self):
            return self._depth
        def __set__(self, value):
            self._depth = value

    property parent_node:
        def __get__(self):
            return self._parent_node
        def __set__(self, value):
            self._parent_node = value

    property left_child:
        def __get__(self):
            return self._left_child
        def __set__(self, value):
            self._left_child = value

    property right_child:
        def __get__(self):
            return self._right_child
        def __set__(self, value):
            self._right_child = value

    property position:
        def __get__(self):
            return self._position
        def __set__(self, value):
            self._position = value

    property index:
        def __get__(self):
            return self._index
        def __set__(self, value):
            self._index = value

    property loss_decrease:
        def __get__(self):
            return self._loss_decrease
        def __set__(self, value):
            self._loss_decrease = value

    property mapping_categorical:
        def __get__(self):
            return self.mapping_categorical
        def __set__(self, value):
            self.mapping_categorical = value

    property nb_samples:
        def __get__(self):
            return self.nb_samples
        def __set__(self, value):
            self.nb_samples = value

cdef class CARTRegressor_cython:
    """
    Tree regression model class containing all dependent nodes (stored in the list nodes). Other attributes denote the
    usual constraints during learning: minobs=maximum number of observations per terminal node (=leaf), maximum interaction depth
    (= number of splitting node in the tree), if boostrap is performed (random sampling of observations with or without replacement only before fitting)
    and the number of covariates used for sampling in the original covariate at each node.
    The fairness constraint are the epsilon (proportion difference between men and women allowed within each node),
    margin (str) if the epsilon is computed in relative or absolute way.
    The matrices X_train, ... and vectors y_train, p_train (being the protected variable => here gender=men/women) are declared
    here as used in the fit function before calling build_tree.
    """

    cdef Node root
    cdef int minobs
    cdef int tree_depth

    cdef list nodes
    cdef int idx
    cdef int J
    cdef int max_interaction_depth
    cdef int interaction_depth
    cdef str name
    cdef str loss
    cdef double delta_loss
    cdef int n
    cdef str parallel
    cdef str pruning
    cdef str path
    cdef str bootstrap
    cdef int max_depth
    cdef str margin

    cdef float prop_root_p0

    cdef int nb_cov
    cdef bint replacement
    cdef float prop_sample
    cdef int id
    cdef float frac_valid
    cdef float epsilon

    cdef dict dic_cov
    cdef list name_cov
    cdef list kind_cov
    cdef double fit_time

    cdef object tree

    #cdef double[:,:] X
    #cdef double[:] y
    #cdef double[:] p

    #cdef double[:,:] X_train
    #cdef double[:] y_train
    #cdef double[:] p_train

    #cdef double[:,:] X_test
    #cdef double[:] y_test
    #cdef double[:] p_test

    #cdef double[:,:] X_shuffle
    #cdef double[:] y_shuffle
    #cdef double[:] p_shuffle

    #cdef np.ndarray[object, ndim=2] X
    #cdef np.ndarray[np.float64_t, ndim=1] y
    #cdef np.ndarray[np.float64_t, ndim=1] p

    #cdef np.ndarray[object, ndim=2] X_train
    #cdef np.ndarray[np.float64_t, ndim=1] y_train
    #cdef np.ndarray[np.float64_t, ndim=1] p_train

    #cdef np.ndarray[object, ndim=2] X_test
    #cdef np.ndarray[np.float64_t, ndim=1] y_test
    #cdef np.ndarray[np.float64_t, ndim=1] p_test

    #cdef np.ndarray[object, ndim=2] X_shuffle
    #cdef np.ndarray[np.float64_t, ndim=1] y_shuffle
    #cdef np.ndarray[np.float64_t, ndim=1] p_shuffle

    #cdef float prop_root_p0

    def __cinit__(self, epsilon=0, prop_root_p0=1.0, id=0, nb_cov=1, replacement=False, prop_sample=1.0, frac_valid=0.2, max_interaction_depth=0, max_depth=0, margin="absolute",
    minobs=1, delta_loss=0, loss="MSE", name=None, parallel="Yes", pruning="No", bootstrap="No"):
     #def __init__(self, float margin=0, int id=0, int nb_cov=1, bint replacement=False, float prop_sample=1.0, float frac_valid=0.2, int max_depth=0, int max_interaction_depth=0, int minobs=1, float delta_loss=0, str loss="MSE", str name=None, str parallel="Yes", str pruning="No", str bootstrap="No"):

        self.root = None

        # Initialisation of tree constraint parameters
        self.max_interaction_depth = max_interaction_depth
        self.max_depth = max_depth
        self.minobs = minobs

        # Initialisation of tree parameters
        self.tree_depth = 0
        self.nodes = []
        self.idx = 0
        self.J = 0
        self.interaction_depth = 0
        self.bootstrap = bootstrap

        self.name = name
        self.loss = loss
        self.delta_loss = delta_loss
        self.n = 0
        self.parallel = parallel
        self.pruning = pruning
        self.path = None

        # related to database
        #self.X = None
        #self.y = None
        #self.p = None

        #self.X_shuffle = None
        #self.y_shuffle = None
        #self.p_shuffle = None

        #self.X_train = None
        #self.y_train = None
        #self.p_train = None

        #self.X_test = None
        #self.y_test = None
        #self.p_test = None

        self.dic_cov = None
        self.name_cov = None
        self.kind_cov = None

        # Bootstrap parameters
        self.nb_cov = nb_cov
        self.replacement = replacement
        self.prop_sample = prop_sample
        self.id = id
        self.frac_valid = frac_valid

        # proportion and parameters for fairness constraint
        self.epsilon = epsilon
        self.margin = margin
        self.prop_root_p0 = prop_root_p0

    property nodes:
        def __get__(self):
            return self.nodes
        def __set__(self, value):
            self.nodes.append(value)

    property max_interaction_depth:
        def __get__(self):
            return self.max_interaction_depth
        def __set__(self, value):
            self.max_interaction_depth = value

    property minobs:
        def __get__(self):
            return self.minobs
        def __set__(self, value):
            self.minobs = value

    property interaction_depth:
        def __get__(self):
            return self.interaction_depth
        def __set__(self, value):
            self.interaction_depth = value

    property margin:
        def __get__(self):
            return self.margin
        def __set__(self, value):
            self.margin = value

    property delta_loss:
        def __get__(self):
            return self.delta_loss
        def __set__(self, value):
            self.delta_loss = value

    property loss:
        def __get__(self):
            return self.loss
        def __set__(self, value):
            self.loss = value

    property fit_time:
        def __get__(self):
            return self.fit_time
        def __set__(self, value):
            self.fit_time = value

    property path:
        def __get__(self):
            return self.path
        def __set__(self, value):
            self.path = value

    property nb_cov:
        def __get__(self):
            return self.nb_cov
        def __set__(self, value):
            self.nb_cov = value

    property tree_depth:
        def __get__(self):
            return self.tree_depth
        def __set__(self, value):
            self.tree_depth = value


    cpdef tuple _split_data(self, np.ndarray[object, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] p, int feature_index, float threshold, dict dic_map, dict inv_map):
        """
        Split data => function which based on numpy arrays divide the original matrix X (variable) and vectors y (response)
        and p (gender) in X_left_p0 (X_left for men => p=0), y_left_p0, p_left_p0 and same for the women in the left node
        X_left_p1 (X_left for women => p=1), y_left_p1, p_left_p1 and the same for right node.
        As this function is used in find_best_split inside of loop, using python object makes it impossible to free the GIL.
        """

        cdef str feature_type = self.kind_cov[feature_index]
        cdef np.ndarray[np.uint8_t, ndim=1] left_indices
        cdef np.ndarray[np.uint8_t, ndim=1] right_indices
        cdef np.ndarray[np.uint8_t, ndim=1] left_indices_p0
        cdef np.ndarray[np.uint8_t, ndim=1] left_indices_p1
        cdef np.ndarray[np.uint8_t, ndim=1] right_indices_p0
        cdef np.ndarray[np.uint8_t, ndim=1] right_indices_p1

        cdef np.ndarray[object, ndim=2] X_left_p0
        cdef np.ndarray[np.float64_t, ndim=1] y_left_p0
        cdef np.ndarray[np.float64_t, ndim=1] p_left_p0

        cdef np.ndarray[object, ndim=2] X_left_p1
        cdef np.ndarray[np.float64_t, ndim=1] y_left_p1
        cdef np.ndarray[np.float64_t, ndim=1] p_left_p1

        cdef np.ndarray[object, ndim=2] X_right_p0
        cdef np.ndarray[np.float64_t, ndim=1] y_right_p0
        cdef np.ndarray[np.float64_t, ndim=1] p_right_p0

        cdef np.ndarray[object, ndim=2] X_right_p1
        cdef np.ndarray[np.float64_t, ndim=1] y_right_p1
        cdef np.ndarray[np.float64_t, ndim=1] p_right_p1

        cdef np.ndarray[np.uint8_t, ndim=1] mask
        cdef str value
        cdef np.ndarray[np.float64_t, ndim=1] vector

        mask = np.array(X[:, feature_index] <= threshold)

        left_indices = mask
        right_indices = ~left_indices

        left_indices_p0 = np.logical_and(left_indices, (p == 0).reshape(-1))
        left_indices_p1 = np.logical_and(left_indices, (p == 1).reshape(-1))

        right_indices_p0 = np.logical_and(right_indices, (p == 0).reshape(-1))
        right_indices_p1 = np.logical_and(right_indices, (p == 1).reshape(-1))

        X_left_p0 = X[left_indices_p0]
        y_left_p0 = y[left_indices_p0]
        p_left_p0 = p[left_indices_p0]

        X_left_p1 = X[left_indices_p1]
        y_left_p1 = y[left_indices_p1]
        p_left_p1 = p[left_indices_p1]

        X_right_p0 = X[right_indices_p0]
        y_right_p0 = y[right_indices_p0]
        p_right_p0 = p[right_indices_p0]

        X_right_p1 = X[right_indices_p1]
        y_right_p1 = y[right_indices_p1]
        p_right_p1 = p[right_indices_p1]

        return X_left_p0, y_left_p0, p_left_p0, X_left_p1, y_left_p1, p_left_p1,\
               X_right_p0, y_right_p0, p_right_p0, X_right_p1, y_right_p1, p_right_p1


    cpdef tuple order_categorical_at_node(self, np.ndarray[object, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y, int feature_index):
        """
        Function returning mapping dictionary between categorical modalities and associated numerical mapping based on
        ordering of modality in mean response ascending order. Note that this function is called inside the first loop
        if the considered variable is categorical so making yet also impossibility to free the GIL.
        """

        cdef np.ndarray[object, ndim=1] categories = np.unique(X[:, feature_index])
        cdef list tuples_cat = []
        cdef dict dic_mapping
        cdef dict inv_mapping
        cdef int n_samples = X.shape[0]
        cdef np.ndarray[np.uint8_t, ndim=1] idx_samp_cat
        cdef np.ndarray[np.float64_t, ndim=1] means
        cdef np.ndarray[object, ndim=2] X_transf = X.copy()
        cdef int i

        for cat in categories:
            idx_samp_cat = np.array(X[:, feature_index] == cat)
            means = y[idx_samp_cat]
            tuples_cat.append((cat, np.mean(means)))

        # Sort categories based on the mean
        sorted_cat = sorted(tuples_cat, key=sort_key)

        dic_mapping = {sorted_cat[i][0]: float(i) for i in range(len(sorted_cat))}
        inv_mapping = {v: k for k, v in dic_mapping.items()}

        # Apply the mapping to the feature
        for i in range(n_samples):
            X_transf[i, feature_index] = dic_mapping[X[i, feature_index]]

        return X_transf, y, dic_mapping, inv_mapping

    cpdef tuple _find_best_split(self, np.ndarray[object, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] p):
        """
        Function returning the best split based on above both functions, It first select randomly a certain number of
        variables based on nb_cov parameter and then for each variable determine all unique value as candidates for
        threshold values. Then, for each unique value compute the associated loss and saved the feature index,
        the threshold value and the associated loss value if this latter is lower than the current one and that the
        epsilon constraint (for fairness) is fulfilled.
        """

        cdef double best_loss = np.inf
        cdef double best_dloss = 0
        cdef int best_feature_index = -1
        cdef float best_threshold = -1
        cdef dict best_feature_mapping = None
        cdef dict best_feature_mapping_inv = None

        cdef np.ndarray[np.float64_t, ndim=1] unique_val
        cdef np.ndarray[np.float64_t, ndim=1] thresholds
        cdef list retained_cov
        cdef list retained_cov_idx

        cdef str feature_type

        cdef double loss_parent
        cdef double loss_left
        cdef double loss_right
        cdef double loss
        cdef double dloss
        cdef int idx_threshold

        cdef float prop_left_p0
        cdef float prop_right_p0
        cdef float prop_root_p0
        cdef float tolerance_prop

        # Select randomly covariates (nb_cov) at each node
        # We consider only covariates with more than one unique value
        covariates = [self.name_cov[i] for i in range(X.shape[1]) if np.unique(X[:, i]).shape[0]>1]
        #print(f"Covariates ({type(covariates)}): {covariates}")
        #print(my_random_choice(list(covariates), k=int(self.nb_cov)))

        if self.nb_cov>len(covariates):
            retained_cov = covariates
        else:
            retained_cov = my_random_choice(covariates, k=int(self.nb_cov))

        #retained_cov = covariates

        #print(f"Retained: {retained_cov}")
        retained_cov_idx = [covariates.index(c) for c in retained_cov]
        #print(f"Idx: {retained_cov_idx}")

        for feature_index in retained_cov_idx:
            # for each variable in the retained variable sampled

            feature_type = self.kind_cov[feature_index]
            #print(feature_type)

            # check if the variable is categorical and if it is the case get the mapping of modalities.
            if feature_type == 'category':
                X, y, dic_map, inv_map = self.order_categorical_at_node(X, y, feature_index)
            else:
                dic_map, inv_map = None, None

            # Determine all unique value for threshold candidates.
            #print(np.array(list(sorted(np.unique(X[:, feature_index])))))
            unique_val = np.array(list(sorted(np.unique(X[:, feature_index]))))
            thresholds = (unique_val[1:] + unique_val[:-1]) / 2

            if len(thresholds)>0:
                # If threshold list not empty
                for threshold in thresholds:
                    # Split data based on this threshold value
                    #print(f"Split data")
                    X_left_p0, y_left_p0, p_left_p0,\
                    X_left_p1, y_left_p1, p_left_p1,\
                    X_right_p0, y_right_p0, p_right_p0, \
                    X_right_p1, y_right_p1, p_right_p1 = self._split_data(X, y, p, feature_index, threshold, dic_map, inv_map)

                    # Agregate
                    y_left = np.concatenate((y_left_p0, y_left_p1))
                    y_right = np.concatenate((y_right_p0, y_right_p1))

                    if y_left.shape[0]>self.minobs and y_right.shape[0]>self.minobs:
                        # If one children nodes is not empty (no observation inside after splitting) then

                        loss_parent = _loss(y)
                        loss_left = _loss(y_left)
                        loss_right = _loss(y_right)

                         # Compute proportion of men inside the left and right nodes (not need for women as variable binary)
                        prop_left_p0 = y_left_p0.shape[0] / y_left.shape[0]
                        prop_right_p0 = y_right_p0.shape[0] / y_right.shape[0]

                        # Compute the loss function and the variation of loss between parent node and children nodes
                        loss = loss_parent * y.shape[0]
                        dloss = loss_parent * y.shape[0] - (loss_left * y_left.shape[0] + loss_right*y_right.shape[0])

                    else:
                        loss = 0
                        dloss = 0
                        prop_left_p0=0
                        prop_right_p0=0

                    # Determine the tolerance in proportion based on if the margin is computed in absolute or relative way.
                    if self.margin == "absolute":
                        tolerance_prop = self.epsilon
                    else:
                        tolerance_prop = self.epsilon * self.prop_root_p0

                    if dloss > best_dloss and (np.abs(prop_left_p0 - self.prop_root_p0) <= tolerance_prop) and (np.abs(prop_right_p0 - self.prop_root_p0) <= tolerance_prop):
                        # If the loss variation is higher than the best one saved and the fairness constraint fullfilled update best split.
                        best_loss = loss
                        best_feature_index = feature_index
                        best_threshold = threshold
                        best_dloss = dloss
                        best_feature_mapping = dic_map
                        best_feature_mapping_inv = inv_map

                        #print(f"New best split found: idx: {best_feature_index}, threshold: {best_threshold}, dloss:{best_dloss}")

        return best_feature_index, best_threshold, best_dloss, best_feature_mapping, best_feature_mapping_inv

    cpdef _create_leaf_node(self, np.ndarray[np.float64_t, ndim=1] y, Node parent_node, str position):
        """
        Function creating leaf nodes (we do not need all Nodes attribute as it is terminal nodes)
        """

        cdef Node node

        # If no parent node then the leaf is the root node (it means that the whole tree contains only this node)
        if parent_node is None:
            node = Node(average_value=np.mean(y), kind="Root", depth=0, nb_samples=y.shape[0])

        else:
            node = Node(average_value=np.mean(y), kind="Leaf", depth=parent_node.depth + 1, nb_samples=y.shape[0])

        # print(f"{'  ' * (node.depth)} {node.kind}, Depth: {node.depth}, "
        #      f"Nb observations: {len(y)}, Mean_value: {np.mean(y)}")

        node.position = position
        node.parent_node = parent_node
        node.index = self.idx
        node.loss = _loss(y)
        self.nodes.append(node)

        #print(f"Leaf idx: {node.index}, Parent node: {node.parent_node.index}")

        self.idx += 1
        self.J += 1
        return node


    cpdef _build_tree(self, np.ndarray[object, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] p, Node parent_node, str position):
        """
        Function building recursively the full tree by first finding the best split for the given node, then split data
        accordingly (execute back the mapping to ensure that the X which will be used get back it modality in str).
        Verify that the conditions of depth, minobs, delta loss are fulfilled then split the observations in right and
        left node and recall this function (a random uniform variable is sampled to determine if we are starting to
        split again on left or right node).
        """

        cdef int depth
        cdef str kind
        #cdef Node node

        cdef int feature_index
        cdef double threshold
        cdef double dloss

        cdef np.ndarray[np.float64_t, ndim=1] y_left
        cdef np.ndarray[np.float64_t, ndim=1] y_right

        cdef np.ndarray[object, ndim=2] X_left
        cdef np.ndarray[object, ndim=2] X_right

        cdef np.ndarray[np.float64_t, ndim=1] p_left
        cdef np.ndarray[np.float64_t, ndim=1] p_right

        cdef np.ndarray[object, ndim=2] X_left_p0
        cdef np.ndarray[np.float64_t, ndim=1] y_left_p0
        cdef np.ndarray[np.float64_t, ndim=1] p_left_p0

        cdef np.ndarray[object, ndim=2] X_left_p1
        cdef np.ndarray[np.float64_t, ndim=1] y_left_p1
        cdef np.ndarray[np.float64_t, ndim=1] p_left_p1

        cdef np.ndarray[object, ndim=2] X_right_p0
        cdef np.ndarray[np.float64_t, ndim=1] y_right_p0
        cdef np.ndarray[np.float64_t, ndim=1] p_right_p0

        cdef np.ndarray[object, ndim=2] X_right_p1
        cdef np.ndarray[np.float64_t, ndim=1] y_right_p1
        cdef np.ndarray[np.float64_t, ndim=1] p_right_p1

        cdef double loss
        cdef double loss_left
        cdef double loss_right
        cdef double loss_decrease
        cdef int interaction_depth
        cdef double random_uniform

        cdef int size_parent = y.shape[0]

        cdef Node node
        cdef Node right_child
        cdef Node left_child

        cdef dict dic_map
        cdef dict inv_map
        cdef str feature_type

        # Determine if the node is the root one or not
        if parent_node is None:
            depth = 0
            kind = "Root"
        else:
            depth = parent_node.depth + 1
            kind = "Node"

        # Find best split for the parent node and split the data
        #print("Looking for best split ...")
        feature_index, threshold, dloss, dic_map, inv_map = self._find_best_split(X, y, p)
        #print("Best split: ", feature_index, threshold, dloss)

        # Check if a best split has been found
        if (feature_index==-1 or threshold==-1):
            node = self._create_leaf_node(y, parent_node, position)

        # Then, split the observations
        else:
            feature_type = self.kind_cov[feature_index]
            if feature_type == 'category':
                X, y, dic_map, inv_map = self.order_categorical_at_node(X, y, feature_index)

            X_left_p0, y_left_p0, p_left_p0, \
            X_left_p1, y_left_p1, p_left_p1, \
            X_right_p0, y_right_p0, p_right_p0, \
            X_right_p1, y_right_p1, p_right_p1 = self._split_data(X, y, p, feature_index, threshold, dic_map, inv_map)

            y_left = np.concatenate((y_left_p0, y_left_p1))
            y_right = np.concatenate((y_right_p0, y_right_p1))

            if feature_type == 'category':
                # reverse the matrix getting old value for modalities
                X_left_p0[:, feature_index] = [inv_map.get(value, 'unknown') for value in X_left_p0[:, feature_index]]
                X_right_p0[:, feature_index] = [inv_map.get(value, 'unknown') for value in X_right_p0[:, feature_index]]
                X_left_p1[:, feature_index] = [inv_map.get(value, 'unknown') for value in X_left_p1[:, feature_index]]
                X_right_p1[:, feature_index] = [inv_map.get(value, 'unknown') for value in X_right_p1[:, feature_index]]

            X_left = np.concatenate((X_left_p0, X_left_p1))
            X_right = np.concatenate((X_right_p0, X_right_p1))

            p_left = np.concatenate((p_left_p0, p_left_p1))
            p_right = np.concatenate((p_right_p0, p_right_p1))

            loss = _loss(y)
            loss_left = _loss(y_left)
            loss_right = _loss(y_right)

            loss_decrease = loss*<double>size_parent - (loss_left * y_left.shape[0] + loss_right * y_right.shape[0]) #/ <double>size_parent
            #loss_decrease = loss - (loss_left + loss_right) #/ <double>size_parent
            #loss_decrease = (_loss(y)*len(y) - (_loss(y_left) * len(y_left) + _loss(y_right) * len(y_right)))
            self.interaction_depth = len([c for c in self.nodes if c.kind == 'Node']) + 1
            #self.tree_depth = max([c.depth for c in self.nodes])

            #print(f"Node loss: {loss}, Node loss decrease: {loss_decrease}, Feature: {self.name_cov[feature_index]}, Threshold: {threshold}")

            # If the required condition to continue to grow the tree are met then
            if y_left.shape[0] > self.minobs and y_right.shape[0] > self.minobs and \
                    loss_decrease >= self.delta_loss and loss > 0 and self.interaction_depth<self.max_interaction_depth :# and self.tree_depth < self.max_depth + 1:
                #print("Depth: %s, Max depth: %s" % (depth, self.max_depth))
                #print(f"loss parent ({y.shape[0]}): {loss}, loss left ({y_left.shape[0]}): {loss_left}, loss right ({y_right.shape[0]}): {loss_right}, loss decrease: {loss_decrease}")

                node = Node(kind=kind, depth=depth, feature_index=feature_index, threshold=threshold, loss=loss,
                            average_value=np.mean(y), loss_decrease=loss_decrease, mapping=dic_map, nb_samples=y.shape[0])

                node.parent_node = parent_node
                node.position = position

                node.index = self.idx
                self.idx += 1
                self.nodes.append(node)

                # random variable used to know if we are starting again first in left or right node (as interaction depth = number of splitting nodes) is limited.
                random_uniform = np.random.uniform(0,1)

                if random_uniform <=0.5:
                    # Reiterate procedure for child nodes
                    left_child = self._build_tree(X_left, y_left, p_left, node, "left")
                    node.left_child = left_child
                    left_child.parent_node = node

                    right_child = self._build_tree(X_right, y_right, p_right, node, "right")
                    node.right_child = right_child
                    right_child.parent_node = node

                else:
                    # Reiterate procedure for child nodes
                    right_child = self._build_tree(X_right, y_right, p_right, node, "right")
                    node.right_child = right_child
                    right_child.parent_node = node

                    left_child = self._build_tree(X_left, y_left, p_left, node, "left")
                    node.left_child = left_child
                    left_child.parent_node = node

                print(f"{'  ' * node.depth} {node.kind}, Depth: {node.depth}, "
                      f"Feature: {node.feature_index}, Threshold: {node.threshold}, Loss: {node.loss}, "
                      f", Mean_value: {node.average_value}")

            # If not grow a leaf
            else:
                node = self._create_leaf_node(y, parent_node, position)

        return node

    cdef tuple sample_arrays(self, np.ndarray[object, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] p, float prop_sample, bint replacement):
        """
        Function used to sample randomly the original X_train, y_train, p_train for the boostrap process before fitting the tree.
        """

        cdef int num_samples = int(X.shape[0] * prop_sample)
        cdef np.ndarray[object, ndim=2] sampled_x
        cdef np.ndarray[np.float64_t, ndim=1] sampled_y
        cdef np.ndarray[np.float64_t, ndim=1] sampled_p

        # if replacement=True then sample with replacement else not.
        if replacement:
            sampled_indices = np.random.choice(X.shape[0], size=num_samples, replace=True)
        else:
            sampled_indices = np.random.choice(X.shape[0], size=num_samples, replace=False)

        sampled_x = X[sampled_indices, :]
        sampled_y = y[sampled_indices]
        sampled_p = p[sampled_indices]
        return sampled_x, sampled_y, sampled_p

    cpdef fit(self, np.ndarray[object, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] p, dict dic_cov):
        """
        Function using the X_train, y_train, p_train, sample it (boostrap process) and then call build_tree to start
        building the tree.
        """

        cdef double time_elapsed
        cdef double time_start = time.time()
        cdef double time_end

        cdef np.ndarray[object, ndim=2] X_train
        cdef np.ndarray[np.float64_t, ndim=1] y_train
        cdef np.ndarray[np.float64_t, ndim=1] p_train

        cdef np.ndarray[object, ndim=2] X_shuffle
        cdef np.ndarray[np.float64_t, ndim=1] y_shuffle
        cdef np.ndarray[np.float64_t, ndim=1] p_shuffle

        cdef np.ndarray[object, ndim=2] X_processed = X.copy() #preprocess_data(X, dic_cov)
        self.root = None

        #if len([c for c in dic_cov.keys() if c!='categorical' and c!='float64'])>0:
            #X_processed = preprocess_data(X, dic_cov)

        #self.X = X
        #self.y = y
        #self.p = p

        self.n = X.shape[0]

        self.dic_cov = dic_cov
        self.name_cov = list(dic_cov.keys())
        self.kind_cov = list(dic_cov.values())

        # We could transform categorical into integers or do not change anything
        #self.X, self.y, self.dic_mapping = self.preprocess_categorical_numpy(X, y, self.kind_cov)

        # Random sampling of the original observations with replacement
        if self.bootstrap == "Yes":
            print("Boostraping ...")
            X_shuffle, y_shuffle, p_shuffle = self.sample_arrays(X_processed, y, p, self.prop_sample, self.replacement)
        else:
            X_shuffle, y_shuffle, p_shuffle = X_processed, y, p

        # Defining training and testing sets
        X_train = X_shuffle[:int(self.frac_valid * self.n)]
        y_train = y_shuffle[:int(self.frac_valid * self.n)]
        p_train = p_shuffle[:int(self.frac_valid * self.n)]

        X_test = X_shuffle[int(self.frac_valid * self.n):]
        y_test = y_shuffle[int(self.frac_valid * self.n):]
        p_test = p_shuffle[int(self.frac_valid * self.n):]

        y_train = y_train.reshape(-1)
        p_train = p_train.reshape(-1)


        #self.prop_root_p0 = np.sum((np.array(p_train) == 0) * 1) / X_train.shape[0]

        #print(X.shape)
        #print(f"Frac_valid={self.frac_valid}")
        #print(f"n={self.n}")
        #print(X_shuffle.shape)
        #print(X_train.shape)
        #print(X_test.shape)

        #print(f"prop_0={self.prop_root_p0}")
        #quit()

        # Start building the tree
        #print(f"Starting building the tree ...")
        self.tree = self._build_tree(X_train, y_train, p_train, self.root, "None")
        self.tree_depth = max([c.depth for c in self.nodes])
        self.interaction_depth = len([c for c in self.nodes if c.kind=='Node'])+1

        time_end = time.time()
        time_elapsed = (time_end - time_start)
        self.fit_time = time_elapsed

        print("\n")
        print('*******************************')
        print(f"Tree {self.id}: Params(id={self.max_interaction_depth}, cov={self.nb_cov})")
        print("Time elapsed: %s" % self.fit_time)
        print("Tree depth: %s" % self.tree_depth)
        print(f"Nb nodes: {len(self.nodes)}")
        print('*******************************')

        if self.pruning == "Yes":
            # Prune the tree
            time_start_pruning = time.time()
            self.prune()
            time_elapsed_pruning = (time.time() - time_start_pruning)
            self.prune_time = time_elapsed_pruning
            print("\n")
            print('*******************************')
            print("Time elapsed for pruning: %s" % time_elapsed_pruning)
            print("New Tree depth: %s" % self.tree_depth)
            print('*******************************')

    def _predict_instance(self, np.ndarray[object, ndim=1] x, Node node):
        """
        Function used to make a prediction based on a single observation x \in X at a specific node of the tree.
        If the node is not a terminal node then, the observations continues is path inside the tree until finishing in
        the leaf where the prediction associated is the mean response observed (attribute average_value).
        """

        cdef double val

        # if node is a terminal node return average value as the prediction
        if node.kind == "Leaf" or (len(self.nodes)==1 and node.kind == "Root"):
            return node.average_value

        else:
            # if not the case split the observations in the different nodes of the tree based on splitting rules (threshold
            #associated at each node.

            #print(node.kind, x[node.feature_index], node.mapping_categorical)
            if node.mapping_categorical is not None:
                #print(f"Original: {x[node.feature_index]}, Mapped: {np.vectorize(node.mapping_categorical.get)(x[node.feature_index])}")
                if x[node.feature_index] not in node.mapping_categorical.keys():
                    return node.average_value
                else:
                    val = np.vectorize(node.mapping_categorical.get)(x[node.feature_index])
            else:
                val = x[node.feature_index]

            try:
                if val <= node.threshold:
                    pass
            except TypeError as e:
                print(f"TypeError occurred: {e}")
                print("val:", val)
                print("val2:", node.threshold)
                print("Type of val:", type(val))
                print("Type of threshold:", type(node.threshold))
                print("Mapping:", node.mapping_categorical)
                return node.average_value


            if val <= node.threshold:
                return self._predict_instance(x, node.left_child)
            else:
                return self._predict_instance(x, node.right_child)

    def predict(self, np.ndarray[object, ndim=2] X):
        """
        Based on a matrix X, make prediction for the whole observation by calling predict_instance
        """

        return [self._predict_instance(x, self.nodes[0]) for x in X]


    def display_tree(self, int depth):
        """
        Function only python based which draw with matplotlib the tree with all nodes and associated characteritics.
        """

        cdef Node node, parent
        cdef int idx_parent
        cdef double radius
        cdef double spacing
        cdef double font_size
        cdef double arrow_width
        cdef list coord_nodes = []
        cdef list idx_nodes = []
        cdef double min_X, max_X
        cdef double X0, Y0, X, Y, X_parent, Y_parent

        nodes = [node for node in self.nodes if node.depth <= depth]

        radius = 0.08 / (2 ** depth + 1)  # 0.05
        side_length = 2.5 * radius  # 0.06

        X0 = 0.5
        Y0 = 0.9
        d = 0
        spacing = radius * 2 ** (depth + 1)  # len(cart.nodes)
        font_size = 72 / len(nodes)
        arrow_width = 0.005 / (depth)

        X, Y = X0, Y0

        fig, ax = plt.subplots(figsize=(8.0, 5.0))

        ax.text(X0, Y0 + 0.05,
                self.name,
                va='center', ha='center', fontsize=12)

        for node in nodes:
            # print(f"Depth: {node.depth}, {node.kind}, Child: ({node.left_child.kind}, {node.right_child.kind})")

            parent = node.parent_node

            if parent is not None:
                idx_parent = parent.index
                X_parent, Y_parent = coord_nodes[idx_nodes.index(idx_parent)]

            else:
                X_parent = X
                Y_parent = Y

            if node.kind == "Root":
                square = patches.Rectangle((X, Y), side_length, side_length, facecolor='lightblue')
                ax.add_patch(square)

            else:
                Y = 0.9 - node.depth * 0.1

                if node.position == "left":
                    X = X_parent - spacing / (parent.depth + 1)
                else:
                    X = X_parent + spacing / (parent.depth + 1)

                if node.kind == "Leaf":
                    circle = patches.Circle((X, Y), radius, facecolor='lightgreen')
                else:
                    circle = patches.Circle((X, Y), radius, facecolor='lightblue')

                ax.add_patch(circle)

            coord_nodes.append([X, Y])
            idx_nodes.append(node.index)

            if X_parent != X and Y_parent != Y:
                if parent.kind != "Root":
                    if node.position == "left" and parent is not None:
                        ax.arrow(X_parent, Y_parent - radius, X - X_parent + 3 * arrow_width,
                                 Y - Y_parent + 2 * radius + (depth + 1) * 1 * arrow_width,
                                 width=arrow_width, color="black")
                        ax.text(X_parent - (X_parent - X + 2 * radius) / 2, Y + (Y_parent - Y + 0.75 * radius) / 2,
                                f"X{parent.feature_index}" + r"$\leq$" + f"{np.round(parent.threshold, 2)}",
                                va='center', ha='center', fontsize=font_size)

                    elif node.position == "right" and parent is not None:
                        ax.arrow(X_parent, Y_parent - radius, X - X_parent - 3 * arrow_width,
                                 Y - Y_parent + 2 * radius + (depth + 1) * 1 * arrow_width,
                                 width=arrow_width, color="black")
                        ax.text(X - (X - X_parent - 2 * radius) / 2, Y + (Y_parent - Y + 0.75 * radius) / 2,
                                f"X{parent.feature_index}" + ">" + f"{np.round(parent.threshold, 2)}",
                                va='center', ha='center', fontsize=font_size)

                else:

                    if node.position == "left" and parent is not None:
                        ax.arrow(X_parent + side_length / 2, Y_parent, X - X_parent - side_length / 2 + 3 * arrow_width,
                                 Y - Y_parent + radius + (depth + 1) * 0.75 * arrow_width,
                                 width=arrow_width, color="black")
                        ax.text(X_parent - (X_parent - X + 2 * radius) / 2, Y + (Y_parent - Y + 1.5 * radius) / 2,
                                f"X{parent.feature_index}" + r"$\leq$" + f"{np.round(parent.threshold, 2)}",
                                va='center', ha='center', fontsize=font_size)

                    elif node.position == "right" and parent is not None:
                        ax.arrow(X_parent + side_length / 2, Y_parent, X - X_parent - side_length / 2 - 3 * arrow_width,
                                 Y - Y_parent + radius + (depth + 1) * 0.75 * arrow_width, width=arrow_width,
                                 color="black")
                        ax.text(X - (X - X_parent - 2 * 2 * radius) / 2, Y + (Y_parent - Y + 1.5 * radius) / 2,
                                f"X{parent.feature_index}" + ">" + f"{np.round(parent.threshold, 2)}",
                                va='center', ha='center', fontsize=font_size)

            # print(node.index)

            if node.kind != 'Root':
                print(f"{node.kind}, Position: {node.position}, Depth: {node.depth}, "
                      f"Index: {node.index}, Parent_index: {node.parent_node.index}, {(X, Y)}")
                if node.kind != "Leaf":
                    ax.text(X, Y, f"{node.kind} \n" + r"$\widehat{y}$" + f"={np.round(node.average_value, 3)}  \n" +
                            r"$\mathcal{L}$" + f"={np.round(node.loss, 3)} \n N={node.nb_samples}",
                            va='center', ha='center', fontsize=font_size)
                else:
                    ax.text(X, Y,
                            f"{node.kind} \n" + r"$\widehat{y}$" + f"={np.round(node.average_value, 3)} \n N={node.nb_samples}",
                            va='center', ha='center', fontsize=font_size)

            else:
                print(f"{node.kind}, Position: {node.position}, Depth: {node.depth}, "
                      f"Index: {node.index}, {(X, Y)}")
                if node.kind != "Leaf" and node.loss is not None:
                    ax.text(X + side_length / 2, Y + side_length / 2,
                            f"{node.kind} \n" + r"$\widehat{y}$" + f"={np.round(node.average_value, 3)} \n" +
                            r"$\mathcal{L}$" + f"={np.round(node.loss, 3)} \n N={node.nb_samples}",
                            va='center', ha='center', fontsize=font_size)
                else:
                    ax.text(X + side_length / 2, Y + side_length / 2,
                            f"{node.kind} \n" + r"$\widehat{y}$" + f"={np.round(node.average_value, 3)} \n N={node.nb_samples}",
                            va='center', ha='center', fontsize=font_size)

        # plot depth
        min_X = min([c[0] for c in coord_nodes])
        max_X = max([c[0] for c in coord_nodes])

        for depth in range(self.tree_depth + 1):
            Y = Y0 - depth * 0.1
            ax.text(min_X - spacing / 2, Y,
                    f"Depth: {depth}", va='center', ha='center', fontsize=12)

        plt.axis('off')
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(8, 6)


        if self.loss == 'poisson':
            if self.n > 100000:
                plt.savefig(self.path + "T_" + str(self.id) + "_d_" + str(self.max_interaction_depth) + "_cov_" + str(
                    self.nb_cov) + "_m_" +
                            str(self.epsilon) + "_2.jpg", dpi=100)
            else:
                plt.savefig(self.path + "T_" + str(self.id) + "_d_" + str(self.max_interaction_depth) + "_cov_" + str(
                    self.nb_cov) + "_m_" +
                            str(self.epsilon) + "_1.jpg", dpi=100)

        print(f"Saved in {self.path}")
        plt.close()
        #plt.show()

    def compute_importance2(self):
        """
        Function which compute importance of each variable in the tree, based on the loss reduction observed at each
        node of the tree where each variable is used. Return a vector of importance (importance of each variable).
        """

        list_imp = []
        for idx in range(len(self.kind_cov)):
            loss_reduction_feature = 0
            for node in self.nodes:
                # print(f"Node Index {node.index}, Feature_index: {feature_index}")
                if node.feature_index == idx:
                    importance = node.loss*node.nb_samples - (node.left_child.loss*node.left_child.nb_samples+node.right_child.loss*node.right_child.nb_samples)
                    loss_reduction_feature += importance / self.n
            list_imp.append(loss_reduction_feature)
        list_imp = np.array(list_imp) * 100 / np.sum(list_imp)
        return list(list_imp)
