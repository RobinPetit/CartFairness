import numpy as np
import pandas as pd
# import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import mean_poisson_deviance
#from tqdm import tqdm
import time
import timeit
from load_data import load_dataset
from itertools import combinations
from math import log
import matplotlib.patches as patches
np.set_printoptions(precision=18)


class Node:
    """
    Node class being elements of CARTRegressor_python. The characteristics of the nodes are its kind (Root for the first node
    (on top of the tree), Node as splitting node or Leaf as terminal node), feature index, threshold, and associated loss.
    The average value denotes the prediction associated with this node as the mean response observed on the training observations.
    Mapping categorical denotes the mapping (dictionary) between modalities of the categorical variable and integer value used for splitting.
    Integer values are determined in ordering the mean response for each modality in ascending order. Note that The mapping
    or a same variable could be different at each node of the tree depending on the modalities present (depending on observations)
    in the nodes as well as mean response observed that could be different based on the observations. Other attributes are
    saved as the number of observations in the node, the depth of the node in the tree, the parent node and children nodes
    of the current node.
    """
    def __init__(self, kind=None, depth=None, feature_index=None, threshold=None, loss=None, average_value=None, loss_decrease=None, mapping=None, nb_samples=None):
        # Main characteristic of the node (kind=Node/Leaf) split feature, split threshold, split mse, split samples)
        self.kind = kind
        self.feature_index = feature_index
        self.threshold = threshold
        self.loss = loss
        self.average_value = average_value
        self.mapping_categorical = mapping
        self.nb_samples = nb_samples

        # Secondary characteristic allowing to identify the position of the node in the tree
        self.depth = depth
        self.parent_node = None
        self.left_child = None
        self.right_child = None
        self.position = None
        self.index = None
        self.loss_decrease = loss_decrease


def all_combinations(strings):
    result = []
    nb_max_groupment = 1

    if len(strings)<=nb_max_groupment:
        for r in range(1, len(strings) + 1):  # Generate combinations of different lengths
            #print(strings, r)
            #print(list(combinations(strings, r)))
            result.extend(list(combinations(strings, r)))

        return result[:-1]

    else:
        for r in range(1, nb_max_groupment + 1):  # Generate combinations of different lengths
            #print(strings, r)
            #print(list(combinations(strings, r)))
            result.extend(list(combinations(strings, r)))

        return result


class CARTRegressor_python:
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
    def __init__(self, epsilon=0, id=0, nb_cov=1, replacement=False, prop_sample=1.0, frac_valid=0.2, max_depth=0, max_interaction_depth=0, minobs=1, delta_loss=0, loss="MSE", name=None, parallel="Yes", pruning="No", bootstrap="No", **kwargs):
        self.root = None

        # Initialisation of tree constraint parameters
        self.max_depth = max_depth
        self.max_interaction_depth = max_interaction_depth
        self.minobs = minobs
        self.bootstrap = bootstrap

        # Initialisation of tree parameters
        self.tree_depth = 0
        self.nodes = []
        self.idx = 0
        self.J = 0
        self.interaction_depth = 0

        self.name = name
        self.loss = loss
        self.delta_loss = delta_loss
        self.n = 0
        self.parallel = parallel
        self.pruning = pruning
        self.path = None
        self.dic_mapping = None
        self.fit_time = None

        # related to database
        self.X_shuffle = None
        self.y_shuffle = None
        self.p_shuffle = None

        self.X_train = None
        self.y_train = None
        self.p_train = None

        self.X_test = None
        self.y_test = None
        self.p_test = None

        # Bootstrap parameters
        self.nb_cov = nb_cov
        self.replacement = replacement
        self.prop_sample = prop_sample
        self.id = id
        self.frac_valid = frac_valid

        # proportion to the discrimination
        self.epsilon = epsilon

    def _loss(self, y):
        # compute loss function chosen

        # Definition of the usual loss functions
        def _mse(y):
            mean = np.mean(y, dtype=np.float64)
            mse = np.mean((y - mean) ** 2, dtype=np.float64)
            return mse

        def _poisson_deviance_simp2(y, y_pred):
            epsilon = 10 ** -18
            dev = 0.0
            n = y.shape[0]

            # Check if the array is empty
            if n == 0:
                raise ValueError("Input arrays cannot be empty.")

            # Iterate over the arrays and calculate the deviance
            for i in range(n):
                # Ensure that neither y[i] nor y_pred[i] is too small or zero
                if y[i] > epsilon and y_pred[i] > epsilon:
                    dev += 2 * (y[i] * np.log((y[i] + epsilon) / (y_pred[i] + epsilon)) + (y_pred[i] - y[i]))
                else:
                    # Handle the case where either y[i] or y_pred[i] is 0 or too small
                    dev += 0.0  # Optionally: handle as zero contribution, or use a default small value

            # Return the deviance divided by the number of elements
            return dev / n

        if self.loss == "MSE":
            return _mse(y)
        elif self.loss == "poisson":
            y = np.array(y)
            y_pred = np.repeat(np.mean(y, dtype=np.float64), y.shape[0])
            return _poisson_deviance_simp2(y, y_pred)

    def order_categorical_at_node(self, X, y, feature_index):
        """
        Function returning mapping dictionary between categorical modalities and associated numerical mapping based on
        ordering of modality in mean response ascending order. Note that this function is called inside the first loop
        if the considered variable is categorical so making yet also impossibility to free the GIL.
        """

        tuples_cat = []
        n_samples = X.shape[0]
        X_transf = X.copy()
        categories = np.unique(X[:, feature_index])

        for cat in categories:
            idx_samp_cat = np.array(X[:, feature_index] == cat)
            means = y[idx_samp_cat]
            tuples_cat.append((cat, np.mean(means)))

        # Sort categories based on the mean
        sorted_cat = sorted(tuples_cat, key=lambda x: x[1])

        dic_mapping = {sorted_cat[i][0]: float(i) for i in range(len(sorted_cat))}
        inv_mapping = {v: k for k, v in dic_mapping.items()}

        # Apply the mapping to the feature
        for i in range(n_samples):
            X_transf[i, feature_index] = dic_mapping[X[i, feature_index]]

        return X_transf, y, dic_mapping, inv_mapping

    def _split_data(self, X, y, p, feature_index, threshold):
        """
       Split data => function which based on numpy arrays divide the original matrix X (variable) and vectors y (response)
       and p (gender) in X_left_p0 (X_left for men => p=0), y_left_p0, p_left_p0 and same for the women in the left node
       X_left_p1 (X_left for women => p=1), y_left_p1, p_left_p1 and the same for right node.
       As this function is used in find_best_split inside of loop, using python object makes it impossible to free the GIL.
       """

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

        return X_left_p0, y_left_p0, p_left_p0, \
               X_left_p1, y_left_p1, p_left_p1,\
               X_right_p0, y_right_p0, p_right_p0,\
               X_right_p1, y_right_p1, p_right_p1

    def _find_best_split(self, X, y, p):
        """
        Function returning the best split based on above both functions, It first select randomly a certain number of
        variables based on nb_cov parameter and then for each variable determine all unique value as candidates for
        threshold values. Then, for each unique value compute the associated loss and saved the feature index,
        the threshold value and the associated loss value if this latter is lower than the current one and that the
        epsilon constraint (for fairness) is fulfilled.
        """
        best_loss = np.inf
        best_dloss = 0.0
        best_feature_index = None
        best_threshold = None
        best_feature_mapping = None
        best_feature_mapping_inv = None

        # Select randomly covariates (nb_cov) at each node
        # We consider only covariates with more than one unique value
        covariates = [self.name_cov[i] for i in range(X.shape[1]) if len(np.unique(X[:, i]))>1]
        #print(self.nb_cov, len(covariates), covariates)
        if self.nb_cov >= len(covariates):
            retained_cov = covariates
        else:
            retained_cov = np.random.choice(covariates, self.nb_cov, replace=False)

        #retained_cov = covariates
        #print(f"Retained cov: {retained_cov}")
        retained_cov_idx = [covariates.index(c) for c in retained_cov]

        for feature_index in retained_cov_idx:
            # for each variable in the retained variable sampled
            feature_type = self.kind_cov[feature_index]

            # check if the variable is categorical and if it is the case get the mapping of modalities.
            if feature_type == 'category':
                X, y, dic_map, inv_map = self.order_categorical_at_node(X, y, feature_index)
            else:
                dic_map, inv_map = None, None

            # Determine all unique value for threshold candidates.
            unique_val = np.array(list(sorted(np.unique(X[:, feature_index]))))
            thresholds = (unique_val[1:] + unique_val[:-1]) / 2

            if len(thresholds)>0:
                # If threshold list not empty
                for threshold in thresholds:
                    # Split data based on this threshold value
                    X_left_p0, y_left_p0, p_left_p0,\
                    X_left_p1, y_left_p1, p_left_p1,\
                    X_right_p0, y_right_p0, p_right_p0, \
                    X_right_p1, y_right_p1, p_right_p1 = self._split_data(X, y, p, feature_index, threshold)

                    # Agregate
                    y_left = np.concatenate((y_left_p0, y_left_p1))
                    y_right = np.concatenate((y_right_p0, y_right_p1))

                    if y_left.shape[0]>self.minobs and y_right.shape[0]>self.minobs:
                        # If one children nodes is not empty (no observation inside after splitting) then
                        loss_parent = self._loss(y)
                        loss_left = self._loss(y_left)
                        loss_right = self._loss(y_right)

                        # Compute proportion of men inside the left and right nodes (not need for women as variable binary)
                        prop_left_p0 = y_left_p0.shape[0] / y_left.shape[0]
                        prop_right_p0 = y_right_p0.shape[0] / y_right.shape[0]

                        # Compute the loss function and the variation of loss between parent node and children nodes
                        loss = loss_left * y_left.shape[0] + loss_right * y_right.shape[0]
                        dloss = loss_parent*y.shape[0]
                        dloss -= (loss_left * y_left.shape[0] + loss_right * y_right.shape[0])

                    else:
                        loss = 0
                        dloss = 0
                        prop_left_p0=0
                        prop_right_p0=0

                    prop_root_p0 = np.sum((np.array(self.p) == 0) * 1) / self.n

                    # print(dloss, best_dloss, prop_left_p0, prop_root_p0, prop_root_p0, end=''); input()
                    if dloss > best_dloss and \
                            (np.abs(prop_left_p0 - prop_root_p0) <= self.epsilon * prop_root_p0) and \
                            (np.abs(prop_right_p0 - prop_root_p0) <= self.epsilon * prop_root_p0):
                        # If the loss variation is higher than the best one saved and the fairness constraint fullfilled update best split.
                        best_loss = loss
                        best_feature_index = feature_index
                        best_threshold = threshold
                        best_dloss = dloss
                        best_feature_mapping = dic_map
                        best_feature_mapping_inv = inv_map

                        #print(f"New best split found: idx: {best_feature_index}, threshold: {best_threshold}, dloss:{best_dloss}")

        return best_feature_index, best_threshold, best_dloss, best_feature_mapping, best_feature_mapping_inv

    def _create_leaf_node(self, y, parent_node, position):
        """
        Function creating leaf nodes (we do not need all Nodes attribute as it is terminal nodes)
        """
        if parent_node is None:
            node = Node(average_value=np.mean(y), kind="Root", depth=0, nb_samples=y.shape[0])
        else:
            node = Node(average_value=np.mean(y), kind="Leaf", depth=parent_node.depth + 1, nb_samples=y.shape[0])
            #node.index = parent_node.index

        # print(f"{'  ' * (node.depth)} {node.kind}, Depth: {node.depth}, "
        #      f"Nb observations: {len(y)}, Mean_value: {np.mean(y)}")

        node.position = position
        node.parent_node = parent_node
        node.index = self.idx
        node.loss = self._loss(y)
        self.nodes.append(node)

        #print(f"Leaf idx: {node.index}, Parent node: {node.parent_node.index}")

        self.idx += 1
        self.J += 1
        return node

    def _build_tree(self, X, y, p, parent_node, position):
        """
        Function building recursively the full tree by first finding the best split for the given node, then split data
        accordingly (execute back the mapping to ensure that the X which will be used get back it modality in str).
        Verify that the conditions of depth, minobs, delta loss are fulfilled then split the observations in right and
        left node and recall this function (a random uniform variable is sampled to determine if we are starting to
        split again on left or right node).
        """

        # Determine if the node is the root one or not
        if parent_node is None:
            depth = 0
            kind = "Root"
        else:
            depth = parent_node.depth + 1
            kind = "Node"

        # Find best split for the parent node and split the data
        feature_index, threshold, dloss, dic_map, inv_map = self._find_best_split(X, y, p)
        #print("Best split: ", feature_index, threshold, dloss)

        # Check if a best split has been found
        if (feature_index is None or threshold is None): # or loss ==0 or loss is None
            node = self._create_leaf_node(y, parent_node, position)

        # Then, split the observations
        else:
            feature_type = self.kind_cov[feature_index]
            if feature_type == 'category':
                X, y, dic_map, inv_map = self.order_categorical_at_node(X, y, feature_index)

            X_left_p0, y_left_p0, p_left_p0, \
            X_left_p1, y_left_p1, p_left_p1, \
            X_right_p0, y_right_p0, p_right_p0, \
            X_right_p1, y_right_p1, p_right_p1 = self._split_data(X, y, p, feature_index, threshold)

            y_left = np.concatenate((y_left_p0, y_left_p1))
            y_right = np.concatenate((y_right_p0, y_right_p1))

            if feature_type == 'category':
                # reverse the matrix getting old value for modalities
                X_left_p0[:, feature_index] = [inv_map.get(value, 'unknown') for value in
                                               X_left_p0[:, feature_index]]
                X_right_p0[:, feature_index] = [inv_map.get(value, 'unknown') for value in
                                                X_right_p0[:, feature_index]]
                X_left_p1[:, feature_index] = [inv_map.get(value, 'unknown') for value in
                                               X_left_p1[:, feature_index]]
                X_right_p1[:, feature_index] = [inv_map.get(value, 'unknown') for value in
                                                X_right_p1[:, feature_index]]

            X_left = np.concatenate((X_left_p0, X_left_p1))
            X_right = np.concatenate((X_right_p0, X_right_p1))
            p_left = np.concatenate((p_left_p0, p_left_p1))
            p_right = np.concatenate((p_right_p0, p_right_p1))

            loss = self._loss(y)
            loss_left = self._loss(y_left)
            loss_right = self._loss(y_right)

            loss_decrease = loss*y.shape[0] - (loss_left * y_left.shape[0] + loss_right * y_right.shape[0])
            self.interaction_depth = len([c for c in self.nodes if c.kind == 'Node']) + 1


            # If the required condition to continue to grow the tree are met then
            if y_left.shape[0] > self.minobs and y_right.shape[0] > self.minobs and \
                    loss_decrease >= self.delta_loss and loss > 0 and self.interaction_depth<self.max_interaction_depth+1:

                #print(f"loss parent ({y.shape[0]}): {loss}, loss left ({y_left.shape[0]}): {loss_left}, loss right ({y_right.shape[0]}): {loss_right}, loss decrease: {loss_decrease}")

                node = Node(feature_index=feature_index, threshold=threshold, loss=loss, kind=kind,
                            average_value=np.mean(y, dtype="float64"), depth=depth, loss_decrease=loss_decrease,
                            mapping=dic_map, nb_samples=y.shape[0])

                node.parent_node = parent_node
                node.position = position

                node.index = self.idx
                self.idx += 1

                self.nodes.append(node)

                print(f"{'  ' * node.depth} {node.kind}, Depth: {node.depth}, "
                      f"Feature: {node.feature_index}, Threshold: {node.threshold}, DLoss: {node.loss_decrease}"
                      f", Mean_value: {node.average_value}")

                # Reiterate procedure for child nodes
                left_child = self._build_tree(X_left, y_left, p_left, node, "left")
                node.left_child = left_child
                left_child.parent_node = node

                right_child = self._build_tree(X_right, y_right, p_right, node, "right")
                node.right_child = right_child
                right_child.parent_node = node

            # If not grow a leaf
            else:
                node = self._create_leaf_node(y, parent_node, position)

        return node

    def sample_arrays(self, X, y, p, prop_sample, replacement):
        """
        Function used to sample randomly the original X_train, y_train, p_train for the boostrap process before fitting the tree.
        """

        num_samples = int(X.shape[0] * prop_sample)

        if replacement:
            sampled_indices = np.random.choice(X.shape[0], size=num_samples, replace=True)
        else:
            sampled_indices = np.random.choice(X.shape[0], size=num_samples, replace=False)

        print(sampled_indices)

        sampled_x = X[sampled_indices, :]
        sampled_y = y[sampled_indices]
        sampled_p = p[sampled_indices]

        return sampled_x, sampled_y, sampled_p

    def fit(self, X, y, p, dic_cov):
        """
        Function using the X_train, y_train, p_train, sample it (boostrap process) and then call build_tree to start
        building the tree.
        """

        time_start = time.time()

        self.root = None

        #X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)

        self.X = X
        self.y = y
        self.p = p
        self.n = X.shape[0]
        self.dic_cov = dic_cov
        self.name_cov = list(dic_cov.keys())
        self.kind_cov = list(dic_cov.values())

        # We could transform categorical into integers or do not change anything
        #self.X, self.y, self.dic_mapping = self.preprocess_categorical_numpy(X, y, self.kind_cov)

        # Random sampling of the original observations with replacement
        if self.bootstrap == "Yes":
            X_shuffle, y_shuffle, p_shuffle = self.sample_arrays(X, y, p, self.prop_sample, self.replacement)
        else:
            X_shuffle, y_shuffle, p_shuffle = X, y, p


        X_shuffle = np.asarray(X_shuffle)#, dtype=np.float64)

        # Defining training and testing sets
        X_train = X_shuffle[:int(self.frac_valid * self.n)]
        y_train = y_shuffle[:int(self.frac_valid * self.n)]
        p_train = p_shuffle[:int(self.frac_valid * self.n)]

        y_train = y_train.reshape(-1)
        p_train = p_train.reshape(-1)

        X_test = X_shuffle[int(self.frac_valid * self.n):]
        y_test = y_shuffle[int(self.frac_valid * self.n):]
        p_test = p_shuffle[int(self.frac_valid * self.n):]

        # Start building the tree
        #print(f"Starting building the tree ...")
        self.tree = self._build_tree(X_train, y_train, p_train, self.root, None)
        self.tree_depth = max([c.depth for c in self.nodes])
        self.interaction_depth = len([c for c in self.nodes if c.kind=='Node'])+1

        time_elapsed = (time.time() - time_start)
        self.fit_time = time_elapsed
        print("\n")
        print('*******************************')
        print(f"Tree {self.id}: Params(id={self.max_interaction_depth}, cov={self.nb_cov})")
        print("Time elapsed: %s" % time_elapsed)
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

    def _predict_instance(self, x, node):
        r"""
        Function used to make a prediction based on a single observation x \in X at a specific node of the tree.
        If the node is not a terminal node then, the observations continues is path inside the tree until finishing in
        the leaf where the prediction associated is the mean response observed (attribute average_value).
        """

        if node.kind == "Leaf" or (len(self.nodes)==1 and node.kind == "Root"):
            return node.average_value

        else:
            if node.mapping_categorical is not None:
                #print(f"Original: {x[node.feature_index]}, Mapped: {np.vectorize(node.mapping_categorical.get)(x[node.feature_index])}")
                if x[node.feature_index] not in node.mapping_categorical.keys():
                    return node.average_value
                else:
                    val = np.vectorize(node.mapping_categorical.get)(x[node.feature_index])
            else:
                val = x[node.feature_index]

            if val <= node.threshold:
                return self._predict_instance(x, node.left_child)
            else:
                return self._predict_instance(x, node.right_child)

    def predict(self, X):
        """
        Based on a matrix X, make prediction for the whole observation by calling predict_instance
        """
        return [self._predict_instance(x, self.nodes[0]) for x in X]

    def display_tree(self, depth):
        """
        Function only python based which draw with matplotlib the tree with all nodes and associated characteritics.
        """
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
        coord_nodes = []
        idx_nodes = []

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

            # idx =

            # X_last = X
            # Y_last = Y

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

        # plt.show()

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
                    importance = node.loss * node.nb_samples - (
                                node.left_child.loss * node.left_child.nb_samples + node.right_child.loss * node.right_child.nb_samples)
                    loss_reduction_feature += importance / self.n
            list_imp.append(loss_reduction_feature)

        list_imp = np.array(list_imp) * 100 / np.sum(list_imp)
        return list(list_imp)
