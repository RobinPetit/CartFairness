# compare.py
# from version_simple_cython21_adj_categorical_clean_epsilon import CARTRegressor_cython, preprocess_data
from version_simple_python_cat_clean_epsilon import CARTRegressor_python
import time
from load_data import load_dataset_charpentier
import numpy as np

from CART import CART as NewCART
from dataset import Dataset
# from load_data import load_dataset
import pandas as pd
from math import log

list_time_cython_obs = []
list_time_cython2_obs = []
list_time_cython3_obs = []
list_time_sklearn_obs = []
range_nb_obs = range(25000, 100001, 250000)


def prepare_dataset(nb_observation):
    # nb_observation = 100_000
    df_fictif, col_features, col_response, col_protected = load_dataset_charpentier(nb_obs=nb_observation, verbose=VERBOSE)
    # df_fictif, col_features, col_response, col_protected = load_dataset(nb_obs=nb_observation,
                                                                                    #verbose=VERBOSE)
    df_fictif.dropna(inplace=True)

    print(col_features)
    # quit()

    # print(df_fictif.dtypes)
    # quit()

    col_features = ['VehPower', 'VehAge', 'DrivAge', 'Density', 'BonusMalus']
    col_features = ['VehBrand', 'Area', 'Region']

    df_fictif = df_fictif[np.concatenate((col_features, col_response, col_protected))]

    if VERBOSE:
        print(df_fictif)

    frac_train = 0.7

    df_train = df_fictif.iloc[:int(len(df_fictif)*frac_train), :]
    df_test = df_fictif.iloc[int(len(df_fictif)*frac_train):, :]

    df_training = df_train.iloc[:int(len(df_train)*frac_train), :]
    df_testing = df_train.iloc[int(len(df_train)*frac_train):, :]

    # Splitting adequately the sets
    X_train = df_train[col_features].values
    y_train = df_train[col_response].values
    p_train = df_train[col_protected].values
    p_train = p_train.astype(np.float64).reshape(-1)
    y_train = y_train.astype(np.float64).reshape(-1)

    X_test = df_test[col_features].values
    y_test = df_test[col_response].values
    p_test = df_test[col_protected].values
    p_test = p_test.astype(np.float64).reshape(-1)
    y_test = y_test.astype(np.float64).reshape(-1)

    X_training = df_training[col_features].values
    y_training = df_training[col_response].values
    p_training = df_training[col_protected].values
    p_training = p_training.astype(np.float64).reshape(-1)
    y_training = y_training.astype(np.float64).reshape(-1)

    X_testing = df_testing[col_features].values
    y_testing = df_testing[col_response].values
    p_testing = df_testing[col_protected].values
    p_testing = p_testing.astype(np.float64).reshape(-1)
    y_testing = y_testing.astype(np.float64).reshape(-1)

    dtypes = df_fictif[col_features].dtypes.values
    dic_cov = {col: str(df_fictif[col].dtype) for col in df_fictif.columns}

    # print(dic_cov)

    # X_train = preprocess_data(X_train, dic_cov)
    # X_test = preprocess_data(X_test, dic_cov)
    # X_training = preprocess_data(X_training, dic_cov)
    # X_testing = preprocess_data(X_testing, dic_cov)

    X_train = np.array(X_train, dtype="object")

    if VERBOSE:
        print(X_train.dtype, X_train)
        print(y_train.dtype, y_train)
        print(p_train.dtype, p_train)
        print(X_train)
        print(y_train)
        print(p_train)

    return X_train, y_train, p_train, X_test, y_test, p_test, dtypes, dic_cov


VERBOSE = False
X_train, y_train, p_train, X_test, y_test, p_test, dtypes, dic_cov = prepare_dataset(1_000_000)
print(X_train.shape[1])
# quit()

print(X_train)
print(y_train)
print(p_train)

# quit()


print("\n")
print("*************************************************************")

margin = 1.0
nb_cov = len(dtypes)
it = 1
interaction_depth = 1000
minobs = 10  # int(0.05*len(df_fictif))
bootstrap = "No"

# Timing Cython function
nb_trees = 100

# check consistance => same tree if no bootstrap
start_time = time.perf_counter()
cart0 = CARTRegressor_python(
    epsilon=margin, margin="absolute", id=0, nb_cov=nb_cov,
    replacement=False, prop_sample=1.0, frac_valid=1.0,
    max_interaction_depth=interaction_depth, minobs=minobs,
    name="DiscriTree_cython", loss="poisson", parallel="No",
    pruning="No", bootstrap=bootstrap
)

cart0.fit(X_train, y_train, p_train, dic_cov=dic_cov)
cart0.path = "Test_save/"
nb_nodes_python = len(cart0.nodes)
running_time_python = time.perf_counter() - start_time

list_nodes_index_0 = [c.index for c in cart0.nodes]
list_nodes0 = [c.index for c in cart0.nodes]

print("*************************************************************")
print("\n")
dataset = Dataset(X_train, y_train, p_train, dtypes)
start_time = time.perf_counter()
cart3 = NewCART(
    epsilon=margin, id=0, nb_cov=nb_cov, replacement=True, prop_sample=1.0,
    frac_valid=1.0, max_interaction_depth=interaction_depth, minobs=minobs,
    name="DiscriTree", loss="poisson", parallel="No", pruning="No", bootstrap="No"
)
nodes = cart3.fit(dataset)

# cart3.path = "Test_save/"
nb_nodes_cython = len(cart3.nodes)
running_time_cython = time.perf_counter() - start_time

def compare_lists(L1, L2, msg: str):
    print(msg)
    a1 = np.asarray(L1)
    a2 = np.asarray(L2)
    if a1.dtype.kind in 'OU':
        diff_indices = np.where(a1 != a2)[0]
    else:
        diff_indices = np.where(np.abs(a1-a2) > 1e-12)[0]
    if diff_indices.shape[0] == 0:
        print('\tCheck TRUE')
        return
    print('\tCheck FALSE')
    print(diff_indices)
    print(a1[diff_indices])
    print(a2[diff_indices])

print("*************************************************************")
print("\n")

# check consistancey
# check nb nodes
print(f"Nb nodes python: {nb_nodes_python}/ cython: {nb_nodes_cython}")
print(f"Check {nb_nodes_python==nb_nodes_cython}")
print("\n")

index_node_python = np.argsort([c.average_value for c in cart0.nodes])#np.array(list(sorted([c.index for c in cart0.nodes])))
index_node_cython = np.argsort([c.avg_value for c in cart3.nodes])

index_node_python = np.array(list(sorted([c.index for c in cart0.nodes])))
index_node_cython = np.array(list(sorted([c.index for c in cart3.nodes])))

kind_node_python = list(np.array([c.kind for c in cart0.nodes])[index_node_python])
kind_node_cython = list(np.array([c.kind for c in cart3.nodes])[index_node_cython])

feature_index_node_python = list(np.array([c.feature_index for c in cart0.nodes])[index_node_python])
feature_index_node_python = [-1 if x is None else x for x in feature_index_node_python]
feature_index_node_cython = list(np.array([c.feature_idx for c in cart3.nodes])[index_node_cython])

threshold_node_python = list(np.array([c.threshold for c in cart0.nodes])[index_node_python])
threshold_node_python = [-1.0 if x is None else x for x in threshold_node_python]
threshold_node_cython = list(np.array([c.threshold for c in cart3.nodes])[index_node_cython])

dloss_node_python = list(np.array([c.loss_decrease for c in cart0.nodes])[index_node_python])
dloss_node_python = [0.0 if x is None else x for x in dloss_node_python]
dloss_node_cython = list(np.array([c.dloss for c in cart3.nodes])[index_node_cython])

loss_node_python = list(np.array([c.loss for c in cart0.nodes])[index_node_python])
loss_node_python = [0.0 if x is None else x for x in loss_node_python]
loss_node_cython = list(np.array([c.loss for c in cart3.nodes])[index_node_cython])

position_node_python = list(np.array([c.position for c in cart0.nodes])[index_node_python])
position_node_cython = list(np.array([c.position for c in cart3.nodes])[index_node_cython])

average_value_node_python = list(np.array([c.average_value for c in cart0.nodes])[index_node_python])
average_value_node_cython = list(np.array([c.avg_value for c in cart3.nodes])[index_node_cython])

depth_node_python = list(np.array([c.depth for c in cart0.nodes])[index_node_python])
depth_node_cython = list(np.array([c.depth for c in cart3.nodes])[index_node_cython])

nb_samples_node_python = list(np.array([c.nb_samples for c in cart0.nodes])[index_node_python])
nb_samples_node_cython = list(np.array([c.nb_samples for c in cart3.nodes])[index_node_cython])

iterations = 1000

def find_index_difference(list1, list2):
    diff_indices = [i for i, (a, b) in enumerate(zip(list1, list2)) if abs(a - b) >= 1e-10]
    return diff_indices

compare_lists(index_node_python, index_node_cython, "Index nodes")

def not_working_currently():
    index_parent_node_python = [c.parent_node.index if c.kind != "Root" else None for c in cart0.nodes]
    index_parent_node_cython = [c.parent_node.index if c.kind != "Root" else None for c in cart3.nodes]
    print(f"Index parent node python :{list(index_parent_node_python)}")
    print(f"Index parent node cython :{list(index_parent_node_cython)}")
    print(f"Check {list(index_parent_node_python)==list(index_parent_node_cython)}")
    if list(index_parent_node_python)!=list(index_parent_node_cython):
        print(find_index_difference(list(index_parent_node_python), list(index_parent_node_cython)))
    print("\n")

    print("\n")
    print(f"Mapping node python :{mapping_node_python}")
    print(f"Mapping node cython :{mapping_node_cython}")
    print(f"Check {mapping_node_python == mapping_node_cython}")
    if list(mapping_node_python) != list(average_value_node_cython):
        print(find_index_difference(list(mapping_node_python), list(mapping_node_cython)))
    print("\n")
    print(f"Mapping node python :{nb_nodes_python}")
    print(f"Mapping node cython :{nb_nodes_cython}")
    print(f"Check {nb_nodes_python == nb_nodes_cython}")
    print("\n")
    print(f"Importance python ({imp_python_time}):{imp_python}")
    print(f"Importance cython1 ({imp_cython_time1}):{imp_cython1}")
    print(f"Check {imp_python == imp_cython1}")
    print(f"Importance cython2 ({imp_cython_time2}):{imp_cython2}")
    print(f"Check {imp_python == imp_cython2}")
    print(f"Importance cython3 ({imp_cython_time3}):{imp_cython3}")
    print(f"Check {imp_python == imp_cython3}")
    print(f"Importance cython3 ({imp_cython_time4}):{imp_cython4}")
    print(f"Check {imp_python == imp_cython4}")


compare_lists(feature_index_node_python, feature_index_node_cython, "Feature index")
boolean_index = find_index_difference(list(feature_index_node_python), list(feature_index_node_cython))

compare_lists(threshold_node_python, threshold_node_cython, "Thresholds")
compare_lists(nb_samples_node_python, nb_samples_node_cython, "Nb samples in nodes")
compare_lists(loss_node_python, loss_node_cython, "Losses")
compare_lists(dloss_node_python, dloss_node_cython, "DLosses")

if boolean_index:
    print(np.array(dloss_node_python)[boolean_index])
    print(np.array(dloss_node_cython)[boolean_index])

compare_lists(position_node_python, position_node_cython, "Positions")
compare_lists(kind_node_python, kind_node_cython, "Kind")
compare_lists(depth_node_python, depth_node_cython, "Depth")
compare_lists(average_value_node_python, average_value_node_cython, "Average values")
compare_lists(cart0.predict(X_train), cart3.predict(X_train), "Predictions on X_train")
compare_lists(cart0.predict(X_test), cart3.predict(X_test), "Predictions on X_test")
compare_lists(cart0.compute_importance2(), cart3.compute_importance2(), "Importance")

print(f"E[Y]={np.mean(y_train)}")
print(f"E[pi_0]={np.mean(cart0.predict(X_test))}")
print(f"E[pi_3]={np.mean(cart3.predict(X_test))}")


print("Comparison running time")
print(f"Running time python: {running_time_python}")
print(f"Running time cython: {running_time_cython} (x{running_time_python/running_time_cython:.2f})")

