# compare.py
import timeit

from version_simple_cython21_adj_categorical_clean_epsilon import CARTRegressor_cython, preprocess_data
from version_simple_python_cat_clean_epsilon import CARTRegressor_python
import time
import matplotlib.pyplot as plt
from load_data import load_dataset_charpentier
import numpy as np

from CART import CART as NewCART
from dataset import Dataset
from load_data import load_dataset
import pandas as pd
import time
from math import log

list_time_cython_obs = []
list_time_cython2_obs = []
list_time_cython3_obs = []
list_time_sklearn_obs = []
range_nb_obs = range(25000, 100001, 250000)


def prepare_dataset(nb_observation):
    #nb_observation = 100_000
    kind_dataset = "PV" #"charpentier" #

    if kind_dataset == "charpentier":
        df_fictif, col_features, col_response, col_protected = load_dataset_charpentier(nb_obs=nb_observation, verbose=VERBOSE)
        col_features = ['VehPower', 'VehAge', 'DrivAge', 'Density', 'BonusMalus']
    else:
        df_fictif, col_features, col_response, col_protected = load_dataset(nb_obs=nb_observation,
                                                                                    verbose=VERBOSE)
        #col_features = ["veh_power", "veh_weight", "veh_value", "veh_age", "driv_number", "driv_m_age", "cont_seniority", "veh_make"]
        print(col_features)
        #quit()
        col_features = ['veh_use', 'veh_adas', 'veh_garage', 'veh_fuel', 'veh_mileage_limit', 'veh_power', 'veh_make',
                       'veh_weight', 'veh_value', 'veh_age', 'driv_number', 'driv_m_age', 'cont_seniority', 'cont_paysplit']#, 'gender'] #

    df_fictif.dropna(inplace=True)

    print(df_fictif.isna().sum())


    print(col_features)

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

    #print(dic_cov)

    print(dtypes)

    #quit()


    #X_train = preprocess_data(X_train, dic_cov)
    #X_test = preprocess_data(X_test, dic_cov)
    #X_training = preprocess_data(X_training, dic_cov)
    #X_testing = preprocess_data(X_testing, dic_cov)

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
X_train, y_train, p_train, X_test, y_test, p_test, dtypes, dic_cov = prepare_dataset(10_000)#000)
print(X_train.shape[1])
#quit()

print(pd.Series(y_train).value_counts())
#quit()

print(X_train)
print(y_train)
print(p_train)

#quit()




print("\n")
print("*************************************************************")

margin = 1.0 #0.001
nb_cov = len(dtypes)
it = 1
interaction_depth = 1000
minobs = 10 #int(0.05*len(df_fictif))
bootstrap = "No" #"Yes"#"No"
replacement = False # True #
kind_margin = "relative" #
max_depth = 10
all_modalities = False # True #
split = "depth" # "best" #

# Timing Cython function
nb_trees = 100

print(f"Prop women training: {p_train.mean()} / Men: {1-p_train.mean()}")
print(f"Prop women testing: {p_test.mean()} / Men: {1-p_test.mean()}")
if kind_margin == "relative":
    print(f"Tolerate margin: {(1-p_test.mean())*margin*100}%")
else:
    print(f"Tolerate margin: {margin * 100}%")
#quit()

print(pd.Series(X_train[:, 0]).value_counts())

#quit()

print("\n")

# check consistance => same tree if no bootstrap
start_time = time.perf_counter()
cart0 = CARTRegressor_python(epsilon=margin, margin=kind_margin, id=0, nb_cov=nb_cov,
                replacement=replacement, prop_sample=1.0, frac_valid=1.0,
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

#cart0.prune(X_test, y_test)
#cart0.prune_all_combi(X_test, y_test)
#quit()


print("*************************************************************")
print("\n")
dataset = Dataset(X_train, y_train, p_train, dtypes, np.ones(p_train.shape))
start_time = time.perf_counter()
cart3 = NewCART(epsilon=margin, margin=kind_margin, id=0, nb_cov=nb_cov,
                replacement=replacement, prop_sample=1.0, frac_valid=1.0,
                max_interaction_depth=interaction_depth, minobs=minobs,
                name="DiscriTree", loss="poisson", parallel="No",
                pruning="No", bootstrap=bootstrap,
                split=split, exact_categorical_splits=all_modalities)

cart3.fit(dataset)

# for node in cart3.nodes:
#     print(node.loss)
#
#quit()

#cart3.path = "Test_save/"
nb_nodes_cython = len(cart3.nodes)
running_time_cython = time.perf_counter() - start_time
#cart3.display_tree(cart3.tree_depth)

# print([c.feature_idx for c in cart3.nodes])
# print([c.feature_index for c in cart0.nodes])

#quit()


def benchmark_loss_function():
    eps = 10**-18
    size_test = len(df_test)

    pred_IS = np.array(cart0.predict(df_fictif[col_features].values))
    pred_OOS = np.array(cart0.predict(df_test[col_features].values))[:size_test]
    y_IS = df_fictif[col_response].values.reshape(-1)
    y_OOS = df_test[col_response].values.reshape(-1)[:size_test]

    print(pd.Series(y_IS).value_counts())
    print(pd.Series(y_OOS).value_counts())
    #quit()

    print(df_fictif[col_response].values.reshape(-1).shape)
    print(pred_IS.shape)
    print(df_test[col_response].values.reshape(-1).shape)
    print(pred_OOS.shape)

    print(f"y_OOS:{y_OOS}")
    print(f"pred_OOS:{pred_OOS}")

    dev_IS = _poisson_deviance_simp3(y_IS, pred_IS)
    dev_OOS = _poisson_deviance_simp3(y_OOS, pred_OOS)

    dev_IS2 = 2/len(df_fictif)*np.sum(y_IS*np.log((y_IS+eps)/(pred_IS+eps))-y_IS+pred_IS)
    dev_OOS2 = 2/size_test*np.sum(y_OOS*np.log((y_OOS+eps)/(pred_OOS+eps))-y_OOS+pred_OOS)

    int_IS3 = y_IS*np.log((y_IS+eps)/(np.array(pred_IS)+eps))-y_IS+np.array(pred_IS)
    int_OOS3 = y_OOS*np.log((y_OOS+eps)/(np.array(pred_OOS)+eps))-y_OOS+np.array(pred_OOS)

    dev_IS3 = 2/len(df_fictif)*np.sum(int_IS3)
    dev_OOS3 = 2/size_test*np.sum(int_OOS3)

    dev_IS4 = 0
    dev_OOS4 = 0
    dev_IS5 = 0
    dev_OOS5 = 0

    print("\n")
    for i in range(pred_IS.shape[0]):
        dev_IS4 +=  y_IS[i]*np.log((y_IS[i]+eps)/(pred_IS[i]+eps)) - y_IS[i] + pred_IS[i]
        dev_IS5 += y_IS[i] * log((y_IS[i] + eps) / (pred_IS[i] + eps)) - y_IS[i] + pred_IS[i]
    for i in range(pred_OOS.shape[0]):
        dev_OOS4 += y_OOS[i] * np.log((y_OOS[i] + eps) / (pred_OOS[i] + eps)) - y_OOS[i] + pred_OOS[i]
        dev_OOS5 += y_OOS[i] * log((y_OOS[i] + eps) / (pred_OOS[i] + eps)) - y_OOS[i] + pred_OOS[i]
        if _poisson_deviance_simp3(np.array(y_OOS[i]).reshape(-1), np.array(pred_OOS[i]).reshape(-1)) != 2*(y_OOS[i] * log((y_OOS[i] + eps) / (pred_OOS[i] + eps)) - y_OOS[i] + pred_OOS[i]):
            print(f"Index {i}, y = {y_OOS[i]}, y_pred = {pred_OOS[i]}, dev_contrib ={2*(y_OOS[i] * log((y_OOS[i] + eps) / (pred_OOS[i] + eps)) - y_OOS[i] + pred_OOS[i])}")
            print(f"Index {i}, y = {y_OOS[i]}, y_pred = {pred_OOS[i]}, dev_contrib ={_poisson_deviance_simp3(np.array(y_OOS[i]).reshape(-1), np.array(pred_OOS[i]).reshape(-1))}")
            print("\n")

    #quit()
    dev_IS4 = (2/y_IS.shape[0])*dev_IS4
    dev_OOS4 = (2/y_OOS.shape[0])*dev_OOS4
    dev_IS5 = (2/y_IS.shape[0])*dev_IS5
    dev_OOS5 = (2/y_OOS.shape[0])*dev_OOS5
    #quit()
    #print(int_IS3)
    #print(int_OOS3)

    iterations = 100000



    print(f"dev_OOS or: {_poisson_deviance_or(y_OOS, pred_OOS)}, dev_simp: {_poisson_deviance_simp(y_OOS, pred_OOS)}, dev_simp_2: {_poisson_deviance_simp2(y_OOS, pred_OOS)}, dev_simp_3: {_poisson_deviance_simp3(y_OOS, pred_OOS)}")

    start_time = time.perf_counter()
    for _ in range(iterations):
        _poisson_deviance_or(y_OOS, pred_OOS)
    time_dev_or = time.perf_counter() - start_time
    print(f"Time dev or: {time_dev_or}")

    start_time = time.perf_counter()
    for _ in range(iterations):
        _poisson_deviance_simp(y_OOS, pred_OOS)
    time_dev_simp = time.perf_counter() - start_time
    print(f"Time dev simp: {time_dev_simp}")

    start_time = time.perf_counter()
    for _ in range(iterations):
        _poisson_deviance_simp2(y_OOS, pred_OOS)
    time_dev_simp2 = time.perf_counter() - start_time
    print(f"Time dev simp 2: {time_dev_simp2}")

    start_time = time.perf_counter()
    for _ in range(iterations):
        _poisson_deviance_simp3(y_OOS, pred_OOS)
    time_dev_simp3 = time.perf_counter() - start_time
    print(f"Time dev simp 3: {time_dev_simp3}")


    print(f"dev_IS: {dev_IS}, {dev_IS2}, np={dev_IS3}, {dev_IS4}, {dev_IS5}")
    print(f"dev_OOS: {dev_OOS}, {dev_OOS2}, np={dev_OOS3}, {dev_OOS4}, {dev_OOS5}")
    #quit()

    print(list(y_OOS[100:150]))
    print(list(pred_OOS[100:150]))
    print(list(y_OOS[100:150]*np.log((y_OOS[100:150]+eps)/(pred_OOS[100:150]+eps))))
    print(_poisson_deviance_simp3(y_OOS[:10], pred_OOS[:10]))
    print(_poisson_deviance_simp3(y_OOS[:50], pred_OOS[:50]))
    print(_poisson_deviance_simp3(y_OOS[:100], pred_OOS[:100]))
    print(_poisson_deviance_simp3(y_OOS, pred_OOS))

#quit()

print("*************************************************************")
print("\n")

# check consistancey
# check nb nodes
print(f"Nb nodes python: {nb_nodes_python}/ cython: {nb_nodes_cython}")
print(f"Check {nb_nodes_python==nb_nodes_cython}")
print("\n")

# check split
# sort by index node
#index_node_python = np.array(list(sorted([c.index for c in cart0.nodes])))
#index_node_cython = np.array(list(sorted([c.index for c in cart3.nodes])))



#print(index_node_python)
#print(index_node_cython)


# print([c for c in cart0.nodes])
# print([c for c in cart3.nodes])

# print([c.index for c in cart0.nodes])
# print([c.index for c in cart3.nodes])

#quit()


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

#mapping_node_python = list(np.array([c.mapping_categorical for c in cart0.nodes])[index_node_python])
#mapping_node_cython = list(np.array([c.mapping_categorical for c in cart3.nodes])[index_node_python])

nb_samples_node_python = list(np.array([c.nb_samples for c in cart0.nodes])[index_node_python])
nb_samples_node_cython = list(np.array([c.nb_samples for c in cart3.nodes])[index_node_cython])


iterations = 1000

def find_index_difference(list1, list2):
    diff_indices = [i for i, (a, b) in enumerate(zip(list1, list2)) if a != b]
    #print(diff_indices)
    return diff_indices

# print(f"Index node python :{list([c.index for c in cart0.nodes])}")
# print(f"Index node python :{list(index_node_python)}")
# print(f"Index node cython :{list(index_node_cython)}")
# print(f"Check {list(index_node_python)==list(index_node_cython)}")
# if list(index_node_python)!=list(index_node_cython):
#     print(find_index_difference(list(index_node_python), list(index_node_cython)))
# print("\n")

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


print("\n")
print(f"Index feature node python :{feature_index_node_python}")
print(f"Index feature node cython :{feature_index_node_cython}")
print(f"Check {feature_index_node_python==feature_index_node_cython}")
if list(feature_index_node_python)!=list(feature_index_node_cython):
    print(find_index_difference(list(feature_index_node_python), list(feature_index_node_cython)))

boolean_index = find_index_difference(list(feature_index_node_python), list(feature_index_node_cython))

# print(f"Index categorical node cython :{[c.is_categorical for c in cart3.nodes]}")

print("\n")
print(f"Threshold node python :{threshold_node_python}")
print(f"Threshold node cython :{threshold_node_cython}")
print(f"Check {threshold_node_python==threshold_node_cython}")
if list(threshold_node_python)!=list(threshold_node_cython):
    print(find_index_difference(list(threshold_node_python), list(threshold_node_cython)))
print("\n")

print("\n")
print(f"Nb of samples node python :{nb_samples_node_python}")
print(f"Nb of samples node cython :{nb_samples_node_cython}")
print(f"Check {nb_samples_node_python==nb_samples_node_cython}")
if list(nb_samples_node_python)!=list(nb_samples_node_cython):
    print(find_index_difference(list(nb_samples_node_python), list(nb_samples_node_cython)))
print("\n")

print(f"Loss node python :{loss_node_python}")
print(f"Loss node cython :{loss_node_cython}")
print(f"Check {loss_node_python == loss_node_cython}")
if list(loss_node_python) != list(loss_node_cython):
    print(find_index_difference(list(loss_node_python), list(loss_node_cython)))
print("\n")

print(f"Dloss node python :{dloss_node_python}")
print(f"Dloss node cython :{dloss_node_cython}")
print(f"Check {dloss_node_python == dloss_node_cython}")
if list(dloss_node_python) != list(dloss_node_cython):
    print(find_index_difference(list(dloss_node_python), list(dloss_node_cython)))

print(f"Index feature of difference: {np.array(feature_index_node_python)[boolean_index]} vs {np.array(feature_index_node_cython)[boolean_index]}")
print(f"Dloss des noeuds différents python: {np.array(dloss_node_python)[boolean_index]}")
print(f"Dloss des noeuds différents cython: {np.array(dloss_node_cython)[boolean_index]}")

print("\n")
print(f"Position node python :{position_node_python}")
print(f"Position node cython :{position_node_cython}")
print(f"Check {position_node_python == position_node_cython}")
if list(position_node_python) != list(position_node_cython):
    print(find_index_difference(list(position_node_python), list(position_node_cython)))
print("\n")

print(f"Kind node python :{kind_node_python}")
print(f"Kind node cython :{kind_node_cython}")
print(f"Check {kind_node_python==kind_node_cython}")
if list(kind_node_python)!=list(kind_node_cython):
    print(find_index_difference(list(kind_node_python), list(kind_node_cython)))

print("\n")
print(f"Depth node python :{depth_node_python}")
print(f"Depth node cython :{depth_node_cython}")
print(f"Check {depth_node_python==depth_node_cython}")
if list(depth_node_python)!=list(depth_node_cython):
    print(find_index_difference(list(depth_node_python), list(depth_node_cython)))

print(f"Average value node python :{average_value_node_python}")
print(f"Average value cython :{average_value_node_cython}")
print(f"Check {average_value_node_python == average_value_node_cython}")
if list(average_value_node_python) != list(average_value_node_cython):
    print(find_index_difference(list(average_value_node_python), list(average_value_node_cython)))

print("\n")
print(f"Predictions python :{list(cart0.predict(X_train[:100, :]))}, {list(cart0.predict(X_train[:100, :]))}")
print(pd.Series(cart0.predict(X_train)).value_counts())
print(f"Predictions cython :{list(cart3.predict(X_train[:100, :]))}, {list(cart3.predict(X_train[:100, :]))}")
print(pd.Series(cart3.predict(X_train)).value_counts())
print(f"Check {cart0.predict(X_train)==list(cart3.predict(X_train))}")
if list(cart0.predict(X_train))!=list(cart3.predict(X_train)):
    print(find_index_difference(list(cart0.predict(X_train)), list(cart3.predict(X_train))))


print("\n")
print(f"Predictions python :{list(cart0.predict(X_test[:100, :]))}, {list(cart0.predict(X_test[:100, :]))}")
print(pd.Series(cart0.predict(X_test)).value_counts())
print(f"Predictions cython :{list(cart3.predict(X_test[:100, :]))}, {list(cart3.predict(X_test[:100, :]))}")
print(pd.Series(cart3.predict(X_test)).value_counts())
print(f"Check {cart0.predict(X_test)==list(cart3.predict(X_test))}")
if list(cart0.predict(X_test))!=list(cart3.predict(X_test)):
    print(find_index_difference(list(cart0.predict(X_test)), list(cart3.predict(X_test))))


print(f"E[Y]={np.mean(y_train)}")
print(f"E[pi_0]={np.mean(cart0.predict(X_test))}")
print(f"E[pi_3]={np.mean(cart3.predict(X_test))}")


print("\n")
#if position_node_python == position_node_cython and kind_node_python==kind_node_cython and
    #list(depth_node_python)!=list(depth_node_cython) and list(average_value_node_python) != list(average_value_node_cython)
    #and cart0.predict(X_train)==list(cart3.predict(X_train) and cart0.predict(X_test)==list(cart3.predict(X_test))

print("Comparison running time")
print(f"Running time python: {running_time_python}")
print(f"Running time cython: {running_time_cython} (x{running_time_python/running_time_cython:.2f})")

