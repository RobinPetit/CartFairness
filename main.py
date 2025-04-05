# compare.py
from version_simple_cython21_adj_categorical_clean_epsilon import CARTRegressor_cython, preprocess_data
from version_simple_python_cat_clean_epsilon import CARTRegressor_python
import time
import matplotlib.pyplot as plt
from load_data import load_dataset
import numpy as np

from CART import Dataset, CART as NewCART


##################################################################################
VERBOSE = False

# load dataset and make sure that all numerical variable are in float64
nb_observation = 100_000
df_fictif, col_features, col_response, col_protected = load_dataset(nb_obs=nb_observation, verbose=VERBOSE)
df_fictif.dropna(inplace=True)
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

X_train = preprocess_data(X_train, dic_cov)
X_test = preprocess_data(X_test, dic_cov)
X_training = preprocess_data(X_training, dic_cov)
X_testing = preprocess_data(X_testing, dic_cov)

if VERBOSE:
    print(X_train.dtype, X_train)
    print(y_train.dtype, y_train)
    print(p_train.dtype, p_train)
    print(X_train)
    print(y_train)
    print(p_train)
##################################################################################
# Tree parameters
margin = 1.0
nb_cov = len(col_features)
it = 1
depth = 3
minobs = 10
range_nb_obs = [1000*k for k in range(1, 6)]
# bootstrap = "Yes"  # "No" # For model replication (dataset is not boostraped so we must end up with same trees) => for benchmarking computation time is better to let it True
bootstrap = 'No'
##################################################################################


# Timing Cython function
models = [NewCART, CARTRegressor_cython, CARTRegressor_python]
model_names = ['Cython (new)', 'Cython (original)', 'Python']

timers = [list() for _ in range(len(models))]

dataset = Dataset(X_train, y_train, p_train, dtypes)

def mse(y, y_pred):
    return np.mean((y-y_pred)**2)

for i, (model, model_name) in enumerate(zip(models, model_names)):
    for _ in range_nb_obs:
        print(f'Fitting {model_name} model')
        start_time = time.time()
        cart = model(
            epsilon=margin, margin="absolute", id=_, nb_cov=nb_cov,
            replacement=True, prop_sample=0.5, frac_valid=1.0,
            max_interaction_depth=depth, minobs=minobs,
            name="DiscriTree_cython", loss="poisson", parallel="No",
            pruning="No", bootstrap=bootstrap
        )
        if model is NewCART:
            cart.fit(dataset)
        else:
            cart.fit(X_train, y_train, p_train, dic_cov=dic_cov)
        timers[i].append(time.time() - start_time)
        print('MSE:', mse(y_test, cart.predict(X_test)))

means = np.mean(timers, axis=1)

timers = [
    [timers[2][j] / timers[i][j] for j in range(len(timers[i]))]
    for i in range(len(timers))
]

for i in range(len(timers)-1):
    print(f'Average speedup of {model_names[i]}: {means[-1] / means[i]:1.2f}')
exit()

for i in range(len(models)):
    plt.plot(range_nb_obs, timers[i], label=model_names[i] + f' Avg speedup {means[2] / means[i]:1.2f}')

plt.xlabel('Training set size')
plt.ylabel('Speedup factor')
plt.suptitle(f"Implementation comparison ({nb_observation:,} observations)")
plt.grid(True)
plt.legend()
# plt.show()
plt.savefig('comparison.png', bbox_inches='tight')
