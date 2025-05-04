from load_data import load_dataset
from loss import poisson_deviance
from dataset import Dataset
from forest import RandomForestRegressor
from CART import CART

import numpy as np

nb_observation = 100_000
df_fictif, col_features, col_response, col_protected = load_dataset(
    nb_obs=nb_observation, verbose=False
)
df_fictif.dropna(inplace=True)
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
dic_cov = {col: str(df_fictif[col].dtype) for col in col_features}

dataset = Dataset(X_train, y_train, p_train, dtypes)

rf = RandomForestRegressor(100, n_jobs=2, interaction_depth=25, max_depth=10)
rf.fit(dataset)
y_pred_rf = rf.predict(X_test)


dt = CART(max_interaction_depth=25, max_depth=10)
dt.fit(dataset)
y_pred_dt = dt.predict(X_test)

print(f'Poisson Deviance of RF: {poisson_deviance(y_test, y_pred_rf)}')
print(f'Poisson Deviance of DT: {poisson_deviance(y_test, y_pred_dt)}')
