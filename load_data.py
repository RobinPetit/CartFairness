import pandas as pd
import numpy as np

# Example
# Example
def load_dataset_charpentier(nb_obs, verbose: bool=True):
    #df_fictif = pd.read_feather("data.feather")

    df_fictif = pd.read_feather("freqMTPLT2freq.feather")

    #print(df_fictif)
    #print(df_fictif.dtypes)

    def inspect_col():
        list_covariates = []
        for col in df_fictif.columns:
            print(f"col: {col}")
            print(df_fictif[col].value_counts())
            list_covariates.append(col)

        print(list_covariates)

    #inspect_col()
    #quit()

    list_covariates = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'Area', 'Density', 'Region']
    #list_covariates = ['VehPower', 'VehAge', 'DrivAge', 'Density']

    df_fictif['gender'] = df_fictif['VehGas'].map({'Regular': 0, 'Diesel': 1})
    p = df_fictif['gender'].values
    df_fictif['p'] = p

    df_fictif['y'] = df_fictif['ClaimNb']

    col_features = list_covariates
    col_features.append("gender")
    col_response = ['y']
    col_protected = ['p']

    df_fictif = df_fictif[np.concatenate((col_features, col_response, col_protected))]

    df_fictif = df_fictif.iloc[:nb_obs, :]

    # if nb_obs<1000000:
    # df_fictif = df_fictif.iloc[:nb_obs, :]

    # df_fictif = df_fictif.sample(frac=1, random_state=2023).reset_index(drop=True)

    if verbose:
        print(df_fictif)
        print(df_fictif.dtypes)
        print(df_fictif.isna().sum())
        print(df_fictif.describe())
    return df_fictif, col_features, col_response, col_protected


def load_dataset_wutricht(nb_obs, verbose: bool=True):
    df_fictif = pd.read_feather("data.feather")

    #df_fictif = pd.read_feather("freqMTPLT2freq.feather")

    print(df_fictif)
    print(df_fictif.dtypes)

    def inspect_col():
        list_covariates = []
        for col in df_fictif.columns:
            print(f"col: {col}")
            print(df_fictif[col].value_counts())
            list_covariates.append(col)

        print(list_covariates)

    #inspect_col()
    #quit()

    list_covariates = ['age', 'ac', 'power', 'brand', 'area', 'dens', 'ct']

    p = df_fictif['gas'].values
    df_fictif['p'] = p
    df_fictif['gender'] = df_fictif['gas'].map({'Regular': 0, 'Diesel': 1})
    df_fictif['y'] = df_fictif['claims']

    col_features = list_covariates
    col_features.append("gender")
    col_response = ['y']
    col_protected = ['p']

    df_fictif = df_fictif[np.concatenate((col_features, col_response, col_protected))]

    df_fictif = df_fictif.iloc[:nb_obs, :]

    # if nb_obs<1000000:
    # df_fictif = df_fictif.iloc[:nb_obs, :]

    # df_fictif = df_fictif.sample(frac=1, random_state=2023).reset_index(drop=True)

    if verbose:
        print(df_fictif)
        print(df_fictif.dtypes)
        print(df_fictif.isna().sum())
        print(df_fictif.describe())
    return df_fictif, col_features, col_response, col_protected

def load_dataset(nb_obs, verbose: bool=True):
    df_fictif = pd.read_feather("master_gender_agg2.feather")
    #df_fictif = df_fictif.iloc[:nb_obs, :]
    df_fictif['veh_power'] = df_fictif['veh_power'].apply(lambda x: np.nan if x == 0 else x)
    df_fictif['veh_value'] = df_fictif['veh_value'].apply(lambda x: np.nan if x == 0 else x)
    df_fictif['driv_m_age'] = df_fictif['driv_m_age'].apply(lambda x: int(x))
    df_fictif['veh_age'] = df_fictif['veh_age'].apply(lambda x: int(x))
    df_fictif['cont_seniority'] = df_fictif['cont_seniority'].apply(lambda x: int(x))
    df_fictif['gender'] = df_fictif['gender'].map({'M':0, 'F':1})
    df_fictif['veh_make'] = df_fictif['veh_make'].astype('category')
    df_fictif['geo_mosaic_code_lib'] = df_fictif['geo_mosaic_code_lib'].astype('category')

    if verbose:
        print(df_fictif)
        for col in df_fictif.columns:
            print(col)

    np.random.seed(2023)
    X = df_fictif[['driv_m_age', 'veh_age']].values
    #y = df_fictif['Y'].values
    #y = np.random.randint(0, 2, 1000)
    p = df_fictif['gender'].values
    df_fictif['p'] = p
    df_fictif['y'] = df_fictif['claim_at_fault_flg'].fillna(0)
    y = df_fictif['y']

    list_covariates = [#"geo_mosaic_code_lib",  # "veh_sgmt_informex_lib",
                       # "geo_postcode_lng", "geo_postcode_lat",
                       "veh_use",
                       "veh_adas", "veh_make",  # "veh_type",
                       # "veh_trailer",
                       "veh_garage", "veh_fuel",  # "veh_first_owner_flg",
                       "veh_mileage_limit", "veh_power", "veh_weight",  # "veh_engine_disp",
                       "veh_value", "veh_age",  # "veh_seats",
                       "driv_number", "driv_m_age",  # "driv_m_experience", #'driv_y_age','driv_y_experience',
                       "cont_seniority", "cont_paysplit"]  # , "cont_sepa"]

    #print(X.shape)
    #print(np.linspace(0, 100, 100).reshape(-1,1).shape)
    #quit()

    def cap_values(db):
        db['driv_m_age'] = db['driv_m_age'].clip(lower=17, upper=100)
        db['veh_age'] = db['veh_age'].clip(lower=0, upper=25)
        db['veh_value'] = db['veh_value'].clip(lower=1000, upper=50000)
        db['veh_power'] = db['veh_power'].clip(lower=25, upper=700)
        db['veh_weight'] = db['veh_weight'].clip(lower=600, upper=3500)
        db['cont_seniority'] = db['cont_seniority'].clip(lower=0, upper=83)
        return db


    df_fictif = cap_values(df_fictif)

    if verbose:
        print(f"Before rounding: {print(df_fictif[['veh_value', 'veh_power', 'veh_weight']].nunique())}")

    def reduce_nb_split(df_fictif):
        df_fictif['veh_power'] = df_fictif['veh_power'].apply(lambda x: np.round(x / 10) * 10)
        df_fictif['veh_weight'] = df_fictif['veh_weight'].apply(lambda x: np.round(x / 100) * 100)
        df_fictif['veh_value'] = df_fictif['veh_value'].apply(lambda x: np.round(x / 1000) * 1000)
        return df_fictif

    # df_fictif = reduce_nb_split(df_fictif)

    if verbose:
        print(f"After rounding: {print(df_fictif[['veh_value', 'veh_power', 'veh_weight']].nunique())}")

    feature_index = 1
    threshold = 70
    margin=1

    col_features = list_covariates#['driv_m_age']#, 'veh_age']#, 'Veh_power']#['veh_make']#
    col_features.append("gender")
    col_response = ['y']
    col_protected = ['p']

    df_fictif = df_fictif[np.concatenate((col_features, col_response, col_protected))]

    df_fictif = df_fictif.iloc[:nb_obs, :]

    #if nb_obs<1000000:
        #df_fictif = df_fictif.iloc[:nb_obs, :]

    #df_fictif = df_fictif.sample(frac=1, random_state=2023).reset_index(drop=True)

    if verbose:
        print(df_fictif)
        print(df_fictif.dtypes)
        print(df_fictif.isna().sum())
        print(df_fictif.describe())
    return df_fictif, col_features, col_response, col_protected
