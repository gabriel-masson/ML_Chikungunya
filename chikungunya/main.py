# %%
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from Utils import Utils
import mlflow
utils = Utils()
print("Hello, Chikungunya!")

# %%
df = pd.read_excel('./data/V2_BANCO_DOTLAB_347.xlsx')
df['D0_disestesia_episodio1'].fillna(0, inplace=True)
df.head()
# %%
# Features segundo GUIDELINES FOR THE CLINICAL DIAGNOSIS AND TREATMENT OF DENGUE, CHIKUNGUNYA, AND ZIKA By PAHO

features_paho = ['D0_idade', 'D0_genero', 'D0_cor',

                 'D0_febre_episodio1',         'D0_cefaleia_episodio1',
                 'D0_palidez_episodio1',     'D0_hiperemia_episodio1',          'D0_turvacao_episodio1',
                 'D0_nausea_episodio1',      'D0_vomito_episodio1',             'D0_diarreia_episodio1',
                 'D0_parestesia_episodio1',  'D0_disestesia_episodio1',         'D0_artralgia_episodio1',
                 'D0_edema_episodio1',       'D0_mialgia_episodio1',            'D0_lombalgia_episodio1',
                 'D0_prurido_episodio1',     'D0_rash_episodio1',               'D0_lesoes_episodio1',
                 'D0_alopecia_episodio1',    'D0_alteracaomemoria_episodio1',   'D0_alteracaolibido_episodio1',
                 'D0_enpp_fadiga',

                 'D0_Novo critério (0-3/4-10)',
                 ]

features_pain = ['D0_idade', 'D0_genero', 'D0_cor',

                 'D0_febre_episodio1',         'D0_cefaleia_episodio1',
                 'D0_palidez_episodio1',     'D0_hiperemia_episodio1',          'D0_turvacao_episodio1',
                 'D0_nausea_episodio1',      'D0_vomito_episodio1',             'D0_diarreia_episodio1',
                 'D0_parestesia_episodio1',  'D0_disestesia_episodio1',         'D0_artralgia_episodio1',
                 'D0_edema_episodio1',       'D0_mialgia_episodio1',            'D0_lombalgia_episodio1',
                 'D0_prurido_episodio1',     'D0_rash_episodio1',               'D0_lesoes_episodio1',
                 'D0_alopecia_episodio1',    'D0_alteracaomemoria_episodio1',   'D0_alteracaolibido_episodio1',
                 'D0_enpp_fadiga',

                 'D0_intensdor_ombroe', 'D0_intensdor_ombrod', 'D0_intensdor_punhoe', 'D0_intensdor_punhod', 'D0_intensdor_dedomaoe', 'D0_intensdor_dedomaod',
                 'D0_intensdor_joelhoe', 'D0_intensdor_joelhod', 'D0_intensdor_tornozeloe', 'D0_intensdor_tornozelod', 'D0_intensdor_dedopee', 'D0_intensdor_dedoped',

                 'D0_Novo critério (0-3/4-10)',
                 ]

df_paho = df[features_paho]
df_pain = df[features_pain]

df_paho.columns = [col.replace('D0_', '').replace(
    '_episodio1', '') for col in df_paho.columns]
df_pain.columns = [col.replace('D0_', '').replace(
    '_episodio1', '') for col in df_pain.columns]


# %%
rename_columns = {
    'idade': 'age',
    'genero': 'gender',
    'cor': 'race',
    'febre': 'fever',
    'cefaleia': 'headache',
    'palidez': 'pallor',
    'hiperemia': 'hyperemia',
    'turvacao': 'blurred_vision',
    'nausea': 'nausea',
    'vomito': 'vomiting',
    'diarreia': 'diarrhea',
    'parestesia': 'paresthesia',
    'disestesia': 'dysesthesia',
    'artralgia': 'arthralgia',
    'edema': 'edema',
    'mialgia': 'myalgia',
    'lombalgia': 'low_back_pain',
    'prurido': 'pruritus',
    'rash': 'rash',
    'lesoes': 'lesions',
    'alopecia': 'alopecia',
    'alteracaomemoria': 'memory_impairment',
    'alteracaolibido': 'libido_change',
    'enpp_fadiga': 'fatigue',
    'intensdor_ombroe': 'pain_intensity_left_shoulder',
    'intensdor_ombrod': 'pain_intensity_right_shoulder',
    'intensdor_punhoe': 'pain_intensity_left_wrist',
    'intensdor_punhod': 'pain_intensity_right_wrist',
    'intensdor_dedomaoe': 'pain_intensity_left_hand_fingers'
}
df_pain = df_pain.rename(columns=rename_columns)
df_paho = df_paho.rename(columns=rename_columns)


# %%

X_train_paho, X_test_paho, y_train_paho, y_test_paho = utils.train_test_balance(
    df_paho)

X_train_pain, X_test_pain, y_train_pain, y_test_pain = utils.train_test_balance(
    df_pain)


# %%
print(X_train_paho.shape, y_train_paho.shape)
print(X_train_pain.shape, y_train_pain.shape)


# %%
# Modelos com atributos da PAHO
dt_paho = DecisionTreeClassifier(criterion="entropy", max_depth=79,
                                 max_features=102, min_samples_leaf=9, min_samples_split=7, random_state=42)

rf_paho = RandomForestClassifier(criterion="gini", n_estimators=974, max_depth=1000,
                                 min_samples_leaf=0.01, min_samples_split=0.01, random_state=42, n_jobs=-1)

svm_paho = SVC(kernel="linear", degree=2, gamma=4.2594025,
               C=0.010007202, probability=True, random_state=42)

xgb_paho = xgb.XGBClassifier(colsample_bytree=0.239900212, learning_rate=0.068163796, max_depth=780,
                             n_estimators=699, subsample=0.067780561, objective="binary:logistic", eval_metric="logloss", random_state=42)

ada_paho = AdaBoostClassifier(
    n_estimators=502, learning_rate=0.2607703, random_state=42)

gb_paho = GradientBoostingClassifier(n_estimators=288, learning_rate=0.018212977748381,
                                     max_depth=999, min_samples_split=0.01, min_samples_leaf=0.40386, random_state=42)

# modelos com dor
dt_pain = DecisionTreeClassifier(criterion="gini", max_depth=947, max_features=102,
                                 min_samples_leaf=2, min_samples_split=2, random_state=42)

rf_pain = RandomForestClassifier(criterion="gini", n_estimators=10, max_depth=1000,
                                 min_samples_leaf=0.01, min_samples_split=0.01, random_state=42, n_jobs=-1)

svm_pain = SVC(kernel="rbf", degree=2, gamma=1.263465825,
               C=292237.4272, probability=True, random_state=42)

xgb_pain = xgb.XGBClassifier(colsample_bytree=0.4733170818, learning_rate=0.9440797635, max_depth=128,
                             n_estimators=581, subsample=0.8493754108, objective="binary:logistic", eval_metric="logloss", random_state=42)

ada_pain = AdaBoostClassifier(
    n_estimators=665, learning_rate=1.808715607, random_state=42)

gb_pain = GradientBoostingClassifier(n_estimators=688, learning_rate=0.0509833746,
                                     max_depth=1000, min_samples_split=0.01, min_samples_leaf=0.01, random_state=42)

# %%
models_paho = [dt_paho, rf_paho, svm_paho, xgb_paho, ada_paho, gb_paho]
models_pain = [dt_pain, rf_pain, svm_pain, xgb_pain, ada_pain, gb_pain]

# %%
print("PAHO MODELS")
paho_models = []
for model in models_paho:
    model_name = model.__class__.__name__
    model.fit(X_train_paho, y_train_paho)
    paho_models.append(model)

print("PAIN MODELS")
pain_models = []
for model in models_pain:
    model_name = model.__class__.__name__
    model.fit(X_train_pain, y_train_pain)
    pain_models.append(model)

# %%
# Experimento 1 - Avaliação dos modelos com os atributos da PAHO
print("PAHO MODELS EVALUATION")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id=1)

# %%
for model in paho_models:
    model_name = model.__class__.__name__

    with mlflow.start_run(run_name=model_name):
        mlflow.sklearn.autolog()
        mlflow.xgboost.autolog()
        metrics = utils.avaliation_model(model, X_test_paho, y_test_paho)

        mlflow.log_metrics(metrics)
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        input_example = X_test_paho.iloc[:5]
        y_pred = model.predict(X_test_paho)
        signature = mlflow.models.infer_signature(X_test_paho, y_pred)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
        )

# %%
