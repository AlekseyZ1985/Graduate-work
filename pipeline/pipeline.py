import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, recall_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
import numpy as np
import catboost as cb
from datetime import datetime
import joblib
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from pickle import dump
import dill

def event(df):
    import pandas as pd
    df_copy = df.copy()
    df_copy.loc[df_copy['event_action'].isin(
        ["sub_car_claim_click", "sub_open_dialog_click", "sub_custom_question_submit_click", "sub_call_number_click",
         "sub_callback_submit_click", "sub_submit_success", "sub_car_request_submit_click",
         "sub_car_claim_submit_click"]) == False, 'event_action'] = 0
    df_copy.loc[df_copy['event_action'].isin(
        ["sub_car_claim_click", "sub_open_dialog_click", "sub_custom_question_submit_click", "sub_call_number_click",
         "sub_callback_submit_click", "sub_submit_success", "sub_car_request_submit_click",
         "sub_car_claim_submit_click"]) == True, 'event_action'] = 1
    df_copy["event_action"] = df_copy.event_action.astype(int)
    df1_new = df_copy.groupby('session_id').agg({'event_action': 'max'}).reset_index()
    df1_new = df1_new.drop_duplicates()
    df2 = pd.read_csv('ga_sessions.csv')
    df_copy = df1_new.merge(df2, how="left")
    df_copy = df_copy[~(df_copy["geo_city"].isna())]
    df_copy = df_copy[~(df_copy["utm_source"].isna())]
    return df_copy

def filter(df):
    import pandas as pd
    df_copy = df.copy()
    columns_to_drop = [
        'device_model',
        'utm_keyword',
        'session_id',
        'client_id',
        'visit_number',
        "visit_time",
        'device_os',
    ]
    df_copy = df_copy.drop(columns_to_drop, axis=1)
    return df_copy


def fillna(df):
    import pandas as pd
    df_copy = df.copy()
    df_copy.device_brand = df_copy.device_brand.fillna('other')
    df_copy.utm_campaign = df_copy.utm_campaign.fillna('other')
    df_copy.utm_adcontent = df_copy.utm_adcontent.fillna('other')
    drop_utm = df_copy.utm_campaign.value_counts()[df_copy.utm_campaign.value_counts() < 10000].index
    df_copy.utm_campaign[df_copy.utm_campaign.isin(drop_utm)] = "rare"
    return df_copy


def blowout(df):
    import pandas as pd
    df_copy = df.copy()
    df_copy["visit_date"] = pd.to_datetime(df_copy.visit_date, utc=True)
    df_copy['month'] = df_copy.visit_date.apply(lambda x: x.month)
    df_copy = df_copy.drop('visit_date', axis=1)
    return df_copy


def utm_medium(df):
    import pandas as pd
    df_copy = df.copy()
    df_copy.loc[df_copy['utm_medium'].isin(["(none)", "referral", "organic"]) == False, 'utm_medium'] = "cost"
    df_copy.loc[df_copy['utm_medium'].isin(["(none)", "referral", "organic"]) == True, 'utm_medium'] = "organical"
    return df_copy


def frequency(df):
    import pandas as pd
    df_copy = df.copy()
    fe1 = df_copy.groupby("utm_source").size() / len(df_copy)
    df_copy["utm_source_freq"] = df_copy["utm_source"].map(fe1)
    fe2 = df_copy.groupby("utm_medium").size() / len(df_copy)
    df_copy["utm_medium_freq"] = df_copy["utm_medium"].map(fe2)
    fe3 = df_copy.groupby("device_category").size() / len(df_copy)
    df_copy["device_category_freq"] = df_copy["device_category"].map(fe3)
    fe4 = df_copy.groupby("device_brand").size() / len(df_copy)
    df_copy["device_brand_freq"] = df_copy["device_brand"].map(fe4)
    fe5 = df_copy.groupby("device_browser").size() / len(df_copy)
    df_copy["device_browser_freq"] = df_copy["device_browser"].map(fe5)
    fe6 = df_copy.groupby("utm_campaign").size() / len(df_copy)
    df_copy["utm_campaign_freq"] = df_copy["utm_campaign"].map(fe6)
    fe7 = df_copy.groupby("geo_city").size() / len(df_copy)
    df_copy["geo_city_freq"] = df_copy["geo_city"].map(fe7)
    fe8 = df_copy.groupby("geo_country").size() / len(df_copy)
    df_copy["geo_country_freq"] = df_copy["geo_country"].map(fe8)
    fe9 = df_copy.groupby("utm_adcontent").size() / len(df_copy)
    df_copy["utm_adcontent_freq"] = df_copy["utm_adcontent"].map(fe9)
    fe10 = df_copy.groupby("device_screen_resolution").size() / len(df_copy)
    df_copy["device_screen_resolution_freq"] = df_copy["device_screen_resolution"].map(fe10)
    return df_copy

def main():
    import pandas as pd
    df = pd.read_csv('ga_hits.csv')
    df = event(df)
    X = df.drop(columns=["event_action"])
    Y = df['event_action']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter)),
        ('fillna', FunctionTransformer(fillna)),
        ('blowout', FunctionTransformer(blowout)),
        ('utm_medium', FunctionTransformer(utm_medium)),
        ('frequency', FunctionTransformer(frequency)),
        ('column_transformer', column_transformer)
    ])

    model = cb.CatBoostClassifier(
        eval_metric="AUC",
        depth=10,
        iterations=500,
        l2_leaf_reg=1,
        learning_rate=0.15,
        scale_pos_weight=(Y == 1).sum() / (Y == 0).sum(),
    )

    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    score = cross_val_score(pipe, X, Y, cv=3, scoring="roc_auc")
    print(f'model: {type(model).__name__}, roc_auc: {score.mean():.4f}, roc_std: {score.std():.4f}')

    pipe.fit(X, Y)
    with open('sber_pipe.pkl', 'wb') as file:
        dill.dump(
            {
                "model": pipe,
                "metadata": {
                    "name": "sber prediction pipeline",
                    "author": "Zatsepin Aleksey",
                    "version": 1,
                    "date": datetime.now()
                },
            },
            file,
        )


if __name__ == '__main__':
    main()