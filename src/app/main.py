from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from minio import Minio
import pandas as pd
import io
from io import BytesIO
import os.path
from collections import defaultdict
from functools import lru_cache
from random import randint, choice
import os
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from skl2onnx.common.data_types import FloatTensorType
from coniferest.onnx import to_onnx as to_onnx_add
from coniferest.aadforest import AADForest
from sklearn.model_selection import train_test_split
import itertools
import zipfile
from fink_science.ad_features.processor import FEATURES_COLS
import uvicorn
import requests
import json
from concurrent.futures import ProcessPoolExecutor
from pydantic import BaseModel

executor = ProcessPoolExecutor(max_workers=2)

class ModelData(BaseModel):
    model_name: str
    positive: List[str]
    negative: List[str]

app = FastAPI()
Base = declarative_base()
DATABASE_URL = "sqlite:////data/models.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

MINIO_URL = os.getenv('MINIO_URL')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')
BUCKET_DATASETS_NAME = os.getenv('BUCKET_DATASETS_NAME')

minio_client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)


if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)
if not minio_client.bucket_exists(BUCKET_DATASETS_NAME):
    minio_client.make_bucket(BUCKET_DATASETS_NAME)

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    last_update = Column(DateTime, nullable=True)
    last_download = Column(DateTime, nullable=True)
    last_request_update = Column(DateTime, nullable=False, default=datetime.utcnow())
    num_reactions = Column(Integer, nullable=False, default=0)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def generate_param_comb(param_dict):
    base = itertools.product(*param_dict.values())
    columns = param_dict.keys()
    for obj in base:
        yield dict(zip(columns, obj))


def train_base_AAD(data: pd.DataFrame, train_params, scorer, y_true, use_default_model=False):
    if use_default_model:
        return AADForest().fit(data.values)
    X_train, X_test, y_train, y_test = train_test_split(
        data.values, y_true, test_size=0.2, random_state=42)
    best_est = (0, None, None)
    for cur_params in generate_param_comb(train_params):
        print(cur_params)
        forest = AADForest(**cur_params)
        forest.fit(X_train, y_train)
        cur_score = scorer(forest, X_test, y_test)
        print(cur_score)
        if cur_score > best_est[0]:
            best_est = (cur_score, forest, cur_params)
    print(f'Optimal: {best_est[2]}')
    return AADForest(**best_est[2]).fit(data.values, y_true)


def extract_one(data, key) -> pd.Series:
    """
    Function for extracting data from lc_features
    :param data: dict
                lc_features dict
    :param key: str
                Name of the extracted filter
    :return: pd.DataFrame
                Dataframe with a specific filter
    """
    series = pd.Series(data[key], dtype=float)
    return series


def train_with_forest(data, train_params, scorer_, y_true) -> IsolationForest:
    """
    Training of the IsolationForest model
    :param data: pd.DataFrame
        Training dataset
    :param train_params: dict
        Model hyperparameters
    :param scorer_: function
        Model quality evaluation function
    :param y_true:
        Target
    :return: IsolationForest
        Trained model
    """
    forest = IsolationForest()
    clf = GridSearchCV(forest, train_params, scoring=scorer_, verbose=2, cv=4)
    clf.fit(data.values, y_true)
    print(f' Optimal params: {clf.best_params_}')
    return clf.best_estimator_

def scorer_AAD(estimator, X_test, y_test):
    y_score = estimator.score_samples(X_test)
    return roc_auc_score(y_test, y_score)

def scorer(estimator, x_test, y_test):
    """
    Evaluation function
    :param estimator: sklearn.model
    :param x_test: pd.DataFrame
        Dataset with predictors
    :param y_test: pd.Series
        Target values
    :return: double
        roc_auc_score
    """
    y_score = estimator.decision_function(x_test)
    cur_score = roc_auc_score(y_test, y_score)
    return cur_score


def unknown_pref_metric(y_true, y_pred):
    """
    Recall calculation
    :param y_true: pd.series
        True target values
    :param y_pred: pd.series
        Predicted values target
    :return: double
        recall score
    """
    correct_preds_r = sum(y_true & y_pred)
    trues = sum(y_true)
    return (correct_preds_r / trues) * 100


unknown_pref_scorer = make_scorer(unknown_pref_metric, greater_is_better=True)


def get_stat_param_func(data):
    """
    Function for extracting attributes from dataframe
    :param data: pd.DataFrame
    :return: function
        Returns a function that allows extraction from the feature column of the dataframe data param attribute
    """
    @lru_cache
    def get_stat_param(feature, param):
        return getattr(data[feature], param)()
    return get_stat_param


def generate_random_rows(data, count):
    """
    :param data: pd.DataFrame
    :param count: int
    :return: dict
    """
    get_param = get_stat_param_func(data)
    rows = []
    for _ in range(count):
        row = {}
        for feature in data.columns:
            feature_mean = get_param(feature, 'mean')
            feature_std = get_param(feature, 'std')
            has_negative = get_param(feature, 'min') < 0
            mults = [-1, 1] if has_negative else [1]
            value = feature_mean + feature_std * (randint(1000, 2000) / 1000) * choice(mults)
            row[feature] = value
        rows.append(row)
    return rows


def append_rows(data, rows):
    """

    :param data: pd.DataFrame
    :param rows: dict
    :return: pd.DataFrame
    """
    return data.append(rows, ignore_index=True)


def unknown_and_custom_loss(model, x_data, true_is_anomaly):
    """

    :param model: sklearn.model
    :param x_data: pd.DataFrame
    :param true_is_anomaly: pd.DataFrame
    :return:
    """
    scores = model.score_samples(x_data)
    scores_order = scores.argsort()
    len_for_check = 3000
    found = 0

    for i in scores_order[:len_for_check]:
        if true_is_anomaly.iloc[i]:
            found += 1

    return (found / len_for_check) * 100


def extract_all(data) -> pd.Series:
    """
    Function for extracting data from lc_features
    :param data: dict
                lc_features dict
    :param key: str
                Name of the extracted filter
    :return: pd.DataFrame
                Dataframe with a specific filter
    """
    series = pd.Series(data, dtype=float)
    return series


def get_reactions(positive: List[str], negative: List[str]):
    print('Получаем текущие реакции')
    good_reactions = set(positive)
    bad_reactions = set(negative)
    oids = list(good_reactions.union(bad_reactions))
    # r = await fetch_data(oids)
    r = requests.post(
        'https://api.fink-portal.org/api/v1/objects',
        json={
            'objectId': ','.join(oids),
            'columns': 'd:lc_features_g,d:lc_features_r,i:objectId',
            'output-format': 'json'
        }
    )
    if r.status_code != 200:
        print(r.text)
        return
    else:
        print('Fink API: OK')
    pdf = pd.read_json(io.BytesIO(r.content))
    for col in ['d:lc_features_g', 'd:lc_features_r']:
        pdf[col] = pdf[col].apply(lambda x: json.loads(x))
    feature_names = FEATURES_COLS
    pdf = pdf.loc[(pdf['d:lc_features_g'].astype(str) != '[]') & (pdf['d:lc_features_r'].astype(str) != '[]')]
    feature_columns = ['d:lc_features_g', 'd:lc_features_r']
    common_rems = [
        'percent_amplitude',
        'linear_fit_reduced_chi2',
        'inter_percentile_range_10',
        'mean_variance',
        'linear_trend',
        'standard_deviation',
        'weighted_mean',
        'mean'
    ]
    result = dict()
    for section in feature_columns:
        pdf[feature_names] = pdf[section].to_list()
        pdf_gf = pdf.drop(feature_columns, axis=1).rename(columns={'i:objectId': 'object_id'})
        classes = np.where(pdf_gf['object_id'].isin(good_reactions), True, False)
        
        pdf_gf = pdf_gf.reindex(sorted(pdf_gf.columns), axis=1)
        pdf_gf.drop(common_rems, axis=1, inplace=True)
        pdf_gf['class'] = classes
        pdf_gf.dropna(inplace=True)
        pdf_gf.drop_duplicates(subset=['object_id'], inplace=True)
        pdf_gf.drop(['object_id'], axis=1, inplace=True)
        result[f'_{section[-1]}'] = pdf_gf.copy()
    return result

def retrain_task(name: str, positive: List[str], negative: List[str]):
    reactions_datasets = get_reactions(positive, negative)
    filter_base = ('_r', '_g')
    response = minio_client.get_object(BUCKET_DATASETS_NAME, "base_dataset.parquet")
    dataset = BytesIO(response.read())
    x_buf_data = pd.read_parquet(dataset)
    print('Датасет получен, стартуем предобработку')
    if "lc_features_r" not in x_buf_data.columns:
        features_1 = x_buf_data["lc_features"].apply(lambda data:
            extract_one(data, "1")).add_suffix("_r")
        features_2 = x_buf_data["lc_features"].apply(lambda data:
            extract_one(data, "2")).add_suffix("_g")
    else:
        features_1 = x_buf_data["lc_features_r"].apply(lambda data:
            extract_all(data)).add_suffix("_r")
        features_2 = x_buf_data["lc_features_g"].apply(lambda data:
            extract_all(data)).add_suffix("_g")
        x_buf_data = x_buf_data.rename(columns={'finkclass':'class'}, errors='ignore')

    data = pd.concat([
    x_buf_data[['objectId', 'candid', 'class']],
        features_1,
        features_2,
    ], axis=1).dropna(axis=0)
    datasets = defaultdict(lambda: defaultdict(list))
    with tqdm(total=len(data)) as pbar:
        for _, row in data.iterrows():
            for passband in filter_base:
                new_data = datasets[passband]
                new_data['object_id'].append(row.objectId)
                new_data['class'].append(row['class'])
                for col, r_data in zip(data.columns, row):
                    if not col.endswith(passband):
                        continue
                    new_data[col[:-2]].append(r_data)
            pbar.update()

    main_data = {}
    for passband in datasets:
        new_data = datasets[passband]
        new_df = pd.DataFrame(data=new_data)
        for col in new_df.columns:
            if col in ('object_id', 'class'):
                new_df[col] = new_df[col].astype(str)
                continue
            new_df[col] = new_df[col].astype('float64')
        main_data[passband] = new_df
    data = {key : main_data[key] for key in filter_base}
    assert data['_r'].shape[1] == data['_g'].shape[1], '''Mismatch of the dimensions of r/g!'''
    classes = {filter_ : data[filter_]['class'] for filter_ in filter_base}
    common_rems = [
        'percent_amplitude',
        'linear_fit_reduced_chi2',
        'inter_percentile_range_10',
        'mean_variance',
        'linear_trend',
        'standard_deviation',
        'weighted_mean',
        'mean'
    ]
    data = {key : item.drop(labels=['object_id', 'class'] + common_rems,
                axis=1) for key, item in data.items()}
    for key, item in data.items():
        item.mean().to_csv(f'{key}_means.csv')
    print('Training...')
    result_models_IO = []
    for key in filter_base:
        is_unknown = classes[key] == 'Unknown'
        initial_type = [('X', FloatTensorType([None, data[key].shape[1]]))]
        search_params_aad = {
            "n_trees": (100, 150, 200, 300, 500, 700, 1024),
            "n_subsamples": (int(obj*data[key].shape[0]) for obj in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
            "tau": (1 - sum(is_unknown) / len(data[key]), ),
            "n_jobs": (12,)
        }
        forest_simp = train_base_AAD(
            data[key],
            search_params_aad,
            scorer_AAD,
            is_unknown,
            use_default_model=True
        )
        reactions_dataset = reactions_datasets[key]
        reactions = reactions_dataset['class'].values
        reactions_dataset.drop(['class'], inplace=True, axis=1)
        forest_simp.fit(np.array(reactions_dataset), reactions)
        onx = to_onnx_add(forest_simp, initial_types=initial_type)
        result_models_IO.append(onx.SerializeToString())

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(f'forest{filter_base[0]}_AAD_{name}.onnx', result_models_IO[0])
        zip_file.writestr(f'forest{filter_base[1]}_AAD_{name}.onnx', result_models_IO[1])
    zip_buffer.seek(0)
    try:
        minio_client.put_object(
            BUCKET_NAME, 
            f"anomaly_detection_forest_AAD_{name}.zip", 
            zip_buffer, 
            length=zip_buffer.getbuffer().nbytes,
            content_type='application/zip'
        )
        print(f"Model {name} stored in bucket {BUCKET_NAME}")
        db = SessionLocal()
        model = db.query(Model).filter(Model.name == name).first()
        model.last_update = datetime.utcnow()
    except Exception as e:
        print(f"Failed to upload model {name} to MinIO: {e}")
    db.commit()
    db.close()

@app.get("/list_models", response_model=List[str])
def list_models():
    db = SessionLocal()
    return [model.name for model in db.query(Model).all()]

@app.post("/retrain_model")
def retrain_model(
    data: ModelData, background_tasks: BackgroundTasks
):
    
    db = SessionLocal()
    model = db.query(Model).filter(Model.name == data.model_name).first()
    if not model:
        model = Model(name=data.model_name, last_update=datetime.utcnow(), last_download=None)
        db.add(model)
        db.commit()
    else:
        prev_time = model.last_request_update
        cur_time = datetime.utcnow()
        if (cur_time-prev_time).total_seconds()/60 < 10:
            return {"status": "You recently requested a model update. Try again later."}
        model.num_reactions = len(data.negative) + len(data.positive)
        db.commit()
        db.close()
    print(dict(data))
    background_tasks.add_task(executor.submit, retrain_task, data.model_name, data.positive, data.negative)
    return {"status": "Model retraining started."}


@app.get("/get_model_signal")
def get_model_signal(model_name: str):
    db = SessionLocal()
    model = db.query(Model).filter(Model.name == model_name).first()
    model.last_download = datetime.utcnow()
    db.commit()
    db.close()
    return {'status': 'Ok!'}

@app.get("/get_last_update_model")
def get_last_update_model(model_name: str):
    db = SessionLocal()
    model = db.query(Model).filter(Model.name == model_name).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    last_update = model.last_update.isoformat() if model.last_update else None
    return {"last_update_time": last_update,
            "num_reactions": model.num_reactions}

@app.get("/get_last_download_model")
def get_last_download_model(model_name: str):
    db = SessionLocal()
    model = db.query(Model).filter(Model.name == model_name).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    last_download = model.last_download.isoformat() if model.last_download else None
    return {"last_download_time": last_download,
            "num_reactions": model.num_reactions}

@app.get("/get_count_reactions")
def get_count_reactions(model_name: str):
    db = SessionLocal()
    model = db.query(Model).filter(Model.name == model_name).first()
    if not model:
        return {'count_of_reactions': 0}
    return {'count_of_reactions': model.num_reactions}


if __name__ == '__main__':
    keypath = "certs/privkey.pem"
    certpath = "certs/fullchainl.pem"
    if os.path.isfile(keypath) and os.path.isfile(certpath):
        uvicorn.run("main:app", host="0.0.0.0", port=443, reload=True,
            ssl_keyfile=keypath,
            ssl_certfile=certpath)
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8082, reload=False)
