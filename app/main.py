from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
app = FastAPI(title='Linear Regression Analyzer')
app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')
templates = Jinja2Templates(directory=BASE / 'templates')
DATASETS = {}  # Classroom demo only: uploaded data kept in process memory.
MAX_SIZE = 8 * 1024 * 1024


def clean(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return str(value)


def preview(df, n=10):
    return {'columns': list(df.columns), 'rows': [[clean(x) for x in row] for row in df.head(n).itertuples(index=False, name=None)]}


@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')


@app.post('/api/upload')
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, 'ატვირთეთ CSV ფაილი.')
    raw = await file.read(MAX_SIZE + 1)
    if len(raw) > MAX_SIZE:
        raise HTTPException(400, 'ფაილი არ უნდა აღემატებოდეს 8 MB-ს.')
    try:
        df = pd.read_csv(BytesIO(raw), encoding='utf-8-sig')
    except (UnicodeDecodeError, pd.errors.ParserError):
        try:
            df = pd.read_csv(BytesIO(raw), encoding='cp1252')
        except Exception:
            raise HTTPException(400, 'CSV ფაილის წაკითხვა ვერ მოხერხდა.')
    except Exception:
        raise HTTPException(400, 'CSV ფაილის წაკითხვა ვერ მოხერხდა.')
    if df.empty or len(df.columns) < 2 or not df.columns.is_unique:
        raise HTTPException(400, 'ფაილში საჭიროა მონაცემები და მინიმუმ ორი უნიკალური სვეტი.')
    if len(DATASETS) >= 50:
        DATASETS.pop(next(iter(DATASETS)))
    dataset_id = uuid4().hex
    DATASETS[dataset_id] = df
    return {'dataset_id': dataset_id, 'filename': file.filename, 'count': len(df), 'preview': preview(df), 'types': {c: str(df[c].dtype) for c in df.columns}, 'suggested_categorical': [c for c in df.columns if df[c].dtype == 'object' or isinstance(df[c].dtype, pd.CategoricalDtype)]}


@app.post('/api/analyze')
def analyze(dataset_id: str = Form(...), categorical: str = Form(''), target: str = Form(...), features: str = Form(...), test_size: float = Form(0.2)):
    import json
    if dataset_id not in DATASETS:
        raise HTTPException(404, 'ფაილი ვერ მოიძებნა. ატვირთეთ ხელახლა.')
    df = DATASETS[dataset_id].copy()
    try:
        cats = json.loads(categorical)
        selected = json.loads(features)
    except (ValueError, TypeError):
        raise HTTPException(400, 'სვეტების არასწორი არჩევანი.')
    if not isinstance(cats, list) or not isinstance(selected, list) or not all(isinstance(x, str) for x in cats + selected):
        raise HTTPException(400, 'სვეტების არასწორი არჩევანი.')
    if not 0.1 <= test_size <= 0.4:
        raise HTTPException(400, 'სატესტო წილი უნდა იყოს 10%-დან 40%-მდე.')
    if target not in df or not selected or target in selected or any(c not in df for c in selected + cats) or any(c not in selected for c in cats) or len(set(selected)) != len(selected):
        raise HTTPException(400, 'შეარჩიეთ მიზნობრივი და დამოუკიდებელი სვეტები სწორად.')
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise HTTPException(400, 'დამოკიდებული ცვლადი უნდა იყოს რიცხვითი.')
    if df[target].isna().any():
        raise HTTPException(400, 'დამოკიდებულ ცვლადში არის გამოტოვებული მნიშვნელობები. ჯერ გაწმინდეთ CSV.')
    X = df[selected].copy()
    for c in selected:
        if c not in cats and not pd.api.types.is_numeric_dtype(X[c]):
            raise HTTPException(400, f'სვეტი „{c}“ ტექსტურია. მონიშნეთ კატეგორიულად.')
    if X.isna().any().any():
        raise HTTPException(400, 'დამოუკიდებელ ცვლადებში არის გამოტოვებული მნიშვნელობები. ჯერ გაწმინდეთ CSV.')
    X = pd.get_dummies(X, columns=cats, drop_first=True, dtype=int)
    if X.shape[1] == 0:
        raise HTTPException(400, 'გარდაქმნის შემდეგ არცერთი დამოუკიდებელი ცვლადი არ დარჩა.')
    y = df[target]
    if len(df) < 8:
        raise HTTPException(400, 'ანალიზისთვის საჭიროა მინიმუმ 8 სტრიქონი.')
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        metrics = {'r2': clean(r2_score(y_test, predictions)), 'mae': clean(mean_absolute_error(y_test, predictions)), 'mse': clean(mse), 'rmse': clean(np.sqrt(mse))}
    except (ValueError, TypeError) as e:
        raise HTTPException(400, f'მოდელის გაშვება ვერ მოხერხდა: {str(e)}')
    results = [{'actual': clean(a), 'predicted': clean(p)} for a, p in zip(y_test.iloc[:30], predictions[:30])]
    coefficients = [{'feature': col, 'coefficient': clean(coef)} for col, coef in zip(X.columns, model.coef_)]
    return {'metrics': metrics, 'train_count': len(X_train), 'test_count': len(X_test), 'intercept': clean(model.intercept_), 'coefficients': coefficients, 'predictions': results, 'encoded_preview': preview(X), 'target': target}
