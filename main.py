from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Form, HTTPException, Response, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Annotated

import os
import uuid
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import torch

#from ctgan.synthesizers.tvae import TVAE  non riuscivo a farlo funzionare
from sdv.datasets.demo import download_demo
from sdv.metadata import SingleTableMetadata

from utils.functions import train_ctgan, load_model, train_tvae, inference_and_report

app = FastAPI(
    title="API data gen",
    description="This is an API designed to generate syntethic data.",
    docs_url="/docs",
    redoc_url="/redoc",
    security=[{"oauth2": ["read", "write"]}],
    default_authentication=[{"oauth2": ["read"]}]
)

"""origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)"""

# Recupera la chiave API dalle variabili d'ambiente
try:
    API_KEY = os.environ['API_KEY']
except KeyError:
    API_KEY = "test"


# Endpoint to get information about the API
@app.get("/")
async def get_api_info():
    return "Hey, we're on API data-generator"

# Endpoint to get list of models
@app.get("/get_models")
async def get_models():
    models_dir = "models"
    models_list = os.listdir(models_dir)
    return JSONResponse(content={"models": models_list})

@app.post("/training_model_ctgan")
async def generate_synthetic_data(background_tasks: BackgroundTasks, epochs: Annotated[str, Form()], file_training_data: UploadFile, api_key: Annotated[str, Form()]):

    # Verifica se la chiave API fornita corrisponde alla chiave API memorizzata
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    print("starting ...")
    unique_id = str(uuid.uuid4())
    print(unique_id)

    print("reading file")
    file_path = f"data/{unique_id}_tmp.csv"
    with open(file_path, "wb") as buffer:
        buffer.write(await file_training_data.read())

    real_data = pd.read_csv(file_path)
    len_df = len(real_data)
    print(len_df)

    epochs = int(epochs)

    try:
        # Definisci automaticamente le colonne discrete basate sul tipo di dato
        discrete_columns = []
        for column in real_data.columns:
            if real_data[column].dtype == 'object':
                discrete_columns.append(column)

        print("Discrete columns:", discrete_columns)
        background_tasks.add_task(train_ctgan, real_data, discrete_columns, epochs, unique_id, file_path, len_df)

        return {"status":"200",
                "uuid": unique_id, 
                "message": f"training ctgan model in background."}
    
    except Exception as e:
        # Se si verifica un errore, cancella il file
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"status": "500",
                "error_message": str(e)}


# API endpoint per l'inferenza
@app.post("/inference_ctgan_metrics")
def inference(model_id: str, sample_num: int, api_key: str):
    # Verifica se la chiave API fornita corrisponde alla chiave API memorizzata
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Carica il modello richiesto
    model = load_model(model_id)
    
    # Esegui l'inferenza per generare dati sintetici
    synthetic_data = model.sample(sample_num)
    
    # Salva i dati sintetici in un file CSV
    new_data = "generated_data_inference"
    os.makedirs(new_data, exist_ok=True)
    synthetic_data.to_csv(os.path.join(new_data, f'synthetic_ctgan_{model_id}_data.csv'), index=False)

    return Response(synthetic_data.to_csv(index=False), media_type="text/csv")


#### TVAE

@app.post("/train_model_tvae_adults_dataset")
async def train_model(background_tasks: BackgroundTasks, epochs:int, api_key: str):

    # Verifica se la chiave API fornita corrisponde alla chiave API memorizzata
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    print("starting ...")
    unique_id = str(uuid.uuid4())
    print(unique_id)

    # Definisci la metadata
    metadata = SingleTableMetadata()
    # Scarica il dataset di esempio
    data, metadata = download_demo(modality='single_table', dataset_name='adult')

    # Estrai il numero di epoche dalla richiesta
    epochs_value = epochs

    # Aggiungi il task di background per l'addestramento del modello TVAE
    background_tasks.add_task(train_tvae, epochs_value, data, metadata, unique_id)

    # Restituisci una conferma di avvio dell'addestramento
    return {"status":"200",
                "uuid": unique_id, 
                "message": f"Training TVAE model started in background"}

@app.post("/inference_tvae")
async def inference_and_report_api(unique_id: str, num_rows: int, api_key: str):
    # Verifica se la chiave API fornita corrisponde alla chiave API memorizzata
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    num_rows=int(num_rows)

    metadata = SingleTableMetadata()
    
    # Scarica il dataset di esempio
    real_data, metadata = download_demo(modality='single_table', dataset_name='adult')

    # Esegui l'inferenza e genera il report
    synthetic_data = inference_and_report(unique_id, num_rows, metadata)
    
    # Restituisci il report e i dati sintetici
    return Response(synthetic_data.to_csv(index=False), media_type="text/csv")