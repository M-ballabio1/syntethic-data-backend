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

#from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import OneHotEncoder
#import torch

#from ctgan.synthesizers.tvae import TVAE  non riuscivo a farlo funzionare
from sdv.datasets.demo import download_demo
from sdv.metadata import SingleTableMetadata

from utils.functions import train_ctgan, load_model, train_tvae, inference_and_report, is_csv, is_excel
from database.crud import create_transaction, get_transactions
from database.database import init_db

app = FastAPI(
    title="Synthetic Data Generation API Project",
    description="This API is designed to generate synthetic tabular data for various use cases. It offers an easy-to-use interface for data scientists and developers to create synthetic datasets based on provided data.",
    version="1.0.0",
    docs_url="/docs",  # Specifica l'URL per la documentazione interattiva
    redoc_url="/redoc",  # Specifica l'URL per la documentazione Redoc
    openapi_url="/openapi.json",  # URL per il file OpenAPI JSON
    contact={
        "name": "Matteo Ballabio",
        "url": "https://m-ballabio1.github.io/digital-cv.github.io/",
        "email": "matteoballabio99@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    security=[{"api-key": ["read", "write"]}],  # Definisce il tipo di sicurezza supportato
    default_authentication=[{"api-key": ["read"]}]  # Configura le autorizzazioni predefinite
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

# Inizializza il database all'avvio dell'applicazione
init_db()

# Recupera la chiave API dalle variabili d'ambiente
try:
    API_KEY = os.environ['API_KEY']
except KeyError:
    API_KEY = "test"


# Endpoint to get information about the API
@app.get("/")
async def get_api_info():
    """
    Endpoint per ottenere informazioni sull'API.

    Returns:
        str: Un messaggio di benvenuto.
    """
    return "Hey, we're on API Syntethic Data Generator"

@app.get("/transactions")
async def get_transactions_endpoint():
    """
    Endpoint per ottenere lista delle transazioni.

    Returns:
        str: Un messaggio di benvenuto.
    """
    transactions = get_transactions()
    return transactions

# Endpoint to get list of models
@app.get("/get_models")
async def get_models():
    """
    Endpoint per ottenere l'elenco dei modelli disponibili.

    Returns:
        JSONResponse: Un elenco JSON dei modelli disponibili.
    """
    models_dir = "models"
    models_list = os.listdir(models_dir)
    return JSONResponse(content={"models": models_list})

@app.post("/training_model_ctgan", tags=["Training ctgan"])
async def generate_synthetic_data(background_tasks: BackgroundTasks, epochs: Annotated[str, Form()], file_training_data: UploadFile, api_key: Annotated[str, Form()]):
    """
    Endpoint per avviare il processo di generazione di dati sintetici utilizzando il modello CTGAN.

    Args:
        - background_tasks (BackgroundTasks): Oggetto per l'esecuzione di task in background.
        - epochs (str): Il numero di epoche per l'addestramento del modello CTGAN.
        - file_training_data (UploadFile): Il file contenente i dati di addestramento.
        - api_key (str): La chiave API per l'autenticazione.

    Returns:
        str: Una conferma del successo dell'avvio del processo di generazione.
    """
    # Verifica se la chiave API fornita corrisponde alla chiave API memorizzata
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    print("starting ...")
    unique_id = str(uuid.uuid4())
    print(unique_id)

    print("check type format")
    print(file_training_data.filename)

    if is_csv(file_training_data.filename):
        print("reading file csv")
        file_path = f"data/{unique_id}_tmp.csv"
        with open(file_path, "wb") as buffer:
            buffer.write(await file_training_data.read())
        real_data = pd.read_csv(file_path)
    
    elif is_excel(file_training_data.filename):
        print("reading file xlsx")
        file_path = f"data/{unique_id}_tmp.xlsx"
        with open(file_path, "wb") as buffer:
            buffer.write(await file_training_data.read())
        real_data = pd.read_excel(file_path)
    
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

        # Success API
        create_transaction(model_id=unique_id, method="POST", url="/training_model_ctgan", status_code=200)

        return {"status":"200",
                "uuid": unique_id, 
                "message": f"training ctgan model in background."}
    
    except Exception as e:
        # Se si verifica un errore, cancella il file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Error API
        create_transaction(model_id=unique_id, method="POST", url="/training_model_ctgan", status_code=500)
        return {"status": "500", "error_message": str(e)}


# API endpoint per l'inferenza
@app.post("/inference_ctgan_tvae_metrics", tags=["Inference CPU/GPU"])
async def inference(model_id: str, sample_num: int, api_key: str):
    """
    Funzione per inferenza utilizzando il modello CTGAN e TVAE e generazione di dati sintetici.

    Args:
        - model_id (str): L'ID del modello utilizzato per l'inferenza deve essere un modello .PT
        - sample_num (int): Il numero di campioni da generare.
        - api_key (str): La chiave API per l'autenticazione.

    Returns:
        str: Una conferma del successo del processo di generazione.
    """
    try:
        # Verifica se la chiave API fornita corrisponde alla chiave API memorizzata
        if api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

        # Carica il modello richiesto
        model, model_used = load_model(model_id)
        
        # Esegui l'inferenza per generare dati sintetici
        synthetic_data = model.sample(sample_num)
        
        # Salva i dati sintetici in un file CSV
        new_data = "generated_data_inference"
        os.makedirs(new_data, exist_ok=True)
        synthetic_data.to_csv(os.path.join(new_data, f'synthetic_{model_used}_{model_id}_data.csv'), index=False)

        # Success API
        create_transaction(model_id=model_id, method="POST", url="/inference_ctgan_tvae_metrics", status_code=200)
        return Response(synthetic_data.to_csv(index=False), media_type="text/csv")
    
    except Exception as e:
        # In caso di errore, registra la transazione con il codice di stato dell'errore
        create_transaction(model_id=model_id, method="POST", url="/inference_ctgan_metrics", status_code=500)
        # Solleva un'eccezione HTTP con codice di stato 500
        raise HTTPException(status_code=500, detail=str(e))


#### TVAE

@app.post("/train_model_tvae_adults_dataset", tags=["Training tvae"])
async def train_model(background_tasks: BackgroundTasks, epochs:int, api_key: str):
    """
    Funzione per l'addestramento del modello TVAE utilizzando il dataset adults.csv.

    Args:
        - background_tasks (BackgroundTasks): Oggetto per l'esecuzione di task in background.
        - epochs (int): Il numero di epoche per l'addestramento.
        - api_key (str): La chiave API per l'autenticazione.

    Returns:
        Dict[str, str]: Un dizionario contenente lo stato dell'addestramento.
    """
    try:
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

        # Success API
        create_transaction(model_id=unique_id, method="POST", url="/train_model_tvae_adults_dataset", status_code=200)
        # Restituisci una conferma di avvio dell'addestramento
        return {"status":"200", "uuid": unique_id, "message": f"Training TVAE model started in background"}
    
    except Exception as e:
        # In caso di errore, registra la transazione con il codice di stato dell'errore
        create_transaction(model_id=unique_id, method="POST", url="/train_model_tvae_adults_dataset", status_code=500)
        # Solleva un'eccezione HTTP con codice di stato 500
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference_tvae_gpu", tags=["Inference GPU"])
async def inference_and_report_api(unique_id: str, num_rows: int, api_key: str):
    """
    Funzione per l'inferenza utilizzando il modello TVAE funzionante solo con una GPU e la presenza di cuda.

    Args:
        - unique_id (str): L'ID univoco del modello TVAE in formato .PKL
        - num_rows (int): Il numero di righe di dati sintetici da generare.
        - api_key (str): La chiave API per l'autenticazione.

    Returns:
        Response: Il report e i dati sintetici in formato CSV.
    """
    try:
        # Verifica se la chiave API fornita corrisponde alla chiave API memorizzata
        if api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

        num_rows=int(num_rows)

        metadata = SingleTableMetadata()
        
        # Scarica il dataset di esempio
        real_data, metadata = download_demo(modality='single_table', dataset_name='adult')

        # Esegui l'inferenza e genera il report
        synthetic_data = inference_and_report(unique_id, num_rows, metadata)
        
        # Success API
        create_transaction(model_id=unique_id, method="POST", url="/inference_tvae", status_code=200)
        # Restituisci il report e i dati sintetici
        return Response(synthetic_data.to_csv(index=False), media_type="text/csv")
    
    except Exception as e:
        # In caso di errore, registra la transazione con il codice di stato dell'errore
        create_transaction(model_id=unique_id, method="POST", url="/inference_tvae", status_code=500)
        # Solleva un'eccezione HTTP con codice di stato 500
        raise HTTPException(status_code=500, detail=str(e))