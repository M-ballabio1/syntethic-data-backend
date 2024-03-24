from fastapi import HTTPException

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import io

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

from ctgan import CTGAN
from sdv.single_table import TVAESynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot

def is_csv(file_name):
    return file_name.lower().endswith('.csv')

def is_excel(file_name):
    return file_name.lower().endswith('.xlsx') or file_name.lower().endswith('.xls')

# Background task for training the model
def train_ctgan(data: pd.DataFrame, discrete_columns: list, epochs: int, unique_id: str, file_path: str, len_df: str):
    # Initialize and train the CTGAN model
    ctgan = CTGAN(3)

    try:
        print("using ctgan")
        ctgan.epochs = epochs
        print("sto per fare training")
        ctgan.fit(data, discrete_columns)
        models_folder = "models"
        os.makedirs(models_folder, exist_ok=True)
        print("sto per salvare modello")

        # save sia per come .pt che .pkl file
        torch.save(ctgan, os.path.join(models_folder, f'ctgan_model_{unique_id}_{str(epochs)}_{str(len_df)}.pt'))
        with open(os.path.join(models_folder, f'ctgan_model_{unique_id}_{str(epochs)}_{str(len_df)}.pkl'), 'wb') as f:
            pickle.dump(ctgan, f)

    except Exception as e:
        # Se si verifica un errore, cancella il file
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"status": "500", "error_message": str(e)}


# Caricamento del modello pickle
def load_model(model_id):
    try:
        # Controlla se il modello con l'ID univoco fornito esiste nella cartella models
        models_folder = "models"
        model_found = False
        model_path = None
        for model_file in os.listdir(models_folder):
            if model_file.startswith(f"ctgan_model_{model_id}") and model_file.endswith(f".pt"):
                model_found = True
                model_path = os.path.join(models_folder, model_file)

                # Caricamento del modello assegnandolo alla CPU
                model = torch.load(model_path, map_location=torch.device('cpu'))
                model_used = "ctgan"
                break
            elif model_file.startswith(f"tvae_model_{model_id}") and model_file.endswith(f".pt"):
                model_found = True
                model_path = os.path.join(models_folder, model_file)

                # Caricamento del modello assegnandolo alla CPU
                model = torch.load(model_path, map_location=torch.device('cpu'))
                model_used = "tvae"
                break
        
        if not model_found:
            raise FileNotFoundError(f"Model with ID {model_id} not found")
        
        return model, model_used
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")


# Funzione per addestrare il modello TVAE
def train_tvae(epochs: int, data, metadata, unique_id):
    # Crea un'istanza del TVAESynthesizer
    synthesizer = TVAESynthesizer(metadata=metadata, epochs=epochs)
    # Addestra il modello
    synthesizer.fit(data)
    len_df = len(data)
    # Ottieni i valori di loss
    loss_values = synthesizer.get_loss_values()
    # Salva il modello
    models_folder = "models"
    os.makedirs(models_folder, exist_ok=True)

    # save sia per come .pt che .pkl file
    torch.save(synthesizer, os.path.join(models_folder, f'tvae_model_{unique_id}_{str(epochs)}_{str(len_df)}.pt'))
    synthesizer.save(filepath=os.path.join(models_folder, f'tvae_model_{unique_id}_{str(epochs)}_{str(len_df)}.pkl'))
    print("finish training tvae")
    return loss_values

# Funzione per caricare il modello e generare dati sintetici
def inference_and_report(unique_id, num_rows, metadata):
# Controlla se il modello con l'ID univoco fornito esiste nella cartella models
    models_folder = "models"
    model_found = False
    model_path = None
    for model_file in os.listdir(models_folder):
        if model_file.startswith(f"tvae_model_{unique_id}" or model_file.startswith(f"ctgan_model_{unique_id}")):
            model_found = True
            model_path = os.path.join(models_folder, model_file)
            break
    
    # Se il modello non è stato trovato, restituisci un errore 404
    if not model_found:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # remove gpu
    synthesizer = TVAESynthesizer(metadata)
    print(synthesizer.get_parameters())
    synthesizer = TVAESynthesizer.load(filepath=model_path)
    
    # Genera dati sintetici
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    
    gen_data= "generated_data_inference"
    os.makedirs(gen_data, exist_ok=True)
    synthetic_data.to_csv(os.path.join(gen_data, f'synthetic_tvae_{unique_id}_data.csv'), index=False)
    
    return synthetic_data