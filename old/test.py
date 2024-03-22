import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

import torch
from ctgan import CTGAN
from ctgan import load_demo

# Verifica la disponibilità della GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Utilizzo del dispositivo:", device)

# path root folder
path = os.path.dirname(os.path.abspath(__file__))

# Carica dati di esempio
real_data = pd.read_csv(os.path.join("data", "adult.csv"))  # Assicurati che il percorso del file sia corretto

# Stampa le prime righe dei dati
print(real_data.head())

"""# Codifica one-hot delle variabili categoriche
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first category to avoid multicollinearity
encoded_categorical_data = pd.DataFrame(encoder.fit_transform(real_data[categorical_columns]))
encoded_categorical_data.columns = encoder.get_feature_names_out(categorical_columns)

# Rinomina le colonne in base alle categorie originali
new_column_names = []
for column in categorical_columns:
    unique_categories = real_data[column].unique()
    for category in unique_categories[1:]:  # Ignora la prima categoria per evitare duplicati
        new_column_names.append(f"{column}_{category}")

# Assegna i nuovi nomi delle colonne
encoded_categorical_data.columns = new_column_names

# Concatenazione delle variabili codificate con le altre colonne
real_data_encoded = pd.concat([real_data.drop(categorical_columns, axis=1), encoded_categorical_data], axis=1)

# Nomi delle colonne che sono discrete dopo la codifica one-hot
discrete_columns = list(real_data_encoded.columns)"""

# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income']

# Inizializza e addestra il modello CTGAN
ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns)

# Salva il modello addestrato su disco
models_folder = "models"  # Nome della cartella
os.makedirs(models_folder, exist_ok=True)
with open(os.path.join(models_folder, 'ctgan_model.pkl'), 'wb') as f:
    pickle.dump(ctgan, f)

# Esegui l'inferenza per generare dati sintetici
num_samples = 32561  # Numero di campioni sintetici da generare
synthetic_data = ctgan.sample(num_samples)

# Salva i dati sintetici in un file CSV
new_data = "data"
os.makedirs(new_data, exist_ok=True)
synthetic_data.to_csv(os.path.join(new_data, 'synthetic_data.csv'), index=False)

# Funzione per calcolare il log-likelihood sintetico
def compute_synthetic_log_likelihood(real_data, synthetic_data):
    try:
        # Calcola le probabilità di ogni valore nei dati sintetici
        synthetic_probs = synthetic_data.value_counts(normalize=True)

        # Seleziona solo i valori presenti sia nei dati reali che sintetici
        common_values = real_data[real_data.isin(synthetic_data.unique())].unique()

        # Calcola il logaritmo delle probabilità dei valori comuni
        log_likelihoods = np.log(synthetic_probs[common_values].values)

        # Somma dei log-likelihoods
        synthetic_log_likelihood = log_likelihoods.sum()

        return synthetic_log_likelihood / len(real_data)
    except Exception as e:
        print(f"Error in compute_synthetic_log_likelihood: {e}")
        return None

# Funzione per creare istogrammi delle distribuzioni
def plot_distributions(real_data, synthetic_data, column, save_path):
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        sns.histplot(real_data[column], ax=ax[0], color='blue', kde=True)
        ax[0].set_title('Real Data')

        sns.histplot(synthetic_data[column], ax=ax[1], color='orange', kde=True)
        ax[1].set_title('Synthetic Data')

        plt.suptitle(f'Distribution of {column}')
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error in plot_distributions: {e}")

# Ciclo for per generare istogrammi e calcolare le metriche per ogni colonna
imgs_folder = "img"
os.makedirs(imgs_folder, exist_ok=True)
for column in real_data.columns:
    try:
        # Path per salvare l'immagine
        save_path = os.path.join(imgs_folder, f'distribution_{column}.png')

        # Calcola il log-likelihood sintetico
        synthetic_log_likelihood = compute_synthetic_log_likelihood(real_data[column], synthetic_data[column])
        if synthetic_log_likelihood is not None:
            print(f"{column} Synthetic Log-Likelihood:", synthetic_log_likelihood)

        # Calcola l'errore quadratico medio (MSE)
        mse = mean_squared_error(real_data[column], synthetic_data[column])
        print(f"{column} Mean Squared Error:", mse)

        # Genera e salva l'istogramma
        plot_distributions(real_data, synthetic_data, column, save_path)
    except Exception as e:
        print(f"Error in column {column}: {e}")
