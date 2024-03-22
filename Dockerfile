# Usa l'immagine di Python 3.11.5 come base
FROM python:3.11.5

# Imposta il working directory all'interno del container
WORKDIR /api_backend

# Copia il file requirements.txt nella directory corrente del container
COPY requirements.txt .

# Installa le dipendenze definite nel file requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice nell'applicazione all'interno del container
COPY . .

# Comando di default per avviare l'applicazione FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]