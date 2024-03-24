from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Definisci il motore SQLAlchemy
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, echo=True)  # Imposta echo=True per visualizzare le query eseguite

# Crea una sessione per interagire con il database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Definisci la classe base per i modelli
Base = declarative_base()

class Transaction(Base):
    __tablename__ = 'transactions'

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String)
    method = Column(String)
    url = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status_code = Column(Integer)  # Codice di stato HTTP della risposta

# Funzione per inizializzare il database e creare le tabelle
def init_db():
    Base.metadata.create_all(bind=engine)