import sys


from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.inference import predict_churn
from fastapi import FastAPI 
from pydantic import BaseModel
app=FastAPI(title="TeleCommunication Churn Customer Prediction")

class Client(BaseModel):
    gender: str              
    SeniorCitizen: int      
    Partner: str             
    Dependents: str          
    tenure: int              
    PhoneService: str       
    MultipleLines: str       
    InternetService: str     
    OnlineSecurity: str      
    OnlineBackup: str        
    DeviceProtection: str    
    TechSupport: str         
    StreamingTV: str         
    StreamingMovies: str     
    Contract: str            
    PaperlessBilling: str    
    PaymentMethod: str       
    MonthlyCharges: float     
    TotalCharges:  float

@app.get("/chat")
def chat():
     return {"Status": "Okay!"}
@app.post("/predict")
def predict(client: Client):
     feature= predict_churn(client.dict())
     return {"churn_prediction": feature}
