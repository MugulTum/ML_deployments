from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import uvicorn

# Load your model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data schema
class ChurnPredictionRequest(BaseModel):
    TotalCharges: float
    MonthlyCharges: float
    Contract: int
    tenure: int
    OnlineSecurity: int
    TechSupport: int
    PaperlessBilling: int
    InternetService: int
    MultipleLines: int
    OnlineBackup: int

# Define a prediction endpoint
@app.post("/predict")
def predict_churn(data: ChurnPredictionRequest):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    prediction = int(prediction[0])
    # Return the prediction result
    return {"Churn Prediction": prediction}

if __name__=="__main__":
     uvicorn.run(app, host="127.0.0.1", port=8000)
# Run the app using the command below:
#uvicorn main:app --reload
