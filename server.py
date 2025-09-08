from fastapi import FastAPI
import joblib

app = FastAPI()




@app.post("/predict/")
async def predict(size: int, nb_rooms: int, garden: int):
    model = joblib.load(filename="regression.joblib")
    y_pred = model.predict([[size, nb_rooms, garden]])
    return {"y_pred": y_pred[0]}