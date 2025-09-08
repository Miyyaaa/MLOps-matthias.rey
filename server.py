from fastapi import FastAPI
import joblib
import uvicorn as unicorn

app = FastAPI()

@app.post("/predict/")
async def predict(size: int, nb_rooms: int, garden: int):
    model = joblib.load(filename="regression.joblib")
    y_pred = model.predict([[size, nb_rooms, garden]])
    return {"y_pred": y_pred[0]}

unicorn.run(app, host="0.0.0.0", port=8077)