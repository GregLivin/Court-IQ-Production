from fastapi import FastAPI
from src.courtiq.models.predict import predict_from_last_n

app = FastAPI(title="CourtIQ API")


@app.get("/")
def read_root():
    return {"message": "CourtIQ API is running"}


@app.get("/predict")
def predict(player: str = "Kevin Durant", n: int = 5):
    return predict_from_last_n(player_name=player, n=n)
