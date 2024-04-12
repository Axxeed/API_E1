from fastapi import FastAPI
import joblib
import pandas as pd
import uvicorn

app = FastAPI()


model = joblib.load("xgb.pkl")

@app.post("/predict")
def predict(
    Code_postal_x: float,
    Surface_Carrez_du_1er_lot: float,
    Nombre_pieces_principales: float,
    code_region: float,
):
    modeldata_dict = {
        "Code postal_x": Code_postal_x,
        "Surface Carrez du 1er lot": Surface_Carrez_du_1er_lot,
        "Nombre pieces principales": Nombre_pieces_principales,
        "code_region": code_region,
    }

    index = [0]
    model_data_df = pd.DataFrame(modeldata_dict, index=index)

    result = model.predict(model_data_df).tolist()

    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
