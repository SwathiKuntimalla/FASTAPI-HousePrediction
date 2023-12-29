

from fastapi import FastAPI
import uvicorn
from predict import Housepredictionmodel, Priceofhouse

app = FastAPI()
model = Housepredictionmodel()
@app.get('/')
async def root():
    return {"message": "House Price Prediction"}

@app.post('/prediction')
def prediction_house(house: Priceofhouse):
    data = house.dict()
    prediction = model.predict_house(data['LotArea'], data['OverallQual'], data['YearBuilt'], data['YearRemodAdd'],
                                     data['BsmtFinSF1'], data['BsmtUnfSF'], data['TotalBsmtSF'], data['GrLivArea'],
                                     data['FullBath'], data['BedroomAbvGr'], data['KitchenAbvGr'], data['TotRmsAbvGrd'],
                                     data['Fireplaces'], data['GarageCars'], data['YrSold'])
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
