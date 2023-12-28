import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
import joblib
from pydantic import BaseModel

# data reading
# data = pd.read_csv('HousePricePrediction.csv')


# model
class Priceofhouse(BaseModel):
    LotArea :int
    OverallQual:int
    YearBuilt:int
    YearRemodAdd:int
    BsmtFinSF1:int
    BsmtUnfSF:int
    TotalBsmtSF:int
    GrLivArea:int
    FullBath:int
    BedroomAbvGr:int
    KitchenAbvGr:int
    TotRmsAbvGrd:int
    Fireplaces:int
    GarageCars:int
    YrSold:int
    


class Housepredictionmodel():
    def __init__(self):
        self.df = pd.read_csv('HousePricePrediction.csv')
        self.model_fname = 'housepriceprediction.pkl'
       
        try:
            self.model = joblib.load(self.model_fname)
        except:
            self.model = self.train_model()
            joblib.dump(self.model, self.model_fname)

    def train_model(self):
        X = self.df.drop('SalePrice', axis=1)
        y = self.df['SalePrice']
        rfc = RandomForestRegressor()
        model = rfc.fit(X, y)
        return model

    def predict_house(self, LotArea, OverallQual, YearBuilt, YearRemodAdd, BsmtFinSF1, BsmtUnfSF, TotalBsmtSF,
                      GrLivArea, FullBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageCars, YrSold):
        data_in = [[float(LotArea), float(OverallQual), float(YearBuilt), float(YearRemodAdd), float(BsmtFinSF1),
                    float(BsmtUnfSF), float(TotalBsmtSF), float(GrLivArea), float(FullBath), float(BedroomAbvGr),
                    float(KitchenAbvGr), float(TotRmsAbvGrd), float(Fireplaces), float(GarageCars), float(YrSold)]]

        prediction = self.model.predict(data_in)
        return prediction[0]

        