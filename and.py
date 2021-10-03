from utils.model import perceptron
from utils.all_utils import preparedatac
import numpy as np
import pandas as pd

AND={
    "x1":[0,0,1,1],
     "x2":[0,1,0,1],
     "y":[0,0,0,1]
}
df=pd.DataFrame(AND)


x,y=preparedata(df)
eta=0.3
epochs=10
model=perceptron(eta=eta,epochs=epochs)
model.fit(x,y)
loss=model.total_loss()