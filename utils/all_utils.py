import numpy as np
import pandas as pd
import os 
import joblib

def preparedata(df):
  x=df.drop("y",axis=1)
  y=df["y"]
  return x,y

def save_model(model,filename):
  model_dir="model"
  os.makedirs(model_dir,exist_ok=True)
  filepath=os.path.join(model_dir,filename)
  joblib.dump(model,filepath)  