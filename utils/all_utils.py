import numpy as np
import pandas as pd

def preparedata(df):
  x=df.drop("y",axis=1)
  y=df["y"]
  return x,y