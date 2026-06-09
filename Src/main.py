#this is the main file for the project, it will be used to run the program and call the other files
from model import prediction_model
from warnings import filterwarnings
filterwarnings('ignore')

data = [89, 1, 170, 70.5, 120, 80, 1, 1, 1, 1, 1]
pred = prediction_model(data)
print(pred)