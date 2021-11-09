# Example using MNIST Dataset and Tensorflow's Estimator API

import tensorflow as tf
from tensorflow import feature_column as fc
numeric_column = fc.numeric_column
categorical_column_with_vocabulary_list = fc.categorical_column_with_vocabulary_list

featcols = [
    tf.feature_column.numeric_column("area"),
    tf.feature_column.categorical_column_with_vocabulary_list("type",["bungalow","apartment"])
]

def train_input_fn():
    features = {
        "area"  :[1000,2000,4000,1000,2000,4000],
        "type"  :["bungalow","bungalow","house",
                  "apartment","apartment","apartment"]}
            
    labels = [ 500 , 1000 , 1500 , 700 , 1300 , 1900 ]
    return features, labels

model = tf.estimator.LinearRegressor(featcols)
model.train(train_input_fn, steps=200)

def predict_input_fn():
    features = {"area":[1500,1800],
                "type":["house","apt"]}
    return features

# Using the prediction functions
predictions = model.predict(predict_input_fn)
print(next(predictions))
print(next(predictions))