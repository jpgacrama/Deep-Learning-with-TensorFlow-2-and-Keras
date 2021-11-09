# Example using MNIST Dataset and Tensorflow's Estimator API

import tensorflow as tf
from tensorflow import feature_column as fc
numeric_column = fc.numeric_column
categorical_column_with_vocabulary_list = fc.categorical_column_with_vocabulary_list