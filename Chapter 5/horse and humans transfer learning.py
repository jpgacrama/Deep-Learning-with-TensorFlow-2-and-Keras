#!/usr/bin/env python
# coding: utf-8

import os
import time
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1
    
    # Only allows for equal validation and test splits
    assert val_split == test_split 

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=12)
    indices_or_sections = [int(train_split * len(df)), int((1 - val_split - test_split) * len(df))]
    
    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
    
    return train_ds, val_ds, test_ds

SPLIT_WEIGHTS = (8, 1, 1)
data, metadata = tfds.load(
    'horses_or_humans', with_info=True, as_supervised=True)

raw_train, raw_validation, raw_test = get_dataset_partitions_pd(data)


# In[8]:


# print(raw_train)
# print(raw_validation)
# print(raw_test)


# In[ ]:


get_label_name = metadata.features['label'].int2str

def show_images(dataset): 
  for image, label in dataset.take(10):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))


# Let's inspect some images with an appropriate function
# 

# In[ ]:


show_images(raw_train)


# resize the image to (160x160) with input channels to a range of [-1,1]

# In[ ]:


IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label


# In[ ]:


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)


# In[ ]:


show_images(train)


# Then, we shuffle and batch the training set and batch the validation and test sets

# In[ ]:


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 2000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


# In[ ]:


print (train_batches)
print (validation_batches)
print (test_batches)


# We can now use MobileNet with input (160, 160, 3) where 3 is the number of color channels.
# The top layers are omitted (include_top=False) since we are going to use our own top layer.
# All the layers are frozen because we use use pretrained weights.
# 
# 

# In[ ]:


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


# In[ ]:


base_model.trainable = False


# In[ ]:


base_model.summary()


# In[ ]:


for image_batch, label_batch in train_batches.take(1):
  pass


# Let's inspect a batch and see if the shapes are correct (32, 160, 160, 3) - they are!
# 

# In[ ]:


print (image_batch.shape)


# MobileNetV2 transforms each 160x160x3 image into a 5x5x1280 block of features.
# For instance let's see the transformation applied to the batch

# In[ ]:


feature_batch = base_model(image_batch)
print(feature_batch.shape)


# Now, we can use GlobalAveragePooling2D() to average over the spatial 5x5 spatial locations and obtain a size of (32, 1280)
# 

# In[ ]:


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


# The last layer is a Dense with logit if the prediction is positive the class is 1, if the prediction is negative the class is 0
# 

# In[ ]:


prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


# Our model is ready to be composed by combined the base_model (MobileNet2 pre-trained), a global_average_layer to get the correct shape output given as input to the final prediction_layer
# 

# In[ ]:


model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])


# Now let's compile the model with an RMSProp() optimizer
# 

# In[ ]:


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# If we display the composed model, we notice that there are more than two millions frozen parameters, and more than one thousands trainable parameters
# 

# In[ ]:


model.summary()


# Let's compute the number of training, validation, and testing example

# In[ ]:


num_train, num_val, num_test = (
  metadata.splits['train'].num_examples*weight/10
  for weight in SPLIT_WEIGHTS
)


# In[ ]:


print (num_train, num_val, num_test)


# and compute the initial accuracy given by the pre-trained MobileNetv2
# 

# In[ ]:


initial_epochs = 20
steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps = 4

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)


# In[ ]:


print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


# We can now fine-tune the composed network with by training for a few iteration and optimizing the non-frozen layers
# 

# In[ ]:


history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)


# In[ ]:




