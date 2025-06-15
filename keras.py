import keras
from keras import layers

# Define a Sequential model
model = keras.Sequential(name="my_first_model")

# Add layers to the model. Each layer performs a specific computation.
# The 'Dense' layer is a standard, fully-connected layer.
model.add(layers.Dense(8, activation="relu")) # Hidden layer with 8 nodes
model.add(layers.Dense(4, activation="relu")) # Hidden layer with 4 nodes
model.add(layers.Dense(1, activation="sigmoid")) # Output layer

# In this structure, you simply .add() layers in the order you want the data to flow through them
