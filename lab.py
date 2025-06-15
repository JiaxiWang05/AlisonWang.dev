# Full example of a basic neural network in Keras

# Import necessary libraries
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# 1. Load your data (this example loads from a file)
# In your lab, you will be provided with educational software for this part[9].
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8] # Input features (columns 0-7)
y = dataset[:,8]  # Output label (column 8)

# 2. Define the Keras model by adding layers
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu')) # Hidden Layer 1
model.add(Dense(8, activation='relu'))                 # Hidden Layer 2
model.add(Dense(1, activation='sigmoid'))              # Output Layer

# 3. Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Train (fit) the model on the data
model.fit(X, y, epochs=150, batch_size=10)

# After training, you can evaluate its performance
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
