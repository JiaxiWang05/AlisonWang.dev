# Template code with parameter comments for team members
model = keras.Sequential()

# PARAMETER TO ADJUST: Number of neurons (try 8, 16, 32)
model.add(layers.Dense(16, activation='relu'))

# PARAMETER TO ADJUST: L1 regularization (try 0.01, 0.001, 0.0001)
model.add(layers.Dense(8, activation='relu', 
                      kernel_regularizer=keras.regularizers.l1(0.01)))

# PARAMETER TO ADJUST: L2 regularization (try 0.01, 0.001, 0.0001)  
model.add(layers.Dense(4, activation='relu',
                      kernel_regularizer=keras.regularizers.l2(0.01)))

model.add(layers.Dense(1, activation='sigmoid'))

# PARAMETER TO ADJUST: Learning rate (try 0.001, 0.01, 0.1)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
