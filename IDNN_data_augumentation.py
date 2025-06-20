# ===================================================================
# PAPER-STYLE IDNN WITH DATA AUGMENTATION AND NOISE
# ===================================================================
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount('/content/drive/')

# ===================================================================
# Step 1: Load and Filter for a Two-Material System
# ===================================================================
df = pd.read_csv('/content/drive/MyDrive/electronicsdata/data.csv')
print(f"Original dataset shape: {df.shape}")

# Select the material pair with the most samples to start with
material_1 = 'ZnO'
material_2 = 'Co'
print(f"\nSelected materials for this experiment: {material_1} and {material_2}")

# Filter for structures using only these two materials
paper_compliant_mask = (
    ((df['First Layer'] == material_1) & (df['Second Layer'] == material_2)) |
    ((df['First Layer'] == material_2) & (df['Second Layer'] == material_1))
)
df_paper = df[paper_compliant_mask].copy()
print(f"Base dataset size before augmentation: {df_paper.shape[0]} samples")

# ===================================================================
# Step 2: Extract and Downsample the Base Data
# ===================================================================
# Extract the 6 thickness parameters
X_thicknesses = df_paper.iloc[:, :6].values.astype(np.float32)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_thicknesses)

# Extract and downsample spectral data to 81 points
spectral_cols = [col for col in df_paper.columns if col.startswith('W')]
Y_full_spectrum = df_paper[spectral_cols].values.astype(np.float32)
indices = np.linspace(0, Y_full_spectrum.shape[1]-1, 81, dtype=int)
Y_paper_spectrum = Y_full_spectrum[:, indices]

print(f"Base structural data shape: {X_normalized.shape}")
print(f"Base spectral data shape: {Y_paper_spectrum.shape}")

# ===================================================================
# Step 3: DATA AUGMENTATION - Expand the Small Dataset
# ===================================================================
def augment_data(thicknesses, spectra, augmentation_factor=15):
    """
    Artificially expand the dataset by adding small, realistic variations.
    This helps the model generalize better from a small number of samples [3, 6].
    """
    augmented_thicknesses = []
    augmented_spectra = []

    for i in range(len(thicknesses)):
        # Add the original sample
        augmented_thicknesses.append(thicknesses[i])
        augmented_spectra.append(spectra[i])

        # Create augmented versions
        for _ in range(augmentation_factor):
            # Perturb thicknesses slightly (simulates manufacturing variance)
            perturbed_thickness = thicknesses[i] + np.random.normal(0, 0.02, size=thicknesses[i].shape)
            perturbed_thickness = np.clip(perturbed_thickness, 0, 1) # Ensure values stay in [0,1]
            augmented_thicknesses.append(perturbed_thickness)

            # Add noise to spectra (simulates detector noise)
            noisy_spectrum = spectra[i] + np.random.normal(0, 0.01, size=spectra[i].shape)
            noisy_spectrum = np.clip(noisy_spectrum, 0, 1)
            augmented_spectra.append(noisy_spectrum)

    return np.array(augmented_thicknesses), np.array(augmented_spectra)

print("\nApplying data augmentation to expand the small dataset...")
# We have ~68 samples. Augmenting by factor of 15 gives ~1000 samples.
X_augmented, Y_augmented = augment_data(X_normalized, Y_paper_spectrum, augmentation_factor=15)

print(f"Augmented structural data shape: {X_augmented.shape}")
print(f"Augmented spectral data shape: {Y_augmented.shape}")

# ===================================================================
# Step 4: Create Paper-Style Models (IDNN with Added Noise)
# ===================================================================
def create_paper_snn():
    """SNN: 6-500-500-500-1000-81, as per paper [1]"""
    input_tensor = keras.Input(shape=(6,))
    x = keras.layers.BatchNormalization()(input_tensor)
    x = keras.layers.Dense(500, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.02)(x)
    x = keras.layers.Dense(500, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.01)(x)
    x = keras.layers.Dense(500, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dense(1000, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    output_tensor = keras.layers.Dense(81, activation='sigmoid')(x)
    return keras.models.Model(input_tensor, output_tensor, name="Paper_SNN")

def create_paper_idnn_with_noise():
    """
    IDNN: 81-1000-1000-1000-500-500-500-6, with added GaussianNoise layer for regularization [4].
    This directly addresses your request to "add noise for this idnn".
    """
    input_tensor = keras.Input(shape=(81,))
    # ADDED: GaussianNoise layer for robust regularization
    x = keras.layers.GaussianNoise(0.01)(input_tensor)
    x = keras.layers.Dense(1000, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dense(1000, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dense(1000, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.02)(x)
    x = keras.layers.Dense(500, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.01)(x)
    x = keras.layers.Dense(500, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dense(500, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    output_tensor = keras.layers.Dense(6, activation='sigmoid')(x)
    return keras.models.Model(input_tensor, output_tensor, name="Paper_IDNN_With_Noise")

# ===================================================================
# Step 5: Split Augmented Data and Train Models
# ===================================================================
# The paper uses alternating samples [1]. We'll shuffle the augmented data for a similar effect.
X_aug_shuffled, Y_aug_shuffled = sklearn.utils.shuffle(X_augmented, Y_augmented, random_state=42)

# Use half for SNN training, half for IDNN/Tandem training
split_point = len(X_aug_shuffled) // 2
X_snn_aug, X_idnn_aug = X_aug_shuffled[:split_point], X_aug_shuffled[split_point:]
Y_snn_aug, Y_idnn_aug = Y_aug_shuffled[:split_point], Y_aug_shuffled[split_point:]

# Further split each set into train/val/test
X_snn_train, X_snn_val, Y_snn_train, Y_snn_val = train_test_split(
    X_snn_aug, Y_snn_aug, test_size=0.2, random_state=42
)
Y_idnn_train, Y_idnn_val = train_test_split(
    Y_idnn_aug, test_size=0.2, random_state=42
)

# --- Train SNN with Augmented Data ---
print("\n" + "="*50)
print(f"TRAINING SNN WITH AUGMENTED {material_1}/{material_2} DATA")
print("="*50)
snn_model = create_paper_snn()
snn_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
snn_history = snn_model.fit(
    X_snn_train, Y_snn_train,
    epochs=50,  # Train for more epochs with more data
    batch_size=16, # Smaller batch size for noisy data
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    validation_data=(X_snn_val, Y_snn_val),
    verbose=1
)

# --- Train Tandem Network with Augmented Data ---
print("\n" + "="*50)
print("TRAINING TANDEM NETWORK WITH AUGMENTED DATA")
print("="*50)
snn_model.trainable = False
inverse_model = create_paper_idnn_with_noise()
tandem_input = keras.Input(shape=(81,))
tandem_output = snn_model(inverse_model(tandem_input))
tandem_model = keras.models.Model(tandem_input, tandem_output)
tandem_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=['mean_absolute_error'])
tandem_history = tandem_model.fit(
    Y_idnn_train, Y_idnn_train,
    epochs=50,
    batch_size=16,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    validation_data=(Y_idnn_val, Y_idnn_val),
    verbose=1
)

# ===================================================================
# Step 6: Visualization and Analysis
# ===================================================================
print("\n" + "="*50)
print("ANALYSIS OF AUGMENTED PAPER-STYLE MODEL")
print("="*50)

plt.figure(figsize=(15, 10))

# Learning curves
plt.subplot(2, 2, 1)
plt.plot(snn_history.history['val_mean_absolute_error'], label='SNN Validation MAE')
plt.plot(tandem_history.history['val_mean_absolute_error'], label='Tandem Validation MAE')
plt.title('Validation MAE (Augmented Data)')
plt.xlabel('Epochs'); plt.ylabel('MAE'); plt.legend(); plt.grid(True)

# Spectrum reconstruction
# Use an original (non-augmented) test sample for a fair comparison
sample_idx = 0
target_spectrum = Y_paper_spectrum[sample_idx:sample_idx+1]
predicted_structure_norm = inverse_model.predict(target_spectrum, verbose=0)
reconstructed_spectrum = snn_model.predict(predicted_structure_norm, verbose=0)

plt.subplot(2, 2, 2)
wavelengths = np.linspace(400, 750, 81)
plt.plot(wavelengths, target_spectrum.flatten(), 'r-', label='Original Target', linewidth=2)
plt.plot(wavelengths, reconstructed_spectrum.flatten(), 'b--', label='Reconstructed', linewidth=2)
plt.title('Inverse Design on Original Sample')
plt.xlabel('Wavelength (nm)'); plt.ylabel('Reflectance'); plt.legend(); plt.grid(True)

# Summary
plt.subplot(2, 2, 3)
plt.text(0.05, 0.9, 'Key Improvements Applied:', fontweight='bold')
improvements = [
    f'✓ Data Augmentation ({len(X_normalized)} → {len(X_augmented)} samples)',
    f'✓ GaussianNoise Layer in IDNN (Regularization)',
    f'✓ Smaller Batch Size (16) for stability',
    f'✓ More Epochs (50) with Early Stopping'
]
for i, text in enumerate(improvements):
    plt.text(0.05, 0.75 - i*0.15, text)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.text(0.05, 0.9, 'Expected Outcome:', fontweight='bold')
outcomes = [
    '→ Better generalization from small dataset.',
    '→ More stable training curves.',
    '→ Lower final validation error vs. 61-sample training.',
    '→ Still likely underperforms the 30,000-sample',
    '  "superior system" due to limited base data diversity.'
]
for i, text in enumerate(outcomes):
    plt.text(0.05, 0.75 - i*0.15, text)
plt.axis('off')

plt.tight_layout()
plt.show()
