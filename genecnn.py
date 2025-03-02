import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Step 1: Load the data
file_path = r'C:/Users/mailv/OneDrive/Desktop/genedata.xlsx'
data = pd.read_excel(file_path)

# Step 2: Process SNP data (Encode genotypes in SNP column)
def encode_genotype(genotype_str):
    # Extract genotype (removes 'Rs' and parenthesis)
    genotype = genotype_str.split('(')[-1].split(')')[0]  # e.g., 'G;G'
    alleles = genotype.split(';')  # Split by ';' for homozygous/heterozygous genotypes

    # Map alleles to numeric values
    allele_map = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    encoded_alleles = [allele_map.get(allele, -1) for allele in alleles]  # -1 for unknown alleles
    return encoded_alleles

# Apply encoding function to SNP column
data['EncodedGenotype'] = data['SNP'].apply(encode_genotype)

# Split the encoded genotypes into separate columns for Allele1 and Allele2
data[['Allele1', 'Allele2']] = pd.DataFrame(data['EncodedGenotype'].to_list(), index=data.index)

# Drop original SNP and EncodedGenotype columns
data = data.drop(columns=['SNP', 'EncodedGenotype'])

# Step 3: Handle labels (Disease column)
label_encoder = LabelEncoder()
data['Disease'] = label_encoder.fit_transform(data['Disease'])  # Encode disease labels

# Step 4: Split data into features and labels
X = data.drop(columns=['Disease'])  # Features (Allele1, Allele2, Magnitude, Repute)
y = data['Disease']  # Labels (encoded diseases)

# Step 5: Normalize the features (Magnitude, Repute, Alleles)
scaler = StandardScaler()
X[['Magnitude', 'Repute']] = scaler.fit_transform(X[['Magnitude', 'Repute']])  # Scale Magnitude and Repute

# Step 6: Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Print shapes of data before reshaping
print(f"Original shape of X_train: {X_train.shape}")
print(f"Original shape of X_test: {X_test.shape}")

# Step 8: Reshape the input data to match the expected input shape for CNN (i.e., 2D)
# Reshape to 3D for CNN, where the 2nd dimension is the number of features and 3rd is the channel (1 for grayscale)
X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)  # 3D input for CNN
X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)  # 3D input for CNN

# Step 9: Print shapes after reshaping to ensure correctness
print(f"Reshaped shape of X_train: {X_train_reshaped.shape}")
print(f"Reshaped shape of X_test: {X_test_reshaped.shape}")

# Step 10: One-hot encode the labels (for multi-class classification)
y_train_categorical = to_categorical(y_train, num_classes=7)  # 7 classes (normal + 6 diseases)
y_test_categorical = to_categorical(y_test, num_classes=7)

# Step 11: Define and compile the CNN model
model = models.Sequential()

# Add 1D convolutional layers
model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1), padding='same'))
model.add(layers.MaxPooling1D(2))

model.add(layers.Conv1D(64, 3, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(2))

# Flatten the data for fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))

# Output layer with 7 classes (normal + 6 diseases)
model.add(layers.Dense(7, activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Step 12: Train the model
history = model.fit(X_train_reshaped, y_train_categorical, epochs=10, batch_size=32,
                    validation_data=(X_test_reshaped, y_test_categorical))

# Step 13: Evaluate the model
test_loss, test_acc = model.evaluate(X_test_reshaped, y_test_categorical)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# Save the model
model.save('genecnn.h5')  # Save model as .h5 file

# Optionally, also save the label encoder classes
import numpy as np
np.save('label_encoder_classes.npy', label_encoder.classes_)  # Save label encoder classes to file


# Step 14: Plot training history (accuracy and loss over epochs)
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

