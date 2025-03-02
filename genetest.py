import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

# Function to encode genotypes
def encode_genotype(genotype_str):
    genotype = genotype_str.split('(')[-1].split(')')[0]  # e.g., 'G;G'
    alleles = genotype.split(';')  # Split by ';' for homozygous/heterozygous genotypes
    allele_map = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    encoded_alleles = [allele_map.get(allele, -1) for allele in alleles]  # -1 for unknown alleles
    return encoded_alleles

# Step 1: Load the trained model
model = load_model('genecnn.h5')  # Update this path with where your model is saved

# Step 2: Load the label encoder classes (ensure this is saved previously using np.save)
label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes  # Assign the loaded classes to label encoder

# Step 3: Preprocess the new data (new SNPs, Magnitude, Repute)
new_data = pd.DataFrame({
    'SNP': ['Rs661(A;A)'],  # Example SNPs
    'Magnitude': [9],
    'Repute': [1]
})

# Encode the genotypes in the SNP column
new_data['EncodedGenotype'] = new_data['SNP'].apply(encode_genotype)
new_data[['Allele1', 'Allele2']] = pd.DataFrame(new_data['EncodedGenotype'].to_list(), index=new_data.index)
new_data = new_data.drop(columns=['SNP', 'EncodedGenotype'])  # Drop SNP and EncodedGenotype columns

# Normalize Magnitude and Repute
scaler = StandardScaler()  # Use the same scaler that was fit on the training data
new_data[['Magnitude', 'Repute']] = scaler.fit_transform(new_data[['Magnitude', 'Repute']])

# Step 4: Reshape the input data to match the CNN input shape (samples, features, 1)
new_data_array = new_data.to_numpy()
new_data_reshaped = new_data_array.reshape(new_data_array.shape[0], new_data_array.shape[1], 1)

# Step 5: Predict using the trained model
predictions = model.predict(new_data_reshaped)

# Step 6: Decode the predicted labels (the class numbers) to actual disease names
predicted_classes = np.argmax(predictions, axis=1)  # Get the class index with the highest probability
predicted_diseases = label_encoder.inverse_transform(predicted_classes)

# Step 7: Output the results
print(f'Predicted disease classes: {predicted_classes}')
print(f'Predicted disease names: {predicted_diseases}')
