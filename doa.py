from typing import Any
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model

class DOA_estimator:

    def __init__(self, dataset_length, dataset_snapshots):
        self.dataset_creator(dataset_length, dataset_snapshots)

    def bartlett(self, signal, array_spacing, n_sources, n_bins) -> list:
        doa_estimates = []

        for freq_bin in range(n_bins):
            steering_vectors = np.exp(1j * 2 * np.pi * freq_bin * array_spacing * np.arange(n_sources) / signal.shape[0])
            power_spectrum = np.abs(np.dot(signal, steering_vectors.conj().T))**2
            max_power_index = np.argmax(power_spectrum)
            doa_estimate = np.degrees(np.arcsin(max_power_index / n_sources))
            doa_estimates.append(doa_estimate)

        return doa_estimates
    
    def capon(self, array, snapshot_matrix, n_sources) -> list:
        n_antennas = array.shape[0]
        covariance_matrix = np.dot(snapshot_matrix, snapshot_matrix.conj().T) / snapshot_matrix.shape[1]
        epsilon = 1e-6
        inverse_covariance = np.linalg.inv(covariance_matrix + epsilon * np.identity(n_antennas))
        capon_weights = np.dot(inverse_covariance, array) / np.dot(array.conj().T, np.dot(inverse_covariance, array))
        capon_spectrum = 1 / np.dot(capon_weights.conj().T, array)
        doa_estimates = np.argsort(capon_spectrum)[-n_sources:]
        
        return doa_estimates

    def music(self, data, n_sources, n_antennas, theta_range) -> list:
        cov_matrix = np.dot(data, data.conj().T) / data.shape[1]
        eigvals, eigvecs = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]
        noise_subspace = eigvecs[:, n_sources:]
        doa_estimates = []
        
        for theta in theta_range:
            steering_vector = np.exp(-1j * 2 * np.pi * np.arange(n_antennas) * np.sin(np.deg2rad(theta)))
            spectrum = 1 / np.linalg.norm(np.dot(noise_subspace.conj().T, steering_vector), axis=0)**2
            doa_estimates.append((theta, 10 * np.log10(spectrum.sum())))
        
        return doa_estimates
    
    def generate_data(self, array, n_snapshots, angle, wavelength):
        num_antennas = len(array)
        angle_radians = np.radians(angle)
        phase_delays = np.exp(1j * 2 * np.pi * np.sin(angle_radians) * array / wavelength)
        noise = np.random.randn(num_antennas, n_snapshots) + 1j * np.random.randn(num_antennas, n_snapshots)
        snapshot_matrix = phase_delays[:, np.newaxis] * noise
        
        return snapshot_matrix
    
    def dataset_creator(self, how_many:int, n_snapshots:int, wavelength:float=1):
        array = np.array([1, 2, 3, 4])

        dataset = []
        labels = []
        for i in range(how_many):
            angle = random.randint(0, 180)
            data = self.generate_data(array, n_snapshots, angle, wavelength)
            data = data.reshape(data.shape[0], data.shape[1], 1)
            dataset.append(data)
            labels.append(angle)
        
        with open("dataset.pkl", "wb") as f:
            pickle.dump([dataset, labels], f)

    def create_model(self, input):
        snapshots = Input(shape=(input.shape[1], input.shape[2], 1))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(snapshots)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.25)(x)
        angle = Dense(1, activation="linear")(x)
        model = Model(snapshots, angle)
        model.summary()
        model.compile(optimizer='adam', loss='mse')
        model.save("model.h5")

        return model

    def train(self, *, train_ratio:float=3, epochs):
        checkpoint = ModelCheckpoint("weights.h5", monitor="val_loss", mode="min", save_best_only=True)
        with open("dataset.pkl", "rb") as f:
            dataset, labels = pickle.load(f)

        train_ind = int(len(dataset) * train_ratio)
        train_data = dataset[:train_ind]
        train_labels = labels[:train_ind]
        val_data = dataset[train_ind:]
        val_labels = labels[train_ind:]

        model = self.create_model(train_data)
        model.fit(train_data, train_labels, epochs=epochs, validation_data=(val_data, val_labels), callbacks=[checkpoint])

    def predict(self, snapshot):
        model = load_model("model.h5")
        model = model.load_weights("weights.h5")
        angle = model.predict(snapshot)

        return angle