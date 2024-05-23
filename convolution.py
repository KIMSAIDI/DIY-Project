import numpy as np
from module import Module

class Conv1D(Module):
    def __init__(self, _chan_in, _chan_out, _k_size, stride=1, padding=0):
        super().__init__()
        self._k_size = _k_size
        self._chan_in = _chan_in
        self._chan_out = _chan_out
        
        self._stride = stride
        self._parameters = np.random.randn(_k_size, _chan_in, _chan_out) * np.sqrt(2. / (_k_size * _chan_in))
        self._bias = np.zeros((1, _chan_out))
        self._gradient = np.zeros((_k_size, _chan_in, _chan_out))
        self._bias_gradient = np.zeros((1, _chan_out))
               
    def zero_grad(self):
        """
        Réinitialise les gradients des poids et des biais.
        """
        self._gradient = np.zeros_like(self._parameters)
        self._bias_gradient = np.zeros_like(self._bias)
    
    def forward(self, X):
        """
        Calcule la sortie de la couche de convolution.

        Args:
            X : taille = batch_size * length * chan_in

        Returns:
            numpy.ndarray : taille = batch_size * ((length - k_size) / stride + 1 ) * chan_out
        """
        batch_size, length, _ = X.shape
        out_length = (length - self._k_size) // self._stride + 1
        out = np.zeros((batch_size, out_length, self._chan_out))
        for i in range(out_length):
            out[:, i, :] = np.sum(X[:, i:i+self._k_size, :, np.newaxis] * self._parameters, axis=(1, 2)) + self._bias
        return out
    
    
    def backward_delta(self, X, delta):
        """
        Calcule le gradient de la perte par rapport à l'entrée de la couche de convolution.

        Args:
            X : taille = batch_size * length * chan_in
            delta : taille = batch_size * ((length - k_size) / stride + 1 ) * chan_out

        Returns:
            numpy.ndarray : taille = batch_size * length * chan_in
        """
        batch_size, length, _ = X.shape
        _, out_length, _ = delta.shape
        delta_X = np.zeros_like(X)
        for i in range(out_length):
            delta_X[:, i:i+self._k_size, :] += np.sum(delta[:, i, :, np.newaxis] * self._parameters, axis=2)
        return delta_X
    
    
    def update_parameters(self, gradient_step=1e-3):
        """
        met à jour les poids et les biais de la couche de convolution en utilisant le taux d'apprentissage.
        Args:
            gradient_step : scalaire, pas du gradient
        """
        self._parameters -= gradient_step * self._gradient
        self._bias -= gradient_step * self._bias_gradient
        
        
        
class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        self._k_size = k_size
        self._stride = stride
        self._cache = []
        self._parameters = np.zeros((1, 1))
        self._gradient = np.zeros((1, 1))
        
        
    def forward(self, X):
        """
        Calcule la sortie de la couche de pooling.
        Args:
            X : taille = batch_size * length * chan_in
        Returns:
            numpy.ndarray : taille = batch_size * ((length - k_size) / stride + 1 ) * chan_in
        """
        
        batch_size, length, chan_in = X.shape
        out_length = (length - self._k_size) // self._stride + 1
        out = np.zeros((batch_size, out_length, chan_in))
        self._cache = np.zeros((batch_size, out_length, chan_in), dtype=int)
        
        for i in range(out_length):
            segment = X[:, i*self._stride:i*self._stride+self._k_size, :]
            out[:, i, :] = np.max(segment, axis=1)
            self._cache[:, i, :] = np.argmax(segment, axis=1)
        return out
    
    def backward_delta(self, X, delta):
        """
        Calcule le gradient de la perte par rapport à l'entrée de la couche de pooling.

        Args:
            X : taille = batch_size * length * chan_in
            delta : taille = batch_size * ((length - k_size) / stride + 1 ) * chan_in

        Returns:
            numpy.ndarray : taille = batch_size * length * chan_in
        """
        batch_size, length, chan_in = X.shape
        _, out_length, _ = delta.shape
        delta_X = np.zeros_like(X)
        
        for i in range(out_length):
            delta_X[np.arange(batch_size)[:, None], self._cache[:, i, :], np.arange(chan_in)] += delta[:, i, :]
        return delta_X
    
    
class Flatten(Module):
    def __init__(self):
        pass
    
    def forward(self, X) :
        """
        Applati les données en une seule dimension.

        Args:
            X : taille = batch_size * length * chan_in

        Returns:
            numpy.ndarray : taille = batch_size * (length * chan_in)
        """
        return X.reshape(X.shape[0], -1)
    
    def backward_delta(self, X, delta):
        """
        Calcule le gradient de la perte par rapport à l'entrée de cette couche.

        Args:
            X : taille = batch_size * length * chan_in
            delta : taille = batch_size * (length * chan_in)

        Returns:
            numpy.ndarray : taille = batch_size * length * chan_in
        """
        return delta.reshape(X.shape)
    
    def update_parameters(self, gradient_step) :
        pass
    
    def backward_update_gradient(self, input, delta):
        pass