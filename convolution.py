import numpy as np

class Conv1D(Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self._k_size = k_size
        self._chan_in = chan_in
        self._chan_out = chan_out
        
        self._stride = stride
        self._parameters = np.random.randn(k_size, chan_in, chan_out) * np.sqrt(2. / (k_size * chan_in))
        self._bias = np.zeros((1, chan_out))
        self._gradient = np.zeros((k_size, chan_in, chan_out))
        self._bias_gradient = np.zeros((1, chan_out))
               
    def zero_grad(self):
        """
        Réinitialise les gradients des poids et des biais.
        """
        self._gradient = np.zeros_like(self._parameters)
        self._gradient_bias = np.zeros_like(self.bias)
    
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
        Met à jour les poids et les biais de la couche de convolution en utilisant le taux d'apprentissage.

        Args:
            gradient_step : scalaire, pas du gradient
        """
        self._parameters -= gradient_step * self._gradient
        self._bias -= gradient_step * self._gradient_bias
        
        
        
        
class MaxPool1D(Module):
    def __init__(self, pool_size, stride):
        self._pool_size = pool_size
        self._stride = stride
        self._cache = None
        
        
    def forward(self, X):
        """
        Calcule la sortie de la couche de pooling.

        Args:
            X : taille = batch_size * length * chan_in

        Returns:
            numpy.ndarray : taille = batch_size * ((length - pool_size) / stride + 1 ) * chan_in
        """
        batch_size, length, _ = X.shape
        out_length = (length - self._pool_size) // self._stride + 1
        out = np.zeros((batch_size, out_length, self._chan_in))
        self._cache = np.zeros((batch_size, out_length, self._chan_in))
        for i in range(out_length):
            out[:, i, :] = np.max(X[:, i:i+self._pool_size, :], axis=1)
            self._cache[:, i, :] = np.argmax(X[:, i:i+self._pool_size, :], axis=1)
        return out
    
    
    def backward_delta(self, X, delta):
        """
        Calcule le gradient de la perte par rapport à l'entrée de la couche de pooling.

        Args:
            X : taille = batch_size * length * chan_in
            delta : taille = batch_size * ((length - pool_size) / stride + 1 ) * chan_in

        Returns:
            numpy.ndarray : taille = batch_size * length * chan_in
        """
        batch_size, length, _ = X.shape
        _, out_length, _ = delta.shape
        delta_X = np.zeros_like(X)
        for i in range(out_length):
            delta_X[:, i:i+self._pool_size, :][np.arange(batch_size)[:, None], self._cache[:, i, :], np.arange(self._chan_in)] = delta[:, i, :]
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