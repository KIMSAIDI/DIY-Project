import numpy as np

class Loss(object):
    def forward(self, y, yhat): # fonction coût
        pass

    def backward(self, y, yhat): # gradient du coût
        pass



class MSELoss(Loss):
    """
    Fonction de perte pour la régression, la perte quadratique moyenne (MSE).
    """
    
    def forward(self, y, y_hat) :
        """
        Mesure la différence entre les valeurs prédites par le modèle y_hat 
        et les valeurs réelles y.

        Args:
            y (numpy.ndarray): valeurs réelles, taille = batch * d
            y_hat (numpy.ndarray): valeurs prédites, taille = batch * d

        Returns:
            numpy.ndarray : vecteur de dimension batch, chaque élément représente l'erreur moyenne au carré pour chaque exemple dans le batch
        """
        assert y.shape == y_hat.shape, ValueError("MSELoss : Les dimensions de y et y_hat doivent être les mêmes")
        return np.sum((y - y_hat)**2, axis=1) 
    
    
    def backward(self, y, y_hat) :
        """
        Calcul le gradient de la fonction de perte MSE par rapport aux prédictions

        Args:
            y (numpy.ndarray): Les valeurs réelles, taille = batch * d.
            y_hat (numpy.ndarray): Les valeurs prédites, taille = batch * d.

        Returns:
            numpy.ndarray: Le gradient de la perte par rapport à y_hat, de même taille que y et y_hat.
        """
        assert y.shape == y_hat.shape, ValueError("MSELoss : Les dimensions de y et y_hat doivent être les mêmes")
        return -2 * (y - y_hat) 
        
        
class CrossEntropyLoss(Loss):
    
    def forward(self, y, y_hat) :
        """
        Calcule la perte de la cross-entropie entre les prédictions et les valeurs réelles.

        Args:
            y (numpy.ndarray): Les valeurs réelles, taille = batch * d.
            y_hat (numpy.ndarray): Les valeurs prédites, taille = batch * d.

        Returns:
            numpy.ndarray: La perte de cross-entropie pour chaque exemple dans le batch.
        """
        assert y.shape == y_hat.shape, ValueError("CrossEntropyLoss : Les dimensions de y et y_hat doivent être les mêmes")
        return 1 - np.sum(y_hat * y, axis=1)
    
    
    def backward(self, y, y_hat) :
        """
        Calcule le gradient de la perte de cross-entropie par rapport aux prédictions.

        Args:
            y (numpy.ndarray): Les valeurs réelles, taille = batch * d.
            y_hat (numpy.ndarray): Les valeurs prédites, taille = batch * d.

        Returns:
            numpy.ndarray: Le gradient de la perte par rapport à y_hat, de même taille que y et y_hat.
        """
        assert y.shape == y_hat.shape, ValueError("CrossEntropyLoss : Les dimensions de y et y_hat doivent être les mêmes")
        return y_hat - y
    
    
class CrossEntropyLossLog(Loss) :
    
    def forward(self, y, y_hat):
        """
        Returns:
            numpy.ndarray : coûtt ; taille = batch_size 
        """
        return -(y * np.log(yhat + 1e-100) + (1 - y) * np.log(1 - yhat + 1e-100))

    def backward(self, y, yhat):
        """
        Returns:
            numpy.ndarray : gradient de coût ; taille = batch_size 
        """
        return ((1 - y) / (1 - yhat + 1e-100)) - (y / yhat + 1e-100)


class CrossEntropyLossSoftmax(Loss) :
    
    def forward(self, y, yhat):
        """
        Returns:
            numpy.ndarray : gradient de coût : batch_size * d
        """
        return np.log(np.sum(np.exp(yhat), axis=1) + 1e-100) - np.sum(y * yhat, axis=1)
    
    def backward(self, y, yhat):
        """
        Returns:
            numpy.ndarray : gradient de coût ; taille = batch_size
        """
        return np.exp(yhat) / np.sum(np.exp(yhat), axis=1).reshape((-1, 1)) - y
    
    
    
class BCELoss(Loss) :
    
    def forward(self, y, yhat):
        """
        Returns:
            numpy.ndarray : coût ; taille = batch_size
        """
        return - (y * np.log( yhat + 1e-20) + (1 - y) * p.log(1 - yhat + 1e-20))
    
    def backward(self, y, yhat):
        """
        Returns:
            numpy.ndarray : gradient de coût ; taille = batch_size
        """
        return ((1-y ) / (1 - yhat + 1e-20)) - (y / yhat + 1e-20)