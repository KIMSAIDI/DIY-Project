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
        batch = y.shape[0]
        d = y.shape[1]
        errors = np.zeros(batch)  # accumulation des erreurs
        
        for i in range(batch) :
            # Calcul de l'erreur MSE pour chaque exemple et stockage dans le vecteur d'erreurs
            errors[i] = (np.sum((y[i]-y_hat[i])**2)) / d
            
        return np.array(errors)
    
    
    def backward(self, y, y_hat) :
        """
        Calcul le gradient de la fonction de perte MSE par rapport aux prédictions

        Args:
            y (numpy.ndarray): Les valeurs réelles, taille = batch * d.
            y_hat (numpy.ndarray): Les valeurs prédites, taille = batch * d.

        Returns:
            numpy.ndarray: Le gradient de la perte par rapport à y_hat, de même taille que y et y_hat.
        """
        
        return 2 * (y_hat - y) / y.shape[1] # Pour normaliser
        