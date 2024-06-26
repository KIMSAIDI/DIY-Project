import numpy as np


class Module(object):
    def __init__(self):
        self._parameters = None # stock les paramètres du module (e.g matrice de poids)
        self._gradient = None # accumule le gradient calculé

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X): # calcul les sorties du module
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta): # calcul le gradient de coût
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass
