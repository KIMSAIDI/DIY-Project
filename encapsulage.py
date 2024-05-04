import numpy as np

class Sequentiel:
    
    def __init__(self):
        self.modules = modules # [Module1, Module2, ...]
        
    def add_module(self, module):
        """
        Ajoute un module à la liste des modules du réseau.
        
        Args:
            module (Module): Module à ajouter.
        """
        self.modules.append(module)
        
    def forward_seq(self, input):
        """
        Calcule la sortie du réseau pour une entrée input.
        
        Args:
            input: Entrée du réseau.
            
        Returns:
            Liste des sorties de chaque module du réseau.
        """
        liste = []
        for module in self.modules :
            liste.append(module.forward(input))
        return liste
    
    def backward_delta_seq(self, input, delta):
        """
        Calcule le gradient de la perte par rapport à l'entrée du réseau.
        
        Args:
            input : Entrée du réseau.
            delta : Gradient de la perte par rapport à la sortie du réseau.
            
        Returns:
            Liste des gradients de la perte par rapport à l'entrée de chaque module du réseau.
        """
        liste = []
        for module in self.modules:
            list.append(module.backward_delta(input, delta))
        return liste        
    
    def backward_update_gradient_seq(self, input, delta):
        """
        Met à jour les gradient de chaque module du réseau.
        """
        for module in self.modules:
            module.backward_update_gradient(input, delta)
        
            
    def update_parameters_seq(self, gradient_step):
        """ 
        Met à jour les paramètres de chaque module du réseau.
        """
        for module in self.modules : 
            module.update_parameters(gradient_step)
            module.zero_grad()
            
    
    
    
class Optim() :
    def __init__(self, net, loss, eps):
        self.net = net # réseau
        self.loss = loss
        self.eps = eps # pas
        
        
    def step(self, batch_x, batch_y) :
        """
        Met à jour les paramètres du réseau en utilisant la descente de gradient.
        
        Args:
            batch_x : Entrée du réseau.
            batch_y : Sortie attendue du réseau.
        """
        output = self.net.forward_seq(batch_x)
        loss = self.loss.forward(output[-1], batch_y)
        delta = self.loss.backward(output[-1], batch_y)
        self.net.backward_update_gradient_seq(output[-2], delta)
        self.net.update_parameters_seq(self.eps)
        return loss
    
    
    def SGD(self, X, Y, batch_size, nb_epochs):
        """
        Entraine le réseau en utilisant la descente de gradient stochastique.
        
        Args:
            X : Entrée du réseau.
            Y : Sortie attendue du réseau.
            batch_size : Taille du batch.
            nb_epochs : Nombre d'itérations.
        """
        for i in range(nb_epochs):
            indices = np.random.permutation(X.shape[0])
            for j in range(0, X.shape[0], batch_size):
                batch_x = X[indices[j:j+batch_size]]
                batch_y = Y[indices[j:j+batch_size]]
                loss = self.step(batch_x, batch_y)
            print("Epoch {} : Loss = {}".format(i, loss))
            
        
                  
        
        