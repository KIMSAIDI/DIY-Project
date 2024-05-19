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
        self.net = net
        self.loss = loss
        self.eps = eps 
        
        
    def step(self, batch_x, batch_y) :
        """
        Met à jour les paramètres du réseau en utilisant la descente de gradient.
        
        Args:
            batch_x : Entrée du réseau.
            batch_y : Sortie attendue du réseau.
            
        Returns:
            La valeur de la fonction de perte.                                                      
        """
        liste_forwards = self._net.forward(batch_x)

        batch_y_hat = liste_forwards[-1]
        loss = self._loss.forward(batch_y, batch_y_hat)

        delta = self._loss.backward(batch_y, batch_y_hat)
        list_deltas = self._net.backward_delta(liste_forwards, delta)

        self._net.update_parameters(self._eps)

        return loss
    
    
def SGD(self, data_x, data_y, batch_size, epochs=100):
    """
    Effectue une descente de gradient stochastique (SGD).
    Parameters:
        data_x : np.ndarray
            Jeu de données.
        data_y : np.ndarray
            Labels correspondants.
        batch_size : int
            Taille des lots de données.
        epochs : int, optional
            Nombre d'époques (itérations sur l'ensemble des données). Par défaut 100.
    Returns:
        liste_loss : list
            Liste des pertes moyennes par époque.
    """
    
    # Vérifications de type et de valeur
    if not isinstance(data_x, np.ndarray) or not isinstance(data_y, np.ndarray):
        raise ValueError("data_x et data_y doivent être des tableaux numpy.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size doit être un entier positif.")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs doit être un entier positif.")
    
    nb_data = len(data_x)
    nb_batches = (nb_data + batch_size - 1) // batch_size  # Calcul du nombre de lots

    liste_loss = []

    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Permuter les données
        perm = np.random.permutation(nb_data)
        shuffled_x = data_x[perm]
        shuffled_y = data_y[perm]

        # Effectue la descente de gradient pour chaque batch
        epoch_loss = 0
        for batch_idx in range(nb_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, nb_data)
            batch_x = shuffled_x[start_idx:end_idx]
            batch_y = shuffled_y[start_idx:end_idx]
            batch_loss = self.step(batch_x, batch_y)
            epoch_loss += batch_loss.mean()
        
        epoch_loss /= nb_batches  # Moyenne de la perte pour l'époque
        liste_loss.append(epoch_loss)

    return liste_loss
    
    
    def predict(self, X_test, types):
        """
        Prédit les labels pour les données de test données.
        Parameters:
            X_test : np.ndarray
                Jeu de données pour lequel les prédictions sont à effectuer.
            types : str
                Le type de sortie souhaité ('soft', 'conv', 'tanh', 'sig', 'enc').
        Returns:
            np.ndarray
                Les prédictions du modèle pour les données données.
        """
    
        # Vérification du type de X_test
        if not isinstance(X_test, np.ndarray):
            raise ValueError("X_test doit être un tableau numpy.")
        
        # Calculer la sortie du réseau
        net_output = self._net.forward(X_test)[-1]
        
        if types == 'soft':
            softmax = Softmax()
            return np.argmax(softmax.forward(net_output), axis=1)
        elif types == 'conv':
            softmax = Softmax()
            return softmax.forward(net_output)
        elif types == 'tanh':
            out = np.array(net_output)
            return np.where(out >= 0, 1, 0)
        elif types == 'sig':
            out = np.array(net_output)
            return np.where(out >= 0.5, 1, 0)
        elif types == 'enc':
            return self._net.forward(X_test)[0]
        else:
            raise ValueError("Type de prédiction non reconnu. Les types valides sont 'soft', 'conv', 'tanh', 'sig', 'enc'.")


    def accuracy(self, y_pred, y_test):
        """
        Calcule l'exactitude des prédictions.
        Parameters:
            y_pred : np.ndarray
                Prédictions du modèle.
            y_test : np.ndarray
                Labels réels.
        Returns:
            float
                Exactitude des prédictions.
        """
        
        # Vérification des types d'entrée
        if not isinstance(y_pred, np.ndarray):
            raise ValueError("y_pred doit être un tableau numpy.")
        if not isinstance(y_test, np.ndarray):
            raise ValueError("y_test doit être un tableau numpy.")
        
        # Vérification des dimensions
        if y_pred.shape != y_test.shape:
            raise ValueError("y_pred et y_test doivent avoir la même forme.")
        
        # Calcul de l'exactitude
        return np.sum(y_pred == y_test) / len(y_test)
            