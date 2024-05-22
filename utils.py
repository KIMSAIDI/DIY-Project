import numpy as np
import matplotlib.pyplot as plt



# def charger_donnees(fichier_donnees):
#   """
#   Charge et prétraite les données d'apprentissage automatique.

#   Args:
#     fichier_donnees: Le chemin d'accès au fichier contenant les données.

#   Returns:
#     X: Matrice des données d'entrée (n_échantillons, n_caractéristiques).
#     y: Vecteur des étiquettes de classe (n_échantillons,).
#   """

#   # Charger les données brutes
#   if fichier_donnees.endswith('.csv'):
#     données = np.genfromtxt(fichier_donnees, delimiter=',')
#   elif fichier_donnees.endswith('.npy'):
#     données = np.load(fichier_donnees)
#   else:
#     raise ValueError(f"Format de fichier non reconnu: {fichier_donnees}")

#   # Séparer les caractéristiques et les étiquettes de classe
#   X = données[:, :-1]
#   y = données[:, -1]
#   return X, y



def plot_data(X, y=None):
    """
    Affiche des données 2D avec des labels optionnels.

    Args:
        X (numpy.ndarray): Matrice des données 2D.
        y (numpy.ndarray, optional): Vecteur des labels (discrets).
    """
    cols, marks = ["red", "green", "blue", "orange", "black", "cyan"], ["+", ".", "^", "o", "x", "*"]
    if y is None:
        plt.scatter(X[:, 0], X[:, 1], marker="x")
        return
    for i, l in enumerate(sorted(list(set(y.flatten())))):
        plt.scatter(X[y == l, 0], X[y == l, 1], c=cols[i], marker=marks[i])



def create_grid(data=None, x_min=-5, x_max=5, y_min=-5, y_max=5, step_size=20):
    """
    Crée une grille de points 2D pour la visualisation.

    Args:
        data (numpy.ndarray, optional): Données pour calculer les bornes de la grille.
        x_min (float): Borne inférieure pour l'axe x.
        x_max (float): Borne supérieure pour l'axe x.
        y_min (float): Borne inférieure pour l'axe y.
        y_max (float): Borne supérieure pour l'axe y.
        step_size (int): Nombre de points dans la grille.

    Returns:
        tuple: Une matrice 2D contenant les points de la grille, et les grilles x et y.
    """
    if data is not None:
        x_max, x_min = np.max(data[:, 0]) + 0.1, np.min(data[:, 0]) - 0.1
        y_max, y_min = np.max(data[:, 1]) + 0.1, np.min(data[:, 1]) - 0.1

    x_values = np.linspace(x_min, x_max, step_size)
    y_values = np.linspace(y_min, y_max, step_size)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid_points = np.c_[x_grid.ravel(), y_grid.ravel()]

    return grid_points, x_grid, y_grid


def generate_artificial_data(center=1, sigma=0.1, nbex=1000, data_type=0, epsilon=0.02):
    """
    Générateur de données artificielles.
    
    Args:
        center (float): Centre des gaussiennes.
        sigma (float): Écart-type des gaussiennes.
        nbex (int): Nombre d'exemples.
        data_type (int): Type de données (0: mélange 2 gaussiennes, 1: mélange 4 gaussiennes, 2: échiquier).
        epsilon (float): Bruit dans les données.
        
    Returns:
        tuple: data (numpy.ndarray), y (numpy.ndarray)
    """
    if data_type == 0:
        # Mélange de 2 gaussiennes
        pos_samples = np.random.multivariate_normal([center, center], [[sigma, 0], [0, sigma]], nbex // 2)
        neg_samples = np.random.multivariate_normal([-center, -center], [[sigma, 0], [0, sigma]], nbex // 2)
        data = np.vstack((pos_samples, neg_samples))
        y = np.hstack((np.ones(nbex // 2), -np.ones(nbex // 2)))
    
    elif data_type == 1:
        # Mélange de 4 gaussiennes
        pos_samples1 = np.random.multivariate_normal([center, center], [[sigma, 0], [0, sigma]], nbex // 4)
        pos_samples2 = np.random.multivariate_normal([-center, -center], [[sigma, 0], [0, sigma]], nbex // 4)
        neg_samples1 = np.random.multivariate_normal([-center, center], [[sigma, 0], [0, sigma]], nbex // 4)
        neg_samples2 = np.random.multivariate_normal([center, -center], [[sigma, 0], [0, sigma]], nbex // 4)
        data = np.vstack((pos_samples1, pos_samples2, neg_samples1, neg_samples2))
        y = np.hstack((np.ones(nbex // 2), -np.ones(nbex // 2)))
    
    elif data_type == 2:
        # Échiquier
        data = np.random.uniform(-4, 4, (nbex, 2))
        y = np.floor(data[:, 0]) + np.floor(data[:, 1])
        y = 2 * (y % 2) - 1
    
    # Ajouter du bruit
    data += np.random.normal(0, epsilon, data.shape)
    
    # Mélanger les données
    permutation = np.random.permutation(nbex)
    data = data[permutation]
    y = y[permutation]
    
    return data, y