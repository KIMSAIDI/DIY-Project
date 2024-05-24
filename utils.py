import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns




def generate_artificial_data(center=1, sigma=0.1, nbex=1000, epsilon=0.02):
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
    
    # mélange 4 gaussiennes
    pos_samples1 = np.random.multivariate_normal([center, center], [[sigma, 0], [0, sigma]], nbex // 4)
    pos_samples2 = np.random.multivariate_normal([-center, -center], [[sigma, 0], [0, sigma]], nbex // 4)
    neg_samples1 = np.random.multivariate_normal([-center, center], [[sigma, 0], [0, sigma]], nbex // 4)
    neg_samples2 = np.random.multivariate_normal([center, -center], [[sigma, 0], [0, sigma]], nbex // 4)
    data = np.vstack((pos_samples1, pos_samples2, neg_samples1, neg_samples2))
    y = np.hstack((np.ones(nbex // 2), -np.ones(nbex // 2)))
    

    # ajoute du bruit
    data += np.random.normal(0, epsilon, data.shape)
    
    # mélange les données
    permutation = np.random.permutation(nbex)
    data = data[permutation]
    y = y[permutation]
    
    return data, y

def plot_loss(losses):
    """
    Affiche l'évolution de la perte au cours de l'entraînement.

    Args:
        losses (list): Liste des valeurs de la perte.
    """
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.grid()
    plt.title('Évolution de la perte au cours de l\'entraînement')
    plt.show()

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


def plot_frontiere(data, f, step=20):
    """ Trace la frontiere de decision d'une fonction f"""
    # pour tracer la frontière de décision
    grid, x, y = create_grid(data=data, step_size=step)
    plt.contourf(x, y, f(grid).reshape(x.shape), colors=('cyan', 'beige'), levels=[-1, 0, 1])


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



def plot_confusion_matrice(y_test, y_test_pred):
    """
    Affiche la matrice de confusion.

    Args:
        y_test : étiquettes de test
        y_test_pred : étiquettes prédites
    """
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_test_pred)

    # Affichage
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes')
    plt.title('Matrice de confusion')
    plt.show()
    

    
# Évaluation du modèle
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)
    accuracy = np.mean(correct_predictions)
    return accuracy

