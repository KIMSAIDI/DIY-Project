import numpy as np
import matplotlib.pyplot as plt


import numpy as np

def generate_data(nb_samples=1000, data_type="gaussian", centers=None, sigmas=None, epsilon=0.02):
    """Génère des données artificielles.

    Args:
        nb_samples (int): Nombre d'exemples.
        data_type (str): Type de données à générer ("gaussian", "checkerboard").
        centers (list of tuples): Liste des centres des gaussiennes. Nécessaire si data_type="gaussian".
        sigmas (list of floats): Liste des écart-types des gaussiennes. Nécessaire si data_type="gaussian".
        epsilon (float): Bruit dans les données.

    Returns:
        data (ndarray): Matrice 2D des données.
        labels (ndarray): Étiquettes des données.
    """
    if data_type == "gaussian":
        if centers is None or sigmas is None:
            raise ValueError("Les centres et les écart-types des gaussiennes doivent être spécifiés.")
        num_gaussians = len(centers)
        if num_gaussians != len(sigmas):
            raise ValueError("Le nombre de centres et d'écart-types doit être le même.")
        
        data = np.empty((0, 2))
        labels = np.empty(0, dtype=int)
        for i in range(num_gaussians):
            samples_per_gaussian = nb_samples // num_gaussians
            gaussian_samples = np.random.multivariate_normal(centers[i], np.diag([sigmas[i], sigmas[i]]), samples_per_gaussian)
            data = np.vstack((data, gaussian_samples))
            labels = np.concatenate((labels, np.full(samples_per_gaussian, i)))
        
    elif data_type == "checkerboard":
        side_length = int(np.sqrt(nb_samples))
        x_range = np.linspace(-4, 4, side_length)
        y_range = np.linspace(-4, 4, side_length)
        xx, yy = np.meshgrid(x_range, y_range)
        data = np.column_stack((xx.ravel(), yy.ravel()))
        labels = np.ceil(data[:, 0]) + np.ceil(data[:, 1])
        labels = 2 * (labels % 2) - 1
        
    elif data_type == "gaussian_4":
        num_gaussians = 4
        if centers is None or sigmas is None:
            raise ValueError("Les centres et les écart-types des gaussiennes doivent être spécifiés.")
        if num_gaussians != len(centers) or num_gaussians != len(sigmas):
            raise ValueError("Le nombre de centres et d'écart-types doit être le même.")
        
        data = np.empty((0, 2))
        labels = np.empty(0, dtype=int)
        for i in range(num_gaussians):
            samples_per_gaussian = nb_samples // num_gaussians
            gaussian_samples = np.random.multivariate_normal(centers[i], np.diag([sigmas[i], sigmas[i]]), samples_per_gaussian)
            data = np.vstack((data, gaussian_samples))
            labels = np.concatenate((labels, np.full(samples_per_gaussian, i)))
        
    else:
        raise ValueError("Type de données non valide.")

    # Ajout de bruit
    if data_type == "gaussian" or data_type == "gaussian_4":  # Ajout de bruit uniquement si les données sont gaussiennes
        data[:, 0] += np.random.normal(0, epsilon, nb_samples)
        data[:, 1] += np.random.normal(0, epsilon, nb_samples)

    # Mélange des données
    idx = np.random.permutation(range(labels.size))
    data = data[idx, :]
    labels = labels[idx]

    return data, labels.reshape(-1, 1)


def plot_data(data, labels, titre):
    """Affiche les points de données visuellement.

    Args:
        data (ndarray): Matrice 2D des données.
        labels (ndarray): Étiquettes des données.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', marker='o', edgecolors='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(titre)
    plt.colorbar(label='Labels')
    plt.grid(True)
    plt.show()



# Générer des données gaussiennes
centers = [(1, 1), (-1, -1)]
sigmas = [0.1, 0.1]
data, labels = generate_data(nb_samples=1000, data_type="gaussian", centers=centers, sigmas=sigmas)
plot_data(data, labels, 'Visualisation des données d\'un mélange de 2 gaussiennes')

# Générer des données en échiquier
data, labels = generate_data(nb_samples=1000, data_type="checkerboard")
plot_data(data, labels, 'Visualisation des données en échiquier')

# Générer des données avec un mélange de 4 gaussiennes
centers_4 = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
sigmas_4 = [0.1, 0.1, 0.1, 0.1]
data, labels = generate_data(nb_samples=1000, data_type="gaussian_4", centers=centers_4, sigmas=sigmas_4)
plot_data(data, labels, 'Visualisation des données d\'un mélange de 4 gaussiennes')