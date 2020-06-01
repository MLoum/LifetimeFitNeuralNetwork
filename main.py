from network_V1 import Network
from TrainingDataGenerator import DecayGenerator
import numpy as np
import matplotlib.pyplot as plt

params_dict_ = {}
# Min / max
params_dict_["tau"] = [1, 10]
params_dict_["noise"] = [0, 0]
params_dict_["nb_photon"] = [1000, 10000]
params_dict_["t0"] = [0.5, 0.5]
decay_generator = DecayGenerator(nb_trainning_decay=2000, nb_test_decay=200, model="single_exp", params_dict=params_dict_)
decay_generator.generate_data()
training_data = decay_generator.training_data
test_data = decay_generator.test_data



# 256 points en entrée pour les déclins, en sortie, un seul resultat pour l'instant, le temps tau de déclin.
# net = Network([256, 30, 1])
# net.SGD(training_data, epochs=500, mini_batch_size=10, eta=0.001, test_data=test_data)
#
# net.last_epoch_assess(test_data, epochs=500, eta=0.001)

def calcul_un_pt_espace_phase(epochs, mini_batch_size, eta, network_size, training_data, test_data):
    net = Network(network_size)
    net.SGD(training_data, epochs=epochs, mini_batch_size=mini_batch_size, eta=eta, test_data=test_data)
    return net

nbs_neurones = [10, 20, 50, 100, 120, 150, 200, 250]
# nbs_neurones = [10, 20, 30]
means = []
stds = []


nb_epoch = 225
for nb_neurones in nbs_neurones:
    print(nb_neurones)
    size = [256, nb_neurones, 1]
    net = calcul_un_pt_espace_phase(epochs=nb_epoch, mini_batch_size=10, eta=0.001, network_size=size, training_data=training_data, test_data=test_data)
    net.last_epoch_assess(test_data, epochs=nb_epoch, eta=0.001)
    means.append(net.mean_erreur_relatives)
    stds.append(net.std_erreur_relatives)

plt.errorbar(nbs_neurones, means, yerr=stds, fmt="ro")
plt.xlabel("nb de neurones cachés")
plt.ylabel("Erreur relative en %")
plt.savefig("exploration_nb_neurones.png", dpi=300)
plt.show()

# espace_phase = np.zeros((dimX, dimY))
# x = np.arange(dimX)
# y = np.arange(dimY)
# XX, YY = np.meshgrid(x, y)  # Return coordinate matrices from coordinate vectors.
