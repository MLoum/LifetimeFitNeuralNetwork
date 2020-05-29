from network_V1 import Network
from TrainingDataGenerator import DecayGenerator

params_dict_ = {}
# Min / max
params_dict_["tau"] = [1, 10]
params_dict_["noise"] = [0, 0]
params_dict_["nb_photon"] = [1000, 10000]
params_dict_["t0"] = [0.5, 0.5]
decay_generator = DecayGenerator(nb_trainning_decay=100, nb_test_decay=10, model="single_exp", params_dict=params_dict_)
decay_generator.generate_data()
training_data = decay_generator.training_data
test_data = decay_generator.test_data



# 256 points en entrée pour les déclins, en sortie, un seul resultat pour l'instant, le temps tau de déclin.
net = Network([256, 30, 1])
net.SGD(training_data, epochs=500, mini_batch_size=10, eta=0.01, test_data=test_data)