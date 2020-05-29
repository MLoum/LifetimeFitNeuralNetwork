import numpy as np
from scipy.stats import rv_discrete
import multiprocessing as mp
import shelve

class DecayGenerator():
    def __init__(self, nb_trainning_decay=100, nb_test_decay=10, model="single_exp", params_dict={}):
        self.nb_trainning_decay = nb_trainning_decay
        self.nb_test_decay = nb_test_decay
        #FIXME
        self.time_nbins = 256
        self.time_step_ns = 0.25
        self.model = model
        self.params_dict = params_dict

        self.time_idx = np.arange(self.time_nbins)  # time axis in index units
        self.time_ns = self.time_idx * self.time_step_ns  # time axis in nano-seconds

        self.training_data = None
        self.test_data = None

    def generate_data(self):
        training_data = []
        test_data = []
        if self.model == "single_exp":
            min_tau = self.params_dict["tau"][0]
            max_tau = self.params_dict["tau"][1]
            t0 = self.params_dict["t0"][0]
            noise_min = self.params_dict["noise"][0]
            noise_max = self.params_dict["noise"][1]
            nb_photon_min = self.params_dict["nb_photon"][0]
            nb_photon_max = self.params_dict["nb_photon"][1]



            # Tirer au hasard les parametres
            taus = np.random.uniform(min_tau, max_tau, size=self.nb_trainning_decay)
            noises = np.random.uniform(noise_min, noise_max, size=self.nb_trainning_decay)
            nb_photons = np.random.uniform(nb_photon_min, nb_photon_max, size=self.nb_trainning_decay).astype(np.int)


            # Single core
            training_data = []
            for i in range(self.nb_trainning_decay):
                # TODO vectorization ? Mais avec des parametres tous differents ?
                training_data.append(self.generate_single_exp_decay(taus[i], t0, noises[i], nb_photons[i]))
            """
            ``training_data`` is a list containing 
            2-tuples ``(x, y)``.  ``x`` is a 256-dimensional numpy.ndarray
            containing the decay curve.  ``y`` is a 1-dimensional
            numpy.ndarray representing the unit vector corresponding to decay time ''tau''
            """
            self.training_data = training_data


            # Tirer au hasard les parametres
            taus = np.random.uniform(min_tau, max_tau, size=self.nb_test_decay)
            noises = np.random.uniform(noise_min, noise_max, size=self.nb_test_decay)
            nb_photons = np.random.uniform(nb_photon_min, nb_photon_max, size=self.nb_test_decay).astype(np.int)


            # Single core
            test_data = []
            for i in range(self.nb_test_decay):
                # TODO vectorization ? Mais avec des parametres tous differents ?
                test_data.append(self.generate_single_exp_decay(taus[i], t0, noises[i], nb_photons[i]))

            self.test_data = test_data
            """
            # Multicore
            def generate_single_exp_decay_fct(tau, t0, noise, nb_photons):
                return self.generate_single_exp_decay(tau, t0, noise, nb_photons)

            nb_of_workers = 4
            p = mp.Pool(nb_of_workers)
            training_data_list = [p.apply(generate_single_exp_decay_fct, args=(taus[i], t0, noises[i], nb_photons[i]))
                  for i in range(self.nb_trainning_decay)]
            """

    def generate_single_exp_decay(self, tau, t0, noise, nb_of_generated_photon):
        decay = np.exp(-(self.time_ns - t0) / tau)
        decay[self.time_ns < t0] = 0
        decay /= decay.sum()

        decay_obj = rv_discrete(name='mono_exp', values=(self.time_idx, decay))
        photons = decay_obj.rvs(size=nb_of_generated_photon)
        decay_data = np.bincount(photons)
        nb_of_pad_data = self.time_idx.size - decay_data.size
        zeros = np.zeros(nb_of_pad_data)
        decay_data = np.concatenate((decay_data, zeros))

        decay_data += np.random.random(self.time_idx.size) * noise

        #return (decay_data, np.array([tau, noise]))
        return (decay_data.reshape((256, 1)), np.array([tau]).reshape(1, 1))
        # return (decay_data, np.array([tau]))

    def generate_double_exp_decay(self, a1, a2, tau1, tau2, t0, noise, nb_of_generated_photon):
        C = 1 / (a1 * tau1 + a2 * tau2)
        decay = C * (a1 * np.exp(-(self.time_ns - t0) / tau1) + a2 * np.exp(-(self.time_ns - t0) / tau2))
        decay[self.time_ns < t0] = 0
        decay /= decay.sum()

        decay_obj = rv_discrete(name='biexpconv', values=(self.time_idx, decay))

        decay_data = decay_obj.rvs(size=nb_of_generated_photon) + np.random.random(self.time_idx.size) * noise
        return decay_data

    def convoluted_single_exp_decay(self):
        pass

    def save(self):
        #TODO with shelves
        pass

    def load(self):
        pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    params_dict_ = {}
    # Min / max
    params_dict_["tau"] = [1, 10]
    params_dict_["noise"] = [0, 0]
    params_dict_["nb_photon"] = [1000, 10000]
    params_dict_["t0"] = [0.5,0.5]
    decay_generator = DecayGenerator(nb_trainning_decay=10, nb_test_decay=2, model="single_exp", params_dict=params_dict_)
    decay_generator.generate_data()

    for training_data in decay_generator.training_data:
        plt.semilogy(decay_generator.time_ns, training_data[0], label="tau = " + str(training_data[1][0]) + " ns")

    plt.legend()
    plt.show()


