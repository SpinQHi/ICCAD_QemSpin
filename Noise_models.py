from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
import pickle

def fakekolkata():
    with open('NoiseModel/fakekolkata.pkl', 'rb') as file:
        noise_model = pickle.load(file)
    noise_model1 = noise.NoiseModel()
    noise_fakekolkata = noise_model1.from_dict(noise_model)
    return noise_fakekolkata

def fakecairo():
    with open('NoiseModel/fakecairo.pkl', 'rb') as file:
        noise_model = pickle.load(file)
    noise_model1 = noise.NoiseModel()
    noise_fakecairo = noise_model1.from_dict(noise_model)
    return noise_fakecairo


def fakemontreal():
    with open('NoiseModel/fakemontreal.pkl', 'rb') as file:
        noise_model = pickle.load(file)
    noise_model1 = noise.NoiseModel()
    noise_fakemontreal = noise_model1.from_dict(noise_model)
    return noise_fakemontreal