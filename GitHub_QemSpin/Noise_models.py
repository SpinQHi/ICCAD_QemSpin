import pickle
from qiskit.providers.aer.noise import NoiseModel

def load_noise_model(file_path):
    """
    Utility function to load a noise model from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        NoiseModel: The loaded noise model.
    """

    with open(file_path, 'rb') as file:
        noise_dict = pickle.load(file)
        
    return NoiseModel.from_dict(noise_dict)

def fakekolkata():
    """
    Load the 'fakekolkata' noise model.

    Returns:
        NoiseModel: The 'fakekolkata' noise model.
    """

    return load_noise_model('NoiseModel/fakekolkata.pkl')

def fakecairo():
    """
    Load the 'fakecairo' noise model.

    Returns:
        NoiseModel: The 'fakecairo' noise model.
    """

    return load_noise_model('NoiseModel/fakecairo.pkl')


def fakemontreal():
    """
    Load the 'fakemontreal' noise model.

    Returns:
        NoiseModel: The 'fakemontreal' noise model.
    """

    return load_noise_model('NoiseModel/fakemontreal.pkl')
