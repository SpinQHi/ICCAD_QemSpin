from qiskit_aer.primitives import Estimator
from qiskit.utils import algorithm_globals


def estimator_Qiskit(seed, shot, noise_model):
    """
    Construct a Qiskit estimator with specific seed, shot number, and noise model.

    Args:
        seed (int): Random seed for the estimator.
        shot (int): Number of shots to run for the estimator.
        noise_model (NoiseModel): Noise model to use in the estimator.

    Returns:
        Estimator: An instance of the Qiskit estimator.
    """

    # Set the random seed for global algorithms
    algorithm_globals.random_seed = seed

    # Create an estimator with backend and run options
    estimator = Estimator(
        backend_options={
            'method': 'statevector',
            'device': 'CPU',
            'noise_model': noise_model,
        },
        run_options={
            'shots': shot,
            'seed': seed,
        },
        transpile_options={
            'seed_transpiler': seed,
        }
    )

    return estimator
