from qiskit_aer.primitives import Estimator
from qiskit.utils import  algorithm_globals

def estimator_Qiskit(seed,shot,noise_model):
    algorithm_globals.random_seed = seed
    seed_transpiler = seed
    # options = Options()
    estimator = Estimator(
    backend_options = {
        'method': 'statevector',
        'device': 'CPU',
        'noise_model': noise_model
    },
    run_options = {
        'shots': shot,
        'seed': seed,
        # 'options.resilience_level':2,
        # 'optimization_level':3
    },
    transpile_options = {
        'seed_transpiler':seed_transpiler
    }
)
    return estimator