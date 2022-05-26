from qiskit_finance.data_providers import YahooDataProvider
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit.algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit import IBMQ

import pandas as pd
import numpy as np
import datetime
import tqdm
import itertools
import os
import qiskit.algorithms.optimizers as opts

API_KEY = ""
try:
    IBMQ.load_account()
except Exception as e:
    api_key = API_KEY
    IBMQ.save_account(api_key, overwrite=True)
    IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open')

YEAR = 2021
SAMPLE_COUNT = 1024
RESULTS_FOLDER = "exp_results"

tickers_labels_all = [
    'AAPL',
    'MSFT',
    'AMZN',
    'TSLA',
    'GOOGL',
    'BRK-B',
    'FB',
    'UNH',
    'JNJ',
    'JPM',
    'V',
    'CVX',
]

better = [
    'ibmq_bogota',
    'ibmq_santiago',
    'ibmq_quito',
    'ibmq_belem',
    'ibmq_manila'
]
lookup = [ provider.get_backend(x) for x in better]

iters = [
    1,
    5,
    10,
    50,
    100,
    250,
    500,
]

opts = [
    opts.COBYLA,
    opts.ADAM,
    opts.POWELL,
    opts.SLSQP,
    opts.NELDER_MEAD,
]

qubits = [
    5,
    6,
    7,
    8,
    9,
    10,
    11,
]

pp = [
    1,
    2,
    3,
    4,
    5,
]

exps = list(itertools.product(lookup, iters, opts, qubits, pp))
for x in exps:
    print(x)
print(len(x))

num_assets = 0
rows = []

def get_qp(n):
    global num_assets
    tickers_labels = tickers_labels_all[:n]
    def normalize(v):
        return (v / np.linalg.norm(v)) * 100.0

    tickers = pd.DataFrame(tickers_labels, columns = ['Symbol'])

    data = YahooDataProvider(
        tickers.Symbol.to_list(),
        start=datetime.datetime(YEAR, 1, 1),
        end=datetime.datetime(YEAR, 12, 31),
    )
    data.run()
    mu = data.get_period_return_mean_vector()
    sigma = data.get_period_return_covariance_matrix()

    num_assets = len(tickers)
    q = 0.5
    budget = num_assets // 2
    penalty = num_assets

    portfolio = PortfolioOptimization(
        expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget
    )
    qp = portfolio.to_quadratic_program()

    return qp

def index_to_selection(i, num_assets):
    s = "{0:b}".format(i).rjust(num_assets)
    x = np.array([1 if s[i] == "1" else 0 for i in reversed(range(num_assets))])
    return x

def save_real_result(result, backend_name, iter_count, optimizer, qubits_count, qaoa_depth):
    global rows
    row = {}
    row['selection'] = ''.join(str(int(x)) for x in result.x)
    row['value'] = result.fval
    eigenstate = result.min_eigen_solver_result.eigenstate

    for i in range(2**num_assets):
        s = ''.join(str(x) for x in index_to_selection(i, num_assets)) 
        row[s] = eigenstate.get(s, 0.0) ** 2

    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv('{}/results_{}_{}_{}_{}_{}_{}.csv'.format(
        RESULTS_FOLDER,
        qaoa_depth,
        qubits_count,
        optimizer.__name__,
        backend_name,
        iter_count,
        '_'.join(t for t in tickers_labels_all[:qubits_count])
    ), index=False)

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

for backend, iter_count, optimizer, qubits_count, qaoa_depth in exps:
    backend_name = str(backend)
    print(backend_name, iter_count, optimizer, qubits_count)

    qp = get_qp(qubits_count)

    noise_model = NoiseModel.from_backend(backend)
    backend = AerSimulator(
        method='density_matrix',
        noise_model=noise_model
    )

    for _ in tqdm.tqdm(range(SAMPLE_COUNT), leave=False):

        optimizer_inst = optimizer()
        optimizer_inst.set_options(maxiter=iter_count)

        quantum_instance = QuantumInstance(backend=backend, seed_simulator=42, seed_transpiler=42)

        qaoa_mes = QAOA(optimizer=optimizer_inst, reps=qaoa_depth, quantum_instance=quantum_instance)
        qaoa = MinimumEigenOptimizer(qaoa_mes)

        result = qaoa.solve(qp)

        save_real_result(result, backend_name, iter_count, optimizer, qubits_count, qaoa_depth)
