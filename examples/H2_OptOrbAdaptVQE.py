import numpy as np
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit.algorithms.minimum_eigensolvers import VQE, AdaptVQE
from qiskit.algorithms.optimizers import L_BFGS_B, COBYLA
from qiskit.primitives import Estimator
from qiskit_nature.circuit.library import HartreeFock, UCCSD

from orbital_optimization import PartialUnitaryProjectionOptimizer, OptOrbAdaptVQE

from time import perf_counter
from functools import partial

estimator = Estimator()
qubit_converter = QubitConverter(mapper=JordanWignerMapper())

multiplicity = 1
charge = 0
spin = 0
interatomic_distance = 0.735
molecule = Molecule(geometry=[['H', [0., 0., 0.]],
                              ['H', [0., 0., interatomic_distance]]], 
                        charge=charge, multiplicity=multiplicity)

driver = PySCFDriver(atom=f'H 0 0 0; H 0 0 {0.735}',
                     charge=charge,
                     spin=spin,
                     unit=UnitsType.ANGSTROM,
                     basis='cc-pVTZ')
q_molecule = driver.run()
num_particles = q_molecule.get_property(ParticleNumber).num_particles

l_bfgs_b = L_BFGS_B(maxfun=10000, maxiter=10000)
cobyla = COBYLA(maxiter=10000)


num_reduced_qubits = 4

HF_state = HartreeFock(qubit_converter=qubit_converter, num_spin_orbitals=num_reduced_qubits, num_particles=num_particles)
ansatz = UCCSD(qubit_converter=qubit_converter, num_spin_orbitals=num_reduced_qubits, num_particles=num_particles, initial_state=HF_state)
 
vqe_start_time = perf_counter()
def store_intermediate_vqe_result(optorb_iteration, eval_count, parameters, mean, std):
        global vqe_start_time
        print(f' Outer loop iteration {optorb_iteration} vqe function evaluation: {eval_count}, energy: {mean}, time = {perf_counter() - vqe_start_time}')

        vqe_start_time = perf_counter()

def get_vqe_callback(optorb_iteration: int):

        return partial(store_intermediate_vqe_result, optorb_iteration)


orbital_rotation_start_time = perf_counter()
def store_intermediate_orbital_rotation_result(optorb_iteration, orbital_rotation_iteration, energy):
        global orbital_rotation_start_time
        print(f'Outer loop iteration {optorb_iteration} Iteration: {orbital_rotation_iteration}, energy: {energy}, time: {perf_counter() - orbital_rotation_start_time}')
        orbital_rotation_start_time = perf_counter()


def get_orbital_rotation_callback(optorb_iteration: int):

        return partial(store_intermediate_orbital_rotation_result, optorb_iteration)

partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                              stopping_tolerance=10**-5,
                                                              maxiter=10000,
                                                              gradient_method='autograd',
                                                              device='cpu')

vqe_instance = VQE(ansatz=ansatz,
                   initial_point=np.zeros(ansatz.num_parameters),
                   optimizer=l_bfgs_b,
                   estimator=estimator)

adaptvqe_instance = AdaptVQE(vqe_instance)

optorbadaptvqe_instance = OptOrbAdaptVQE(molecule_driver=driver,
                               integral_tensors= None,
                               num_spin_orbitals=num_reduced_qubits,
                               ground_state_solver=adaptvqe_instance,
                               qubit_converter=qubit_converter,
                               estimator=estimator,
                               initial_partial_unitary=None,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               maxiter=20,
                               wavefuntion_real=True,
                               spin_conserving=True,
                               stopping_tolerance=10**-5,
                               minimum_eigensolver_callback_func=get_vqe_callback,
                               orbital_rotation_callback_func=get_orbital_rotation_callback,
                               callback=None)

ground_state_energy_result = optorbadaptvqe_instance.compute_minimum_energy()
print(ground_state_energy_result)
