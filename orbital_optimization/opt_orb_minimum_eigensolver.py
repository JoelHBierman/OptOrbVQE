from time import perf_counter
from typing import Callable, Optional, Dict, List
from abc import abstractmethod
from qiskit.primitives import BaseEstimator
from qiskit_aer.primitives import Estimator
from .partial_unitary_projection_optimizer import PartialUnitaryProjectionOptimizer
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolverResult, MinimumEigensolver
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolverResult
from qiskit.utils.backend_utils import is_aer_provider
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.drivers.second_quantization import BaseDriver
import torch
import copy
import numpy as np
from pympler import asizeof

from .base_opt_orb_solver import BaseOptOrbSolver, BaseOptOrbResult

class OptOrbMinimumEigensolver(BaseOptOrbSolver):

    def __init__(self,
        molecule_driver: BaseDriver,
        integral_tensors: Optional[tuple[torch.Tensor, torch.Tensor]],
        num_spin_orbitals: int,
        ground_state_solver: MinimumEigensolver,
        qubit_converter: QubitConverter,
        estimator: BaseEstimator,
        partial_unitary_optimizer: PartialUnitaryProjectionOptimizer,
        initial_partial_unitary: Optional[torch.Tensor] = None,
        maxiter: int = 10,
        stopping_tolerance: float = 10**-5,
        spin_conserving: bool = False,
        wavefuntion_real: bool = False,
        callback: Optional[Callable] = None,
        minimum_eigensolver_callback_func: Optional[Callable] = None,
        orbital_rotation_callback_func: Optional[Callable] = None,
        partial_unitary_random_perturbation: Optional[float] = None,
        RDM_ops_batchsize: Optional[int] = None,
        spin_restricted: Optional[bool] = False):
        
        """
        
        Args:
            molecule_driver: The driver from which molecule information such as one and two body integrals is obtained.
            num_qubits: The number of budget qubits for the algorithm to use.
            ground_state_solver: An instance of VQE to use for the wavefunction optimization.
            qubit_converter: A QubitConverter to use for the RDM calculations.
            partial_unitary_optimizer: An instance of PartialUnitaryProjectionOptimizer to use for the basis optimization.
            initial_partial_unitary: The initial guess for the orbital rotation matrix. If ``None``, then a random
                partial unitary matrix will be generated.
            maxiter: The maximum number of outerloop iterations. (The number of times the wavefunction optimization is run.)
            stopping tolerance: The stopping tolerance used to determine if the algorithm should be stopped.
            spin_conserving: A boolean flag that indicates whether or not we are assuming that spin is conserved
                in the system being studied.  Setting to True will skip the calculation of RDM element operators
                which do not conserve spin, automatically setting these elements to zero.
            wavefunction_real: A boolean flag that indicates whether or not we are assuming that the wavefunction is real.
            callback: A callback function that tracks the outerloop progress.
                It takes the outerloop iteration, latest VQE results, and latest outer loop results as arguments.
                This can be used to save the most recent results to a file after each outer loop iteration.
            minimum_eigensolver_callback_func: An optional Callable which takes an integer (indexing the outer loop iteration) as an argument
                and returns a list of callback functions. We can use this to generate a separate callback function
                for each time we run VQE. This is useful for instance, in cases where we want a callback function that
                appends the VQE convergence information to a list, but we want to keep the data for each outerloop iteration 
                separate and append these data to separate lists. If ``None``, then the VQE callback function used will be vqe_instance.callback.
            orbital_rotation_callback_func: An optional Callable directly analogous to vqe_callback_func. This is a Callable that takes
                an integer index as an argument and returns another Callable. If ``None``, then the orbital rotation callback
                used will be partial_unitary_optimizer.callback.

        """
        # generate copies of the VQE instance to use for every outerloop iteration.
        super().__init__(molecule_driver=molecule_driver,
                         integral_tensors=integral_tensors,
                         num_spin_orbitals=num_spin_orbitals,
                         qubit_converter=qubit_converter,
                         estimator=estimator,
                         partial_unitary_optimizer=partial_unitary_optimizer,
                         initial_partial_unitary=initial_partial_unitary,
                         maxiter=maxiter,
                         stopping_tolerance=stopping_tolerance,
                         spin_conserving=spin_conserving,
                         wavefuntion_real=wavefuntion_real,
                         callback=callback,
                         orbital_rotation_callback_func=orbital_rotation_callback_func,
                         partial_unitary_random_perturbation=partial_unitary_random_perturbation,
                         RDM_ops_batchsize=RDM_ops_batchsize,
                         spin_restricted=spin_restricted)

        self._ground_state_solver_list = [copy.deepcopy(ground_state_solver) for n in range(int(maxiter+1))]
        self._energy_convergence_list = []
        self._pauli_ops_expectation_values_dict = None

    @property
    def ground_state_solver_list(self) -> List[MinimumEigensolver]:
        """Returns the list of ground state solver instances."""
        return self._ground_state_solver_list

    @ground_state_solver_list.setter
    def ground_state_solver_list(self, instance_list: List[MinimumEigensolver]) -> None:
        """Sets the list of ground state solver instances."""
        self._ground_state_solver_list = instance_list

    @property
    def energy_convergence_list(self) -> List[float]:
        """Returns the list of outerloop iteration energy values."""
        return self._energy_convergence_list

    @energy_convergence_list.setter
    def energy_convergence_list(self, energy_list: List[float]) -> None:
        """Sets the list of outerloop iteration energy values."""
        self._energy_convergence_list = energy_list

    @property
    def pauli_ops_expectation_values_dict(self) -> Dict:
        """Returns the dictionary containing all of the expectation values
            of all the Pauli string operators necessary for calculating
            the RDMs with respect to a particular wavefunction."""
        return self._pauli_ops_expectation_values_dict

    @pauli_ops_expectation_values_dict.setter
    def pauli_ops_expectation_values_dict(self, some_dict: Dict) -> None:
        """Sets the dictionary containing all of the expectation values
            of all the Pauli string operators necessary for calculating
            the RDMs with respect to a particular wavefunction."""
        self._pauli_ops_expectation_values_dict = some_dict

    def stopping_condition(self, iteration) -> bool:

        """Evaluates whether or not the stopping condition is True.
        Returns True if the algorithm should be stopped, otherwise returns False.
        """

        if len(self._energy_convergence_list) >= 2:
            if iteration == self.maxiter or np.abs(self._energy_convergence_list[-1] - self._energy_convergence_list[-2]) < self.stopping_tolerance:
                return True
            else:
                return False
        
        else:
            return False

    @abstractmethod            
    def parameter_update_rule(self, result: MinimumEigensolverResult, iteration: int):

        raise NotImplementedError("Minimum eigensolver needs to implement a way to update parameters after each orbital optimization.")

    @abstractmethod
    def return_RDM_circuit(self, result: MinimumEigensolverResult, iteration: int):

        NotImplementedError("Minimum eigensolver needs to implement a way to return the circuit used to calculate the one and two RDM.")

    def compute_minimum_energy(self) -> MinimumEigensolverResult:
        
        iteration = 0

        optorb_result = OptOrbMinimumEigensolverResult()
        self._hamiltonian = self.get_rotated_hamiltonian(self._current_partial_unitary)
        
        self._pauli_op_dict = self.construct_pauli_op_dict(qubit_converter=self.qubit_converter)

        while self.stopping_condition(iteration) == False:
            
            result = self._ground_state_solver_list[iteration].compute_minimum_eigenvalue(operator=self._hamiltonian)
            energy = np.real(result.eigenvalue)
            opt_params = result.optimal_parameters

            # update the optorb result to hold the most recent VQE value.
            optorb_result.eigenvalue = energy

            # update the optorb result to hold the most recent VQE parameters.
            optorb_result.optimal_parameters = opt_params

            # update the optorb result to hold the most recent partial unitary basis transformation.
            optorb_result.optimal_partial_unitary = self._current_partial_unitary
            optorb_result.num_vqe_evaluations += 1
            optorb_result.optimal_circuit = result.optimal_circuit
            optorb_result.optimal_point = result.optimal_point
            optorb_result.optimal_value = result.optimal_value

            if self.callback is not None:
                self.callback(iteration, result, optorb_result)

            self._energy_convergence_list.append(energy)
            state = copy.deepcopy(result.optimal_circuit).bind_parameters(opt_params)
            
            if self.stopping_condition(iteration) == True:
                break

            print(f'OptOrbVQE iteration: {iteration}')
            start_time = perf_counter()
            string_op_tuple_list = [(key, self._pauli_op_dict[key]) for key in self._pauli_op_dict]
            
            #results = self.estimator_list[iteration].run([state]*len(string_op_tuple_list), [op_tuple[1] for op_tuple in string_op_tuple_list]).result().values
            results = []
            ops_counter = 1
            num_ops = len(string_op_tuple_list)
            for op_tuple in string_op_tuple_list:
                
                results.append(self.estimator_list[iteration].run([state], [op_tuple[1]]).result().values[0])

                if self.RDM_ops_batchsize is not None:
                    if ops_counter % self.RDM_ops_batchsize == 0:
                        self.estimator_list[iteration] = copy.deepcopy(self.estimator)
                        print(f'computed {ops_counter} out of {num_ops} RDM operators')
                ops_counter += 1

            self._pauli_ops_expectation_values_dict = dict(zip([op_tuple[0] for op_tuple in string_op_tuple_list], results))
            
            stop_time = perf_counter()
            print(f'Pauli ops evaluation time: {stop_time - start_time} seconds')

            self.estimator_list[iteration] = None

            oneRDM = self.get_one_RDM_tensor(qubit_converter=self.qubit_converter, expectval_dict=self._pauli_ops_expectation_values_dict)
            twoRDM = self.get_two_RDM_tensor(qubit_converter=self.qubit_converter, expectval_dict=self._pauli_ops_expectation_values_dict)

            if self.partial_unitary_random_perturbation is not None:

                initial_partial_unitary = self.orth(self._current_partial_unitary + torch.Tensor(np.random.normal(loc=0.0,
                                                scale=self.partial_unitary_random_perturbation, size=(self._current_partial_unitary.size()[0],
                                                self._current_partial_unitary.size()[1]))))
            else:

                initial_partial_unitary = self._current_partial_unitary

            print('starting orbital opimization')
            oneRDM = oneRDM.to(self._partial_unitary_optimizer_list[iteration].device)
            twoRDM = twoRDM.to(self._partial_unitary_optimizer_list[iteration].device)
            self.one_body_integrals = self.one_body_integrals.to(self._partial_unitary_optimizer_list[iteration].device)
            self.two_body_integrals = self.two_body_integrals.to(self._partial_unitary_optimizer_list[iteration].device)
            self._current_partial_unitary = self._partial_unitary_optimizer_list[iteration].compute_optimal_rotation(fun=self.compute_rotated_energy,
                                                                                                                     oneRDM=oneRDM,
                                                                                                                     twoRDM=twoRDM,
                                                                                                                     one_body_integrals=self.one_body_integrals,
                                                                                                                     two_body_integrals=self.two_body_integrals,
                                                                                                                     initial_partial_unitary=initial_partial_unitary)[0]
            oneRDM = oneRDM.to('cpu')
            twoRDM = twoRDM.to('cpu')
            del oneRDM
            del twoRDM
            del string_op_tuple_list
            self.one_body_integrals = self.one_body_integrals.to('cpu')
            self.two_body_integrals = self.two_body_integrals.to('cpu')
            
            self._hamiltonian = self.get_rotated_hamiltonian(self._current_partial_unitary)
            iteration += 1

            self.parameter_update_rule(result, iteration)
            #self._ground_state_solver_list[iteration].initial_point = result.optimal_point

            self._partial_unitary_optimizer_list[iteration - 1] = None
            self._ground_state_solver_list[iteration - 1] = None

        return optorb_result

class OptOrbMinimumEigensolverResult(BaseOptOrbResult, MinimumEigensolverResult):

    def __init__(self) -> None:
        super().__init__()


