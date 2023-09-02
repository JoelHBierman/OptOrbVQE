from typing import Callable, Optional
from qiskit.primitives import BaseEstimator
from .partial_unitary_projection_optimizer import PartialUnitaryProjectionOptimizer
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolver, MinimumEigensolverResult
from qiskit.algorithms.exceptions import AlgorithmError
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.drivers.second_quantization import BaseDriver
import torch
from copy import deepcopy
import copy

from .opt_orb_minimum_eigensolver import OptOrbMinimumEigensolver, OptOrbMinimumEigensolverResult

class OptOrbAdaptVQE(OptOrbMinimumEigensolver):

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

        super().__init__(molecule_driver=molecule_driver,
                         integral_tensors=integral_tensors,
                         num_spin_orbitals=num_spin_orbitals,
                         ground_state_solver=ground_state_solver,
                         qubit_converter=qubit_converter,
                         estimator=estimator,
                         partial_unitary_optimizer=partial_unitary_optimizer,
                         initial_partial_unitary=initial_partial_unitary,
                         maxiter=maxiter,
                         stopping_tolerance=stopping_tolerance,
                         spin_conserving=spin_conserving,
                         wavefuntion_real=wavefuntion_real,
                         callback=callback,
                         minimum_eigensolver_callback_func=minimum_eigensolver_callback_func,
                         orbital_rotation_callback_func=orbital_rotation_callback_func,
                         partial_unitary_random_perturbation=partial_unitary_random_perturbation,
                         RDM_ops_batchsize=RDM_ops_batchsize,
                         spin_restricted=spin_restricted)

        if ground_state_solver.__class__.__name__ != 'AdaptVQE':

            raise AlgorithmError(f"The ground state solver needs to be of type AdaptVQE, not {ground_state_solver.__class__.__name__}")

        if ground_state_solver is not None:
            for n in range(int(maxiter)):
                self._ground_state_solver_list[n].solver.callback = minimum_eigensolver_callback_func(n)

    def parameter_update_rule(self, result: MinimumEigensolverResult,
                                    iteration: int):

        pass

class OptOrbAdaptVQEResult(OptOrbMinimumEigensolverResult):

    def __init__(self) -> None:
        super().__init__()

