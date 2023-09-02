from time import perf_counter
from typing import Callable, Optional, Dict, List
from qiskit import QuantumCircuit, transpile
from .partial_unitary_projection_optimizer import PartialUnitaryProjectionOptimizer
from qiskit.algorithms.minimum_eigen_solvers import VQE
from qiskit.algorithms.minimum_eigen_solvers import MinimumEigensolverResult
from qiskit.algorithms.variational_algorithm import VariationalResult
from qiskit.utils import QuantumInstance
from qiskit.opflow import CircuitStateFn, StateFn, CircuitSampler, ExpectationBase, PauliSumOp, OperatorBase, ExpectationFactory
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit.utils.backend_utils import is_aer_provider
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy
from qiskit_nature.drivers.second_quantization import BaseDriver
import torch
import copy
import numpy as np

class OptOrbVQE():

    def __init__(self,
        molecule_driver: BaseDriver,
        num_spin_orbitals: int,
        vqe_instance: VQE,
        qubit_converter: QubitConverter,
        quantum_instance: QuantumInstance,
        expectation: ExpectationBase,
        partial_unitary_optimizer: PartialUnitaryProjectionOptimizer,
        initial_partial_unitary: Optional[torch.Tensor] = None,
        maxiter: int = 10,
        stopping_tolerance: float = 10**-5,
        spin_conserving: bool = False,
        wavefuntion_real: bool = False,
        callback: Optional[Callable] = None,
        vqe_callback_func: Optional[Callable] = None,
        orbital_rotation_callback_func: Optional[Callable] = None):
        
        """
        
        Args:
            molecule_driver: The driver from which molecule information such as one and two body integrals is obtained.
            num_qubits: The number of budget qubits for the algorithm to use.
            vqe_instance: An instance of VQE to use for the wavefunction optimization.
            qubit_converter: A QubitConverter to use for the RDM calculations.
            quantum_instance: A QuantumInstance to use for the RDM calculations.
            expectation: An ExpectationBase to use for the RDM calculations.
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
            vqe_callback_func: An optional Callable which takes an integer (indexing the outer loop iteration) as an argument
                and returns a list of callback functions. We can use this to generate a separate callback function
                for each time we run VQE. This is useful for instance, in cases where we want a callback function that
                appends the VQE convergence information to a list, but we want to keep the data for each outerloop iteration 
                separate and append these data to separate lists. If ``None``, then the VQE callback function used will be vqe_instance.callback.
            orbital_rotation_callback_func: An optional Callable directly analogous to vqe_callback_func. This is a Callable that takes
                an integer index as an argument and returns another Callable. If ``None``, then the orbital rotation callback
                used will be partial_unitary_optimizer.callback.

        """
        # generate copies of the VQE instance to use for every outerloop iteration.
        self._vqe_instance_list = [copy.deepcopy(vqe_instance) for n in range(maxiter+1)]
        if vqe_callback_func is not None:
            for n in range(maxiter):
                self._vqe_instance_list[n].callback = vqe_callback_func(n)

        self._qubit_converter = qubit_converter
        self._quantum_instance = quantum_instance
        self._expectation = expectation

        # generate copies of the PartialUnitaryProjectionOptimizer instance to use for every outerloop iteration.
        self._partial_unitary_optimizer_list = [copy.deepcopy(partial_unitary_optimizer) for n in range(maxiter+1)]
        if orbital_rotation_callback_func is not None:
            for n in range(maxiter+1):
                self._partial_unitary_optimizer_list[n].callback = orbital_rotation_callback_func(n)

        self._molecule_driver = molecule_driver
        initial_molecule = molecule_driver.run()
        
        self._one_body_integrals = torch.from_numpy(initial_molecule.one_body_integrals)
        self._two_body_integrals = torch.from_numpy(initial_molecule.two_body_integrals)
        
        if initial_partial_unitary is None:
            initial_numpy_array = np.random.uniform(size=(2*initial_molecule.num_molecular_orbitals, num_spin_orbitals))
            init_V = torch.from_numpy(initial_numpy_array)

            init_L, init_Q = torch.linalg.eigh(torch.t(init_V) @ init_V)
            self._initial_partial_unitary = init_V @ init_Q @ (torch.float_power(torch.inverse(torch.diag(init_L)), 0.5)) @ torch.t(init_Q).double()
            self._initial_partial_unitary.requires_grad = False
        else:
            self._initial_partial_unitary = initial_partial_unitary
        
        self._maxiter = maxiter
        self._spin_conserving = spin_conserving
        self._wavefunction_real = wavefuntion_real
        self._callback = callback

        self._hamiltonian = None
        self._energy_convergence_list = []
        self._iteration = 0
        self._stopping_tolerance = stopping_tolerance
        self._current_partial_unitary = self._initial_partial_unitary
        self._pauli_op_dict = None
        self._pauli_ops_expectation_values_dict = None
        self._num_spin_orbitals = num_spin_orbitals

        del initial_molecule

    @property
    def molecule_driver(self) -> BaseDriver:
        """Returns the molecule driver."""
        return self._molecule_driver

    @molecule_driver.setter
    def molecule_driver(self, mol_driver: BaseDriver) -> None:
        """Sets the molecule driver."""
        self._molecule_driver = mol_driver

    @property
    def spin_conserving(self) -> bool:
        """Returns the boolean flag indicating whether or not
            we are assuming spin is conserved."""
        return self._spin_conserving
    
    @spin_conserving.setter
    def spin_conserving(self, spin_conserve: bool) -> None:
        """Sets the spin conserving boolean flag."""
        self._spin_conserving = spin_conserve

    @property
    def wavefunction_real(self) -> bool:
        """Returns the boolean flag indicating whether or not
            we are assuming that the wavefunction is real."""
        return self._wavefunction_real

    @wavefunction_real.setter
    def wavefunction_real(self, wavefunc_real: bool) -> None:
        """Sets the real wavefunction boolean flag."""
        self._wavefunction_real = wavefunc_real

    @property
    def vqe_instance_list(self) -> List[VQE]:
        """Returns the list of VQE instances."""
        return self._vqe_instance_list

    @vqe_instance_list.setter
    def vqe_instance_list(self, instance_list: List[VQE]) -> None:
        """Sets the list of VQE instances."""
        self._vqe_instance_list = instance_list

    @property
    def qubit_converter(self) -> QubitConverter:
        """Returns the QubitConverter used for RDM calculations."""
        return self._qubit_converter

    @qubit_converter.setter
    def qubit_converter(self, converter: QubitConverter) -> None:
        """Sets the QubitConverter used for RDM calculations."""
        self._qubit_converter = converter

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Return the QuantumInstance used for RDM calculations."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: QuantumInstance) -> None:
        """Sets the QuantumInstance used for RDM calculations."""
        self._quantum_instance = quantum_instance

    @property
    def expectation(self) -> ExpectationBase:
        """Returns the ExpectationBase used for RDM calculations."""
        return self._expectation

    @expectation.setter
    def expectation(self, some_expect: ExpectationBase) -> None:
        """Sets the ExpectationBase used for RDM calculations."""
        self._expectation = some_expect

    @property
    def callback(self) -> Callable:
        """Sets the callback function used to track outer loop iteration
            information."""
        return self._callback

    @callback.setter
    def callback(self, func: Callable) -> None:
        """Sets the outerloop callback function."""
        self._callback = func

    @property
    def partial_unitary_optimizer_list(self) -> List[PartialUnitaryProjectionOptimizer]:
        """Returns the list of partial unitary optimizers used for each outer loop iteration."""
        return self._partial_unitary_optimizer_list

    @partial_unitary_optimizer_list.setter
    def partial_unitary_optimizer_list(self, optimizer_list: List[PartialUnitaryProjectionOptimizer]) -> None:
        """Sets the list of partial unitary optimizers used for each outer loop iteration."""
        self._partial_unitary_optimizer_list = optimizer_list

    @property
    def one_body_integrals(self) -> torch.Tensor:
        """Returns the one body integrals obtained from the molecule driver."""
        return self._one_body_integrals

    @one_body_integrals.setter
    def one_body_integrals(self, integrals: torch.Tensor) -> None:
        """Sets the one body integrals obtained from the molecule driver."""
        self._one_body_integrals = integrals

    @property
    def two_body_integrals(self) -> torch.Tensor:
        """Returns the two body integrals obtained from the molecule driver."""
        return self._two_body_integrals

    @two_body_integrals.setter
    def two_body_integrals(self, integrals: torch.Tensor) -> None:
        """Sets the two body integrals obtained from the molecule driver."""
        self._two_body_integrals = integrals

    @property
    def num_spin_orbitals(self)  -> int:
        """Returns the number of budget spin orbitals."""
        return self._num_spin_orbitals

    @num_spin_orbitals.setter
    def num_spin_orbitals(self, some_int: int) -> None:
        """Sets the number of budget spin orbitals"""
        self._num_spin_orbitals = some_int

    @property
    def maxiter(self) -> int:
        """Returns the maximum number of outer loop iterations."""
        return self._maxiter

    @maxiter.setter
    def maxiter(self, num_iterations: int) -> None:
        """Sets the maximum number of outer loop iterations."""
        self._maxiter = num_iterations

    @property
    def current_partial_unitary(self) -> torch.Tensor:
        """Returns the current basis set rotation partial unitary matrix."""
        return self._current_partial_unitary

    @current_partial_unitary.setter
    def current_partial_unitary(self, unitary: torch.Tensor) -> None:
        """Sets the current basis set rotation partial unitary matrix."""
        self._current_partial_unitary = unitary

    @property
    def hamiltonian(self) -> OperatorBase:
        """Returns the Hamiltonian in the current basis."""
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, op: OperatorBase) -> None:
        """Sets the Hamiltonian in the current basis."""
        self._hamiltonian = op

    @property
    def energy_convergence_list(self) -> List[float]:
        """Returns the list of outerloop iteration energy values."""
        return self._energy_convergence_list

    @energy_convergence_list.setter
    def energy_convergence_list(self, energy_list: List[float]) -> None:
        """Sets the list of outerloop iteration energy values."""
        self._energy_convergence_list = energy_list

    @property
    def iteration(self) -> int:
        """Returns the current outer loop iteration."""
        return self._iteration

    @iteration.setter
    def iteration(self, num: int) -> None:
        """Sets the current outer loop iteration."""
        self._iteration = num

    @property
    def stopping_tolerance(self) -> float:
        """Returns the stopping tolerance."""
        return self._stopping_tolerance

    @stopping_tolerance.setter
    def stopping_tolerance(self, num: int) -> None:
        """Sets the stopping tolerance"""
        self._stopping_tolerance = num

    @property
    def pauli_op_dict(self) -> Dict:
        """Returns the dictonary containing all of the Pauli string operators
            necessary for calculating the RDMs."""
        return self._pauli_op_dict

    @pauli_op_dict.setter
    def pauli_op_dict(self, some_dict: Dict) -> None:
        """Sets the dictionary containing all of the Pauli string operators
            necessary for calculating the RDMs."""
        self._pauli_op_dict = some_dict

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

    def is_2body_op_spin_conserving(self, P,Q,R,S) -> bool:

        """Determines whether or not the two body fermionic excitation operator
            involved in the 2-RDM element indexed by (P,Q,R,S) conserves spin or not.
        
        Args:
        
            P,Q,R,S: the RDM element index.

        Returns:
            True if fermionic operator conserves spin, False if it does not conserve spin.
            
        """

        N = self.num_spin_orbitals

        spin_change = 0
        if 0 <= P <= N/2 - 1:
            spin_change += 1
        else:
            spin_change -= 1

        if 0 <= Q <= N/2 - 1:
            spin_change += 1
        else:
            spin_change -= 1

        if 0 <= R <= N/2 - 1:
            spin_change -= 1
        else:
            spin_change += 1

        if 0 <= S <= N/2 - 1:
            spin_change -= 1
        else:
            spin_change += 1

        if spin_change == 0:
                return True
        else:
            return False

    def is_1body_op_spin_conserving(self, P,Q) -> bool:

        """Determines whether or not the one body fermionic excitation operator
            involved in the 1-RDM element indexed by (P,Q) conserves spin or not.
        
        Args:
            
            P,Q: The index of the 1-RDM.

        Returns:
            True if the fermionic operator conserves spin, False if it does not conserve
                spin.
        
        """

        N = self.num_spin_orbitals
            
        spin_change = 0
        if 0 <= P <= N/2 - 1:
            spin_change += 1
        else:
            spin_change -= 1

        if 0 <= Q <= N/2 - 1:
            spin_change -= 1
        else:
            spin_change += 1

        if spin_change == 0:
            return True
        else:
            return False


    def construct_pauli_op_dict(self, qubit_converter: QubitConverter) -> Dict:

        """Constructs a dictionary of all the Pauli string operators necessary for computing the RDMs.
            The dictionary key/value pairs are of the form str(pauli_op): pauli_op. The uniqueness of
            python dictionary keys ensures that no redundant operator evaluations are done.
            
        Args:
            state: The state with respect to which the RDMs are being calculated.
            qubit_converter: The QubitConverter used to map the fermionic excitation
                operators to qubit operators.

        Returns:

            The dictionary consisting of all the Pauli string operators necessary for calculating
                The RDMs.

        """

        N = self.num_spin_orbitals
        pauli_op_dict = {}
    
        def oneRDM_add_pauli_ops_to_set(p,q):

            op = qubit_converter.convert(FermionicOp(data=f'+_{p} -_{q}',
                display_format='sparse',
                register_length=N)).reduce()

            if op.is_hermitian() == True:

                pauli_op_dict[str(op)] = op

            else:

                pauli_string_list = op.primitive.to_list()
                for op_tuple in pauli_string_list:
                    pauli_op_dict[str(op_tuple[0])] = PauliSumOp(SparsePauliOp(op_tuple[0]))

            return None

        def twoRDM_add_pauli_ops_to_set(p,q,r,s):

            op = qubit_converter.convert(FermionicOp(data=f'+_{p} +_{q} -_{s} -_{r}',
                display_format='sparse',
                register_length=N)).reduce()

            if op.is_hermitian() == True:

                pauli_op_dict[str(op)] = op

            else:

                pauli_string_list = op.primitive.to_list()
                for op_tuple in pauli_string_list:
                    pauli_op_dict[str(op_tuple[0])] = PauliSumOp(SparsePauliOp(op_tuple[0]))

            return None

  

        # if element is True, then we still need to generate the pauli operators relevant to this fermionic operator.
        # if element is False, then we do not need to generate any additional pauli operators for this fermionic operator.
        # needs_evaluation_array keeps track of this to avoid redundancy.
        oneRDM_needs_evaluation_array = np.full((N,N), fill_value=True, dtype=bool)
        twoRDM_needs_evaluation_array = np.full((N,N,N,N),fill_value=True, dtype=bool)

        for p in range(N):
            for q in range(N):
                for r in range(N):
                        for s in range(N):
                                    
                            if twoRDM_needs_evaluation_array[p,q,r,s] == True:

                                if p == q or r == s or (self.spin_conserving == True and self.is_2body_op_spin_conserving(p,q,r,s) == False):
                                        
                                        # we do not need to evaluate these operators as these entries will be zero in the 2-RDM.
                                    twoRDM_needs_evaluation_array[p,q,r,s] = False

                                else:
                                        
                                        # we add the relevant pauli ops to the set, then we set this entry to False
                                        # so we can set other elements to False to take advantage of symmetries in the 2RDM
                                        # to avoid redundant evaluations.
                                    twoRDM_add_pauli_ops_to_set(p,q,r,s)
                                    twoRDM_needs_evaluation_array[p,q,r,s] = False
                                        
                                twoRDM_needs_evaluation_array[q,p,r,s] = twoRDM_needs_evaluation_array[p,q,r,s]
                                twoRDM_needs_evaluation_array[p,q,s,r] = twoRDM_needs_evaluation_array[p,q,r,s]
                                twoRDM_needs_evaluation_array[q,p,s,r] = twoRDM_needs_evaluation_array[p,q,r,s]

                                    

                                twoRDM_needs_evaluation_array[r,s,p,q] = twoRDM_needs_evaluation_array[p,q,r,s]
                                twoRDM_needs_evaluation_array[r,s,q,p] = twoRDM_needs_evaluation_array[q,p,r,s]
                                twoRDM_needs_evaluation_array[s,r,p,q] = twoRDM_needs_evaluation_array[p,q,s,r]
                                twoRDM_needs_evaluation_array[s,r,q,p] = twoRDM_needs_evaluation_array[q,p,s,r]

        for p in range(N):
            for q in range(N):
                    
                if oneRDM_needs_evaluation_array[p,q] == True:

                    if self.spin_conserving == True and self.is_1body_op_spin_conserving(p,q) == False:

                            # we don't need the pauli operators from these terms because
                            # they are just zero in the 1-RDM
                        oneRDM_needs_evaluation_array[p,q] = False

                    else:
                    
                        oneRDM_add_pauli_ops_to_set(p,q)
                        oneRDM_needs_evaluation_array[p,q] = False
                        oneRDM_needs_evaluation_array[q,p] = oneRDM_needs_evaluation_array[p,q]

        return pauli_op_dict

    def get_two_RDM_tensor(self, qubit_converter: QubitConverter) -> torch.Tensor:

        """Constructs and returns the 2-RDM tensor. The class attribute pauli_ops_expectation_values_dict stores the expectation values
            of all the Pauli operators necessary for this calculation. get_two_RDM_tensor simply retrieves these expectation values
            and constructs the 2-RDM.
        
        Args:

            state: The state with respect to which the 2-RDM is being calculated.
            qubit_converter: The QubitConverter used for mapping fermionic operators to qubit operators.

        Returns:
            The 2-RDM with respect to the given state.        
        
        """
        
        #N = state.num_qubits # Associating this number with the number of qubits could fail if symmetry reductions for some mappings is used.
        N = self.num_spin_orbitals                    # This should change if this is something we want to do eventually.
        global two_RDM_found_complex_value_flag
        two_RDM_found_complex_value_flag = False
        def get_two_RDM_element(p: int,q: int,r: int,s: int, qubit_converter: QubitConverter):

            global two_RDM_found_complex_value_flag

            op = qubit_converter.convert(FermionicOp(data=f'+_{p} +_{q} -_{s} -_{r}',
                display_format='sparse',
                register_length=N)).reduce()
            
            if op.is_hermitian() == True:
                
                    mean = self._pauli_ops_expectation_values_dict[str(op)]
            else:
                    pauli_string_list = op.primitive.to_list()
                    mean = 0
                    for op_tuple in pauli_string_list:
                        
                        mean += op_tuple[1]*self._pauli_ops_expectation_values_dict[str(op_tuple[0])]
            
            if not np.isclose(np.imag(mean), 0.0, rtol=10e-12, atol=10e-12):
                
                two_RDM_found_complex_value_flag = True

            if self._wavefunction_real == True:
                return np.real(mean)
            else:
                return mean

        if self._wavefunction_real == True:
            tensor = np.full(shape=(N,N,N,N), fill_value=None, dtype=np.float64)
        else:
            tensor = np.full(shape=(N,N,N,N), fill_value=None, dtype=np.complex128)
        
        for p in range(N):
                for q in range(N):
                        for r in range(N):
                                for s in range(N):
                                    
                                    if np.isnan(tensor[p,q,r,s]):

                                        if p == q or r == s or (self._spin_conserving == True and self.is_2body_op_spin_conserving(p,q,r,s) == False):

                                            tensor[p,q,r,s] = 0

                                        else:

                                            tensor[p,q,r,s] = get_two_RDM_element(p=p,q=q,r=r,s=s, qubit_converter=qubit_converter)
                                        
                                        tensor[q,p,r,s] = -1*tensor[p,q,r,s]
                                        tensor[p,q,s,r] = -1*tensor[p,q,r,s]
                                        tensor[q,p,s,r] = tensor[p,q,r,s]

                                        if self._wavefunction_real == True:
                                            
                                            tensor[r,s,p,q] = tensor[p,q,r,s]
                                            tensor[r,s,q,p] = tensor[q,p,r,s]
                                            tensor[s,r,p,q] = tensor[p,q,s,r]
                                            tensor[s,r,q,p] = tensor[q,p,s,r]

                                        elif self._wavefunction_real == False:

                                            tensor[r,s,p,q] = np.conj(tensor[p,q,r,s])
                                            tensor[r,s,q,p] = np.conj(tensor[q,p,r,s])
                                            tensor[s,r,p,q] = np.conj(tensor[p,q,s,r])
                                            tensor[s,r,q,p] = np.conj(tensor[q,p,s,r])

        if two_RDM_found_complex_value_flag == False:

            tensor = np.real(tensor)
            print('2RDM is real')

        tensor = torch.from_numpy(tensor)
        tensor.requires_grad = False

        return tensor

    def get_one_RDM_tensor(self, qubit_converter: QubitConverter) -> torch.Tensor:
        
        """Constructs and returns the 1-RDM tensor. The class attribute pauli_ops_expectation_values_dict stores the expectation values
            of all the Pauli operators necessary for this calculation. get_one_RDM_tensor simply retrieves these expectation values
            as needed and constructs the 1-RDM.

        Args:
            state: The state with respect to which the 1-RDM is being calculated.
            qubit_converter: The QubitConverter used to map fermionic operators to qubit operators.

        Returns:
            The 1-RDM tensor.
        
        """

        #N = state.num_qubits
        global one_RDM_found_complex_value_flag
        one_RDM_found_complex_value_flag = False
        N = self.num_spin_orbitals
        def get_one_RDM_element(p: int,q: int, qubit_converter: QubitConverter) -> torch.Tensor:
            
            global one_RDM_found_complex_value_flag

            op = qubit_converter.convert(FermionicOp(data=f'+_{p} -_{q}',
                display_format='sparse',
                register_length=N)).reduce()
            
            if op.is_hermitian() == True:
                
                    mean = self.pauli_ops_expectation_values_dict[str(op)]

            else:
                    pauli_string_list = op.primitive.to_list()
                    mean = 0
                    for op_tuple in pauli_string_list:
                        
                            mean += op_tuple[1]*self._pauli_ops_expectation_values_dict[str(op_tuple[0])]
            
            if not np.isclose(np.imag(mean), 0.0, rtol=10e-12, atol=10e-12):

                one_RDM_found_complex_value_flag = True

            if self._wavefunction_real == True:
                return np.real(mean)
            else:
                return mean

        if self._wavefunction_real == True:
            tensor = np.full(shape=(N,N), fill_value=None, dtype=np.float64)
        else:
            tensor = np.full(shape=(N,N), fill_value=None, dtype=np.complex128)
        
        for p in range(N):
                for q in range(N):
                    
                    if np.isnan(tensor[p,q]):

                        if self._spin_conserving == True and self.is_1body_op_spin_conserving(p,q) == False:

                            tensor[p,q] = 0

                        else:
                    
                            tensor[p,q] = get_one_RDM_element(p=p,q=q, qubit_converter=qubit_converter)

                        if self._wavefunction_real == True:
                            
                            tensor[q,p] = tensor[p,q]
                        
                        else:

                            tensor[q,p] = np.conj(tensor[p,q])

        if one_RDM_found_complex_value_flag == False:
            tensor = np.real(tensor)
            print('1RDM is real')
            
        tensor = torch.from_numpy(tensor)
        tensor.requires_grad = False

        return tensor

    def get_rotated_hamiltonian(self, partial_unitary: torch.Tensor) -> OperatorBase:

        """Transforms the one and two body integrals from the initial larger basis and transforms them according to
            a partial unitary matrix U. The transformed Hamiltonian is then constructed from these new integrals.

        Args:
            partial_unitary: The partial unitary transformation U.

        Returns:
            The transformed Hamiltonian.
        
        """

        rotated_one_body_integrals = torch.einsum('pq,pi,qj->ij',
                                     self._one_body_integrals,
                                     partial_unitary, partial_unitary)
        rotated_two_body_integrals = torch.einsum('pqrs,pi,qj,rk,sl->ijkl',
                                     self._two_body_integrals,
                                     partial_unitary, partial_unitary, partial_unitary, partial_unitary)

        electronic_energy_from_ints = ElectronicEnergy.from_raw_integrals(ElectronicBasis.SO,
                        rotated_one_body_integrals.detach().numpy(),
                        rotated_two_body_integrals.detach().numpy())
        fermionic_op = electronic_energy_from_ints.second_q_ops()[0].reduce().to_normal_order()
        return self._qubit_converter.convert(fermionic_op).reduce()

    def stopping_condition(self) -> bool:

        """Evaluates whether or not the stopping condition is True.
        Returns True if the algorithm should be stopped, otherwise returns False.
        """

        if len(self._energy_convergence_list) >= 2:
            if self._iteration == self._maxiter or np.abs(self._energy_convergence_list[-1] - self._energy_convergence_list[-2]) < self._stopping_tolerance:
                return True
            else:
                return False
        
        else:
            return False

    def orth(self, V: torch.Tensor) -> torch.Tensor:
        """
        Generate the orthonormal projection of the matrix V.

        Args:
            V: The matrix to be orthonormalized.
                
        Returns:
            orth(V), the orthogonal projection of the matrix V.
        """
        L, Q = torch.linalg.eigh(torch.t(V) @ V)
        result = V @ Q @ (torch.float_power(torch.inverse(torch.diag(L)), 0.5)) @ torch.t(Q).double()
        return result

    def compute_minimum_energy(self) -> MinimumEigensolverResult:
        
        optorb_result = OptOrbVQEResult()
        self._hamiltonian = self.get_rotated_hamiltonian(self._current_partial_unitary)
        print(f'Operator number of qubits: {self._hamiltonian.num_qubits}')
        self._partial_unitary_optimizer_list[self._iteration].one_body_integral = self._one_body_integrals
        self._partial_unitary_optimizer_list[self._iteration].two_body_integral = self._two_body_integrals
        self._pauli_op_dict = self.construct_pauli_op_dict(qubit_converter=self.qubit_converter)
        print(f'Number of ops: {len(self._pauli_op_dict)}')

        while self.stopping_condition() == False:
            
            result = self._vqe_instance_list[self._iteration].compute_minimum_eigenvalue(operator=self._hamiltonian)
            energy = np.real(result.eigenvalue)
            opt_params = result.optimal_parameters

            # update the optorb result to hold the most recent VQE value.
            optorb_result.eigenvalue = energy

            # update the optorb result to hold the most recent VQE parameters.
            optorb_result.optimal_parameters = opt_params

            # update the optorb result to hold the most recent partial unitary basis transformation.
            optorb_result.optimal_partial_unitary = self._current_partial_unitary
            optorb_result.num_vqe_evaluations += 1
            optorb_result.eigenstate = result.eigenstate
            optorb_result.optimal_point = result.optimal_point
            optorb_result.optimal_value = result.optimal_value

            if self._callback is not None:
                self._callback(self._iteration, result, optorb_result)

            self._energy_convergence_list.append(energy)
            state = copy.deepcopy(self._vqe_instance_list[self._iteration].ansatz).bind_parameters(opt_params)
            
            state = transpile(circuits=state,
                    backend=self._vqe_instance_list[self._iteration].quantum_instance.backend,
                    optimization_level=3)
            circuit_op = CircuitStateFn(state)
            
            if self.stopping_condition() == True:
                break

            print(f'OptOrbVQE iteration: {self._iteration}')
            start_time = perf_counter()
            if self.expectation is not None:
                print(self.expectation)
                self._pauli_ops_expectation_values_dict = {op_string: CircuitSampler(self._vqe_instance_list[self._iteration].quantum_instance,
                        param_qobj=is_aer_provider(self._vqe_instance_list[self._iteration].quantum_instance.backend)).convert(self.expectation.convert(StateFn(self._pauli_op_dict[op_string],
                        is_measurement=True)).compose(circuit_op).reduce()).eval()
                        for op_string in self._pauli_op_dict}
            elif self.expectation is None:
        
                self._pauli_ops_expectation_values_dict = {op_string: CircuitSampler(self._vqe_instance_list[self._iteration].quantum_instance,
                        param_qobj=is_aer_provider(self._vqe_instance_list[self._iteration].quantum_instance.backend)).convert(ExpectationFactory.build(
                operator=StateFn(self._pauli_op_dict[op_string],
                        is_measurement=True),
                backend=self.quantum_instance,
                include_custom=self._vqe_instance_list[self._iteration].include_custom,
                ).convert(StateFn(self._pauli_op_dict[op_string],
                        is_measurement=True)).compose(circuit_op).reduce()).eval()
                        for op_string in self._pauli_op_dict}
            stop_time = perf_counter()
            print(f'Pauli ops evaluation time: {stop_time - start_time} seconds')

            oneRDM = self.get_one_RDM_tensor(qubit_converter=self._qubit_converter)
            twoRDM = self.get_two_RDM_tensor(qubit_converter=self._qubit_converter)

            self._partial_unitary_optimizer_list[self._iteration].oneRDM = oneRDM
            self._partial_unitary_optimizer_list[self._iteration].twoRDM = twoRDM
            self._partial_unitary_optimizer_list[self._iteration].one_body_integral = self._one_body_integrals
            self._partial_unitary_optimizer_list[self._iteration].two_body_integral = self._two_body_integrals
            self._partial_unitary_optimizer_list[self._iteration].current_partial_unitary = self.orth(self._current_partial_unitary + torch.Tensor(np.random.normal(loc=0.0,
                                                scale=0.01, size=(self._current_partial_unitary.size()[0],
                                                self._current_partial_unitary.size()[1]))))
            print('starting orbital opimization')
            self._current_partial_unitary = self._partial_unitary_optimizer_list[self._iteration].compute_optimal_rotation()[0]
            self._hamiltonian = self.get_rotated_hamiltonian(self._current_partial_unitary)
            self._iteration += 1
            self._vqe_instance_list[self._iteration].initial_point = result.optimal_point

            self._partial_unitary_optimizer_list[self._iteration - 1] = None
            self._vqe_instance_list[self._iteration - 1] = None

        return optorb_result

class OptOrbVQEResult(VariationalResult, MinimumEigensolverResult):

    def __init__(self) -> None:
        super().__init__()
        self._num_vqe_evaluations = 0
        self._optimal_partial_unitary = None
    
    @property
    def num_vqe_evaluations(self) -> int:
        """Returns the number of times VQE was run in OptOrbVQE."""
        return self._num_vqe_evaluations

    @num_vqe_evaluations.setter
    def num_vqe_evaluations(self, some_int: int) -> None:
        """Sets the number of times VQE was run in OptOrbVQE."""
        self._num_vqe_evaluations = some_int

    @property
    def optimal_partial_unitary(self) -> torch.Tensor:
        """Returns the optimal partial unitary basis transformation."""
        return self._optimal_partial_unitary

    @optimal_partial_unitary.setter
    def optimal_partial_unitary(self, some_tensor: torch.Tensor) -> None:
        """Sets the optimal partial unitary basis transformation."""
        self._optimal_partial_unitary = some_tensor




    

