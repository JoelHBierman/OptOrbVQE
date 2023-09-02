from time import perf_counter
from typing import Callable, Optional, Dict, List
from qiskit import transpile
from qiskit.primitives import BaseEstimator
from .partial_unitary_projection_optimizer import PartialUnitaryProjectionOptimizer
from qiskit.algorithms.variational_algorithm import VariationalResult
from qiskit.opflow import PauliSumOp, OperatorBase
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
from memory_profiler import profile

class BaseOptOrbSolver():

    def __init__(self,
        molecule_driver: Optional[BaseDriver],
        integral_tensors: Optional[tuple[torch.Tensor, torch.Tensor]],
        num_spin_orbitals: int,
        qubit_converter: QubitConverter,
        estimator: BaseEstimator,
        partial_unitary_optimizer: PartialUnitaryProjectionOptimizer,
        initial_partial_unitary: Optional[torch.Tensor] = None,
        maxiter: int = 10,
        stopping_tolerance: float = 10**-5,
        spin_conserving: bool = False,
        wavefuntion_real: bool = False,
        callback: Optional[Callable] = None,
        orbital_rotation_callback_func: Optional[Callable] = None,
        partial_unitary_random_perturbation: Optional[float] = None,
        RDM_ops_batchsize: Optional[int] = None,
        spin_restricted: Optional[bool] = False):
        
        """
        
        Args:
            molecule_driver: The driver from which molecule information such as one and two body integrals is obtained.
            num_qubits: The number of budget qubits for the algorithm to use.
            vqe_instance: An instance of VQE to use for the wavefunction optimization.
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
            vqe_callback_func: An optional Callable which takes an integer (indexing the outer loop iteration) as an argument
                and returns a list of callback functions. We can use this to generate a separate callback function
                for each time we run VQE. This is useful for instance, in cases where we want a callback function that
                appends the VQE convergence information to a list, but we want to keep the data for each outerloop iteration 
                separate and append these data to separate lists. If ``None``, then the VQE callback function used will be vqe_instance.callback.
            orbital_rotation_callback_func: An optional Callable directly analogous to vqe_callback_func. This is a Callable that takes
                an integer index as an argument and returns another Callable. If ``None``, then the orbital rotation callback
                used will be partial_unitary_optimizer.callback.

        """

        self.qubit_converter = qubit_converter

        # generate copies of the PartialUnitaryProjectionOptimizer instance to use for every outerloop iteration.
        self._partial_unitary_optimizer_list = [copy.deepcopy(partial_unitary_optimizer) for n in range(int(maxiter+1))]
        if orbital_rotation_callback_func is not None:
            for n in range(maxiter+1):
                self._partial_unitary_optimizer_list[n].callback = orbital_rotation_callback_func(n)

        if integral_tensors is not None:

            self.one_body_integrals, self.two_body_integrals = integral_tensors
        
        elif molecule_driver is not None:

            initial_molecule = molecule_driver.run()
            self.one_body_integrals = torch.from_numpy(initial_molecule.get_property(ElectronicEnergy).get_electronic_integral(ElectronicBasis.MO, 1).to_spin())
            self.two_body_integrals = torch.from_numpy(initial_molecule.get_property(ElectronicEnergy).get_electronic_integral(ElectronicBasis.MO, 2).to_spin())
            del initial_molecule
        
        self.spin_restricted = spin_restricted
        if initial_partial_unitary is None:
            
            num_original_spin_orbitals = self.one_body_integrals.size()[0]
            num_original_molecular_orbitals = int(num_original_spin_orbitals/2)
            num_molecular_orbitals = int(num_spin_orbitals/2)
            if self.spin_restricted is False:
            
                initial_partial_unitary_guess = torch.zeros(size=(num_original_spin_orbitals, num_spin_orbitals), dtype=torch.float64)

                for n in range(int(num_spin_orbitals/2)):
                    initial_partial_unitary_guess[n,n] = 1.0
                for n in range(int(num_spin_orbitals/2)):
                    initial_partial_unitary_guess[int(num_original_spin_orbitals/2) + n, int(num_spin_orbitals/2) + n] = 1.0

            elif self.spin_restricted is True:

                initial_partial_unitary_guess = torch.zeros(size=(num_original_molecular_orbitals, num_molecular_orbitals), dtype=torch.float64)
                for n in range(int(num_molecular_orbitals)):
                    initial_partial_unitary_guess[n,n] = 1.0

            self.initial_partial_unitary = initial_partial_unitary_guess

        else:
            self.initial_partial_unitary = initial_partial_unitary
        
        self.estimator = estimator
        self.estimator_list = [copy.deepcopy(estimator) for n in range(maxiter)]
        self.maxiter = maxiter
        self.spin_conserving = spin_conserving
        self.wavefunction_real = wavefuntion_real
        self.callback = callback

        self._hamiltonian = None
        self.stopping_tolerance = stopping_tolerance
        self._current_partial_unitary = self.initial_partial_unitary
        self._pauli_op_dict = None
        self.num_spin_orbitals = num_spin_orbitals
        self.partial_unitary_random_perturbation = partial_unitary_random_perturbation
        self.RDM_ops_batchsize = RDM_ops_batchsize
    
    @property
    def partial_unitary_optimizer_list(self) -> List[PartialUnitaryProjectionOptimizer]:
        """Returns the list of partial unitary optimizers used for each outer loop iteration."""
        return self._partial_unitary_optimizer_list

    @partial_unitary_optimizer_list.setter
    def partial_unitary_optimizer_list(self, optimizer_list: List[PartialUnitaryProjectionOptimizer]) -> None:
        """Sets the list of partial unitary optimizers used for each outer loop iteration."""
        self._partial_unitary_optimizer_list = optimizer_list

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
    def pauli_op_dict(self) -> Dict:
        """Returns the dictonary containing all of the Pauli string operators
            necessary for calculating the RDMs."""
        return self._pauli_op_dict

    @pauli_op_dict.setter
    def pauli_op_dict(self, some_dict: Dict) -> None:
        """Sets the dictionary containing all of the Pauli string operators
            necessary for calculating the RDMs."""
        self._pauli_op_dict = some_dict

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

    def get_two_RDM_tensor(self, expectval_dict: dict, qubit_converter: QubitConverter) -> torch.Tensor:

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
                
                    mean = expectval_dict[str(op)]
            else:
                    pauli_string_list = op.primitive.to_list()
                    mean = 0
                    for op_tuple in pauli_string_list:
                        
                        mean += op_tuple[1]*expectval_dict[str(op_tuple[0])]
            
            if not np.isclose(np.imag(mean), 0.0, rtol=10e-12, atol=10e-12):
                
                two_RDM_found_complex_value_flag = True

            if self.wavefunction_real == True:
                return np.real(mean)
            else:
                return mean

        if self.wavefunction_real == True:
            tensor = np.full(shape=(N,N,N,N), fill_value=None, dtype=np.float64)
        else:
            tensor = np.full(shape=(N,N,N,N), fill_value=None, dtype=np.complex128)
        
        for p in range(N):
                for q in range(N):
                        for r in range(N):
                                for s in range(N):
                                    
                                    if np.isnan(tensor[p,q,r,s]):

                                        if p == q or r == s or (self.spin_conserving == True and self.is_2body_op_spin_conserving(p,q,r,s) == False):

                                            tensor[p,q,r,s] = 0

                                        else:

                                            tensor[p,q,r,s] = get_two_RDM_element(p=p,q=q,r=r,s=s, qubit_converter=qubit_converter)
                                        
                                        tensor[q,p,r,s] = -1*tensor[p,q,r,s]
                                        tensor[p,q,s,r] = -1*tensor[p,q,r,s]
                                        tensor[q,p,s,r] = tensor[p,q,r,s]

                                        if self.wavefunction_real == True:
                                            
                                            tensor[r,s,p,q] = tensor[p,q,r,s]
                                            tensor[r,s,q,p] = tensor[q,p,r,s]
                                            tensor[s,r,p,q] = tensor[p,q,s,r]
                                            tensor[s,r,q,p] = tensor[q,p,s,r]

                                        elif self.wavefunction_real == False:

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

    def get_one_RDM_tensor(self, expectval_dict: dict, qubit_converter: QubitConverter) -> torch.Tensor:
        
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
                
                    mean = expectval_dict[str(op)]

            else:
                    pauli_string_list = op.primitive.to_list()
                    mean = 0
                    for op_tuple in pauli_string_list:
                        
                            mean += op_tuple[1]*expectval_dict[str(op_tuple[0])]
            
            if not np.isclose(np.imag(mean), 0.0, rtol=10e-12, atol=10e-12):

                one_RDM_found_complex_value_flag = True

            if self.wavefunction_real == True:
                return np.real(mean)
            else:
                return mean

        if self.wavefunction_real == True:
            tensor = np.full(shape=(N,N), fill_value=None, dtype=np.float64)
        else:
            tensor = np.full(shape=(N,N), fill_value=None, dtype=np.complex128)
        
        for p in range(N):
                for q in range(N):
                    
                    if np.isnan(tensor[p,q]):

                        if self.spin_conserving == True and self.is_1body_op_spin_conserving(p,q) == False:

                            tensor[p,q] = 0

                        else:
                    
                            tensor[p,q] = get_one_RDM_element(p=p,q=q, qubit_converter=qubit_converter)

                        if self.wavefunction_real == True:
                            
                            tensor[q,p] = tensor[p,q]
                        
                        else:

                            tensor[q,p] = np.conj(tensor[p,q])

        if one_RDM_found_complex_value_flag == False:
            tensor = np.real(tensor)
            
        tensor = torch.from_numpy(tensor)
        tensor.requires_grad = False

        return tensor

    def compute_rotated_energy(self, partial_unitary: torch.Tensor,
                                     oneRDM: torch.Tensor,
                                     twoRDM: torch.Tensor,
                                     one_body_integrals: torch.Tensor,
                                     two_body_integrals: torch.Tensor) -> float:
        """
        Calculates the energy functional with varied U, but fixed wavefunction.

        Args:
            partial_unitary: The partial unitary matrix U.

        Returns:
            P(U), the energy functional for a given rotation U.
        """

        if self.spin_restricted is True:

            partial_unitary = torch.block_diag(partial_unitary, partial_unitary)

        if self.wavefunction_real == True or (oneRDM.dtype !=torch.complex128 and twoRDM.dtype != torch.complex128):
        
            energy = torch.einsum('pq,pi,qj,ij', one_body_integrals,
                                          partial_unitary,
                                          partial_unitary,
                                          oneRDM)
            energy += torch.einsum('pqrs,pi,qj,rk,sl,ijkl', two_body_integrals,
                                                       partial_unitary,
                                                       partial_unitary,
                                                       partial_unitary, 
                                                       partial_unitary, 
                                                       twoRDM)

        else:
            #self._wavefunction_real == False and (self._oneRDM.dtype !=torch.float64 and self._twoRDM.dtype != torch.float64):

            partial_unitary = partial_unitary.cdouble()
            temp_one_body_integrals = one_body_integrals.cdouble()
            temp_two_body_integrals = two_body_integrals.cdouble()

            energy = torch.einsum('pq,pi,qj,ij', temp_one_body_integrals,
                                          partial_unitary,
                                          partial_unitary,
                                          oneRDM)
            energy += torch.einsum('pqrs,pi,qj,rk,sl,ijkl', temp_two_body_integrals,
                                                       partial_unitary,
                                                       partial_unitary,
                                                       partial_unitary, 
                                                       partial_unitary, 
                                                       twoRDM)
        
        return np.real(energy)

    def get_rotated_hamiltonian(self, partial_unitary: torch.Tensor) -> OperatorBase:

        """Transforms the one and two body integrals from the initial larger basis and transforms them according to
            a partial unitary matrix U. The transformed Hamiltonian is then constructed from these new integrals.

        Args:
            partial_unitary: The partial unitary transformation U.

        Returns:
            The transformed Hamiltonian.
        
        """

        if self.spin_restricted is True:

            partial_unitary = torch.block_diag(partial_unitary, partial_unitary)
        
        rotated_one_body_integrals = torch.einsum('pq,pi,qj->ij',
                                     self.one_body_integrals,
                                     partial_unitary, partial_unitary)
        rotated_two_body_integrals = torch.einsum('pqrs,pi,qj,rk,sl->ijkl',
                                     self.two_body_integrals,
                                     partial_unitary, partial_unitary, partial_unitary, partial_unitary)

        electronic_energy_from_ints = ElectronicEnergy.from_raw_integrals(ElectronicBasis.SO,
                        rotated_one_body_integrals.detach().numpy(),
                        rotated_two_body_integrals.detach().numpy())
        fermionic_op = electronic_energy_from_ints.second_q_ops()['ElectronicEnergy'].reduce().to_normal_order()
        return self.qubit_converter.convert(fermionic_op).reduce()

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

class BaseOptOrbResult(VariationalResult):

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




    