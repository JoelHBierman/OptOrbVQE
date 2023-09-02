import numpy as np
from time import perf_counter
from typing import Optional, Callable, Tuple
import torch
#from pympler import asizeof

class PartialUnitaryProjectionOptimizer():

    def __init__(self,
        initial_BBstepsize: float,
        stopping_tolerance: float,
        maxiter: int,
        initial_partial_unitary: Optional[torch.Tensor] = None,
        one_body_integral: Optional[torch.Tensor] = None,
        two_body_integral: Optional[torch.Tensor] = None,
        oneRDM: Optional[torch.Tensor] = None,
        twoRDM: Optional[torch.Tensor] = None,
        callback: Optional[Callable] = None,
        decay_factor: int = 0.8,
        wavefunction_real: bool = True,
        gradient_method: Optional[str] = 'autograd',
        device: Optional[str] = 'cpu'
        ) -> None:
        """
            Args:
                initial_BBstepsize: The initial stepsize to be used in the optimization.
                stopping_tolerance: The stopping tolerance that determines when to end the optimization.
                maxiter: The maximum number of optimization iterations.
                initial_partial_unitary: The initial partial unitary matrix to use in the optimization.
                one_body_integral: The (N,N) matrix of one body integrals used to calculate the energy functional.
                two_body_integral: The (N,N,N,N) tensor of two body integrals used to calculate the energy functional.
                oneRDM: one body reduced density matrix used to calculate the energy functional.
                twoRDM: two body reduced density matrix used to calculate the energy functional.
                    (note: could refactor code later on to remove integrals and RDMs from this class object, but is not necessary for now.)
                callback: A callback function to track the progress of the optimization. This callback takes the iteration number and the energy
                    as arguments.
                decay_factor: The decay factor used in constructing the stopping condition:
                     S_t = (1 - decay_factor)*abs(delta E_t) + decay_factor*S_t-1
                wavefunction_real: A boolean that flags whether or not we can assume the RDM elements to be real.
                gradient_method: One of "finite_difference", "autograd". Setting to "finite_difference" tells the optimization
                    to use a finite difference approximation to calculate the gradient. Setting to "autograd" uses
                    automatic differentiation functionality from Pytorch, which is much faster but can possibly use 
                    more memory.
                device: Which device type to run the optimization on. e.g. setting to 'cpu' runs the optimization
                    on the CPU. Setting to 'cuda:n' runs one the nth GPU in a GPU array.
                    See https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device for complete documentation.
        """
        self._current_partial_unitary = initial_partial_unitary
        self._previous_partial_unitary = None
        self._current_rotated_energy_gradient = None 
        self._previous_rotated_energy_gradient = None
        self._oneRDM = oneRDM
        self._twoRDM = twoRDM
        self._one_body_integral = one_body_integral
        self._two_body_integral = two_body_integral
        self._callback = callback
        self._stopping_tolerance = stopping_tolerance
        self._maxiter = maxiter
        self._iteration_number = 0
        self._BBstepsize = initial_BBstepsize
        self._decay_factor = decay_factor
        self._P4_array = np.array([None, None, None])
        self._St_array = np.array([None, 1.5*self._stopping_tolerance])
        self._wavefunction_real = wavefunction_real
        self._device = device
        self._gradient_method = gradient_method
        self._gradient_function = None
        
    @property
    def wavefunction_real(self) -> bool:
        return self._wavefunction_real

    @wavefunction_real.setter
    def wavefunction_real(self, wavefunc_real: bool) -> None:
        self._wavefunction_real = wavefunc_real

    @property
    def current_partial_unitary(self) -> torch.Tensor:
        """Returns the partial unitary at the current optimization iteration"""
        return self._current_partial_unitary

    @current_partial_unitary.setter
    def current_partial_unitary(self, unitary: torch.Tensor) -> None:
        """Sets the partial unitary at the current optimization iteration"""
        self._current_partial_unitary = unitary

    @property
    def previous_partial_unitary(self) -> torch.Tensor:
        """Returns the partial unitary at the previous optimization iteration.
            Storing this is necessary for calculating the BB step size."""
        return self._previous_partial_unitary

    @previous_partial_unitary.setter
    def previous_partial_unitary(self, unitary: torch.Tensor) -> None:
        """Sets the previous partial unitary."""
        self._previous_partial_unitary = unitary

    @property
    def rotated_energy_gradient(self) -> torch.Tensor:
        """Returns the gradient of the energy functional at the current optimization iteration."""
        return self._current_rotated_energy_gradient

    @rotated_energy_gradient.setter
    def rotated_energy_gradient(self, rotated_gradient: torch.Tensor) -> None:
        """Sets the gradient of the energy functional at the current optimization iteration."""
        self._current_rotated_energy_gradient = rotated_gradient

    @property
    def previous_rotated_energy_gradient(self) -> torch.Tensor:
        """Returns the gradient of the energy functional at the previous optimization iteration.
            Storing this is necessary for calculating the BB step size."""
        return self._previous_rotated_energy_gradient

    @previous_rotated_energy_gradient.setter
    def previous_rotated_energy_gradient(self, rotated_gradient: torch.Tensor):
        """Sets the gradient of the energy functional at the previous optimization iteration."""
        self._previous_rotated_energy_gradient = rotated_gradient

    @property
    def oneRDM(self) -> torch.Tensor:
        """Returns the one body reduced density matrix."""
        return self._oneRDM

    @oneRDM.setter
    def oneRDM(self, oneRDM: torch.Tensor) -> None:
        """Sets the one body reduced density matrix."""
        self._oneRDM = oneRDM

    @property
    def twoRDM(self) -> torch.Tensor:
        """Returns the two body reduced density matrix."""
        return self._twoRDM

    @twoRDM.setter
    def twoRDM(self, twoRDM: torch.Tensor) -> None:
        """Sets the two body reduced density matrix."""
        self._twoRDM = twoRDM

    @property
    def one_body_integral(self) -> torch.Tensor:
        """Returns the one body integral."""
        return self._one_body_integral

    @one_body_integral.setter
    def one_body_integral(self, one_body_integral: torch.Tensor):
        """Sets the one body integral."""
        self._one_body_integral = one_body_integral

    @property
    def two_body_integral(self) -> torch.Tensor:
        """Returns the two body integral."""
        return self._two_body_integral

    @two_body_integral.setter
    def two_body_integral(self, two_body_integral: torch.Tensor) -> None:
        """Sets the two body integral."""
        self._two_body_integral = two_body_integral

    @property
    def callback(self) -> Callable:
        """Returns the callback function."""
        return self._callback

    @callback.setter
    def callback(self, func: Callable) -> None:
        """Sets the callback function."""
        self._callback = func

    @property
    def stopping_tolerance(self) -> float:
        """Returns the stopping tolerance."""
        return self._stopping_tolerance

    @stopping_tolerance.setter
    def stopping_tolerance(self, tol: float) -> None:
        """Sets the stopping tolerance."""
        self._stopping_tolerance = tol

    @property
    def maxiter(self) -> int:
        """Returns the maximum number of optimization iterations."""
        return self._maxiter

    @maxiter.setter
    def maxiter(self, maxiter: int) -> None:
        """Sets the maximum number of optimization iterations."""
        self._maxiter = maxiter

    @property
    def iteration_number(self) -> int:
        """Returns the current optimization iteration."""
        return self._iteration_number

    @iteration_number.setter
    def iteration_number(self, num: int) -> None:
        """Sets the current optimization iteration."""
        self._iteration_number = num

    @property
    def BBstepsize(self) -> float:
        """Returns the current BB step size."""
        return self._BBstepsize

    @BBstepsize.setter
    def BBstepsize(self, stepsize: float) -> None:
        """Sets the current BB step size."""
        self._BBstepsize = stepsize

    @property
    def P4_array(self) -> np.ndarray:
        """ Returns P4_array, an array of three numbers which stores the three most recent energy
            functional values in order to calculate S_t."""
        return self._P4_array

    @P4_array.setter
    def P4_array(self, P4_array: np.ndarray) -> None:
        """Sets P4_array."""
        self._P4_array = P4_array

    @property
    def St_array(self) -> np.ndarray:
        """Returns St_array, an array of two numbers which stores the two most recent values of S_t
            in order to determine whether or not the optimization should stop."""
        return self._St_array

    @St_array.setter
    def St_array(self, some_array: np.ndarray) -> None:
        """Returns St_array."""
        self._St_array = some_array

    @property
    def device(self) -> str:
        """Returns the device on which the optimization will be run."""
        return self._device

    @device.setter
    def device(self, some_device: str) -> None:
        """Sets the device on which the optimization will be run."""
        self._device = some_device

    @property
    def gradient_method(self) -> str:
        """Returns the method of calculating the energy functional gradient."""
        return self._gradient_method

    @gradient_method.setter
    def gradient_method(self, some_str: str) -> None:
        """Sets the method of calculating the energy functional gradient."""
        self._gradient_method = some_str

    @property
    def gradient_function(self) -> Callable:
        """Returns the function used to calculate the energy functional gradient."""
        return self._gradient_function

    @gradient_function.setter
    def gradient_function(self, some_callable: Callable) -> None:
        """Sets the function used to calculate the energy functional gradient."""
        self._gradient_function = some_callable

    def orth(self, V: torch.Tensor) -> torch.Tensor:
        """
        Generate the orthonormal projection of the matrix V.

        Args:
            V: The matrix to be orthonormalized.
                
        Returns:
            orth(V), the orthogonal projection of the matrix V.
        """
        L, Q = torch.linalg.eigh(torch.t(V) @ V)
        #result = torch.tensor(V @ Q @ (torch.float_power(torch.inverse(torch.diag(L)), 0.5)) @ torch.t(Q).double(), requires_grad=False) #check this later.
        result = V @ Q @ (torch.float_power(torch.inverse(torch.diag(L)), 0.5)) @ torch.t(Q).double()
        return result
    
    def compute_rotated_energy(self, partial_unitary: torch.Tensor) -> float:
        """
        Calculates the energy functional with varied U, but fixed wavefunction.

        Args:
            partial_unitary: The partial unitary matrix U.

        Returns:
            P(U), the energy functional for a given rotation U.
        """

        
        if self._wavefunction_real == True or (self._oneRDM.dtype !=torch.complex128 and self._twoRDM.dtype != torch.complex128):
        
            energy = torch.einsum('pq,pi,qj,ij', self._one_body_integral,
                                          partial_unitary,
                                          partial_unitary,
                                          self._oneRDM)
            energy += torch.einsum('pqrs,pi,qj,rk,sl,ijkl', self._two_body_integral,
                                                       partial_unitary,
                                                       partial_unitary,
                                                       partial_unitary, 
                                                       partial_unitary, 
                                                       self._twoRDM)

        else:
            #self._wavefunction_real == False and (self._oneRDM.dtype !=torch.float64 and self._twoRDM.dtype != torch.float64):

            partial_unitary = partial_unitary.cdouble()
            temp_one_body_integral = self._one_body_integral.cdouble()
            temp_two_body_integral = self._two_body_integral.cdouble()

            energy = torch.einsum('pq,pi,qj,ij', temp_one_body_integral,
                                          partial_unitary,
                                          partial_unitary,
                                          self._oneRDM)
            energy += torch.einsum('pqrs,pi,qj,rk,sl,ijkl', temp_two_body_integral,
                                                       partial_unitary,
                                                       partial_unitary,
                                                       partial_unitary, 
                                                       partial_unitary, 
                                                       self._twoRDM)
        
        return np.real(energy)

    def compute_rotated_energy_automatic_gradient(self, partial_unitary: torch.Tensor) -> torch.Tensor:

        """
        Calculates the gradient of the energy function for a given partial unitary matrix U using Pytorch automatic differentiation.

        Args:
            partial_unitary: The partial unitary matrix U at which the gradient is being evaluated.

        Returns: The gradient at the point U.
        """

        partial_unitary.requires_grad = True

        rotated_gradient = torch.autograd.grad([self.compute_rotated_energy(partial_unitary)], inputs=[partial_unitary])[0]

        partial_unitary.requires_grad = False
        
        return rotated_gradient

    def compute_rotated_energy_gradient(self, partial_unitary: torch.Tensor) -> torch.Tensor:

        """
        Calculates the gradient of the energy functional for a given partial unitary matrix U using a finite difference approximation.

        Args:
            partial_unitary: The partial unitary matrix U at which the gradient is being evaluated.
        
        Returns:
            The gradient at the point U.
        """

        point = partial_unitary
        point_size = point.size()
        rotated_gradient = torch.empty(point_size, dtype=torch.float64, device=self.device)
        for i in range(point_size[0]):
            for j in range(point_size[1]):
                temp_tensor = torch.zeros(point_size[0], point_size[1], dtype=torch.float64, device=self.device)
                temp_tensor[i,j] = 10**-8
                rotated_gradient[i,j] = (self.compute_rotated_energy(point + temp_tensor) - self.compute_rotated_energy(point - temp_tensor))/(2*10**-8)

        return rotated_gradient

    def compute_updated_partial_unitary(self) -> torch.Tensor:

        """
        Computes the updated partial unitary using the stored values of the current and previous unitaries and gradients.

        Returns: The updated partial unitary matrix.
        """

        if self._iteration_number == 0:
            pass
        if self._iteration_number % 2 != 0: # iteration number is odd

            delta_U = self._current_partial_unitary - self._previous_partial_unitary
            delta_G = self._current_rotated_energy_gradient - self._previous_rotated_energy_gradient

            self._BBstepsize = (torch.trace(torch.t(delta_U) @ delta_U))/torch.abs(torch.trace(torch.t(delta_U) @ delta_G))

        if self._iteration_number % 2 == 0 and self._iteration_number != 0: # iteration number is even, but not zero.

            delta_U = self._current_partial_unitary - self._previous_partial_unitary
            delta_G = self._current_rotated_energy_gradient - self._previous_rotated_energy_gradient

            self._BBstepsize = torch.abs(torch.trace(torch.t(delta_U) @ delta_G))/(torch.trace(torch.t(delta_G) @ delta_G))
            
        updated_partial_unitary = self.orth(self._current_partial_unitary - torch.mul(self._current_rotated_energy_gradient, self._BBstepsize))
        
        return updated_partial_unitary

    def compute_optimal_rotation(self) -> Tuple[torch.Tensor, float]:

        """
        Performs the optimization with fixed wavefunction and varied partial unitary U in order
            to find the optimal rotation and energy. The first few iterations are written out explicitly
            in order to first fill the arrays Pt_array and St_array with values that are not of type ``None``.
            Then the rest of the optimization is handled in a while loop.

        Returns: A tuple of the form optimal_partial_unitary, optimal_energy.
        """

        if self.gradient_method == 'finite_difference':
            self.gradient_function = self.compute_rotated_energy_gradient
        if self.gradient_method == 'autograd':
            self.gradient_function = self.compute_rotated_energy_automatic_gradient

        self._current_partial_unitary = self._current_partial_unitary.to(self.device)
        self.two_body_integral = self.two_body_integral.to(self.device)
        self.one_body_integral = self.one_body_integral.to(self.device)
        self.oneRDM = self.oneRDM.to(self.device)
        self.twoRDM = self.twoRDM.to(self.device)

        #self._current_partial_unitary
        self._P4_array[2] = self.compute_rotated_energy(self._current_partial_unitary) # calculates f(U_0) and stores it in self._P4_array[2]
        if self._callback is not None:
            self._callback(self._iteration_number, self._P4_array[2].item())

        #print(f'Iteration: {self._iteration_number}, energy: {self._P4_array[2]}')
        #self._current_rotated_energy_gradient = self.compute_rotated_energy_gradient(self._current_partial_unitary) # computes gradf(U_0) and stores it in self._current_rotated_energy_gradient
        self._current_rotated_energy_gradient = self.gradient_function(self._current_partial_unitary) # computes gradf(U_0) and stores it in self._current_rotated_energy_gradient

        new_partial_unitary = self.compute_updated_partial_unitary() # calculates U_1 (from U_0 and gradf(U_0)) and stores it in temporary variable
        #new_rotated_energy_gradient = self.compute_rotated_energy_gradient(new_partial_unitary) # calculates gradf(U_1)
        new_rotated_energy_gradient = self.gradient_function(new_partial_unitary) # calculates gradf(U_1)

        self._previous_partial_unitary = self._current_partial_unitary # copies U_0 into self._previous_partial_unitary for k=1 iteration
        self._previous_rotated_energy_gradient = self._current_rotated_energy_gradient # copies gradf(U_0) into self._previous_rotated_energy_gradient for k=1 iteration
        self._current_partial_unitary = new_partial_unitary # sets the value of self._current_partial_unitary to U_1 for k=1 iteration
        self._current_rotated_energy_gradient = new_rotated_energy_gradient # sets the value of self._current_rotated_energy_gradient to gradf(U_1) for k=1 iteration
        self._iteration_number += 1 # updates iteration number to 1


        self._P4_array[1] = self.compute_rotated_energy(self._current_partial_unitary) # calculates f(U_1) and stores it in self._P4_array[1].
        if self._callback is not None:
            self._callback(self._iteration_number, self._P4_array[1].item())

        #print(f'Iteration: {self._iteration_number}, energy: {self._P4_array[1]}')
        self._St_array[0] = (1 - self._decay_factor)*torch.abs(self._P4_array[1] - self._P4_array[2]) + self._decay_factor*self._St_array[1] # calculate S_1 and store it in self._St_array[0]

        new_partial_unitary = self.compute_updated_partial_unitary() # calculates U_2 (from U_1 and gradf(U_1)) and stores it in temporary variable
        #new_rotated_energy_gradient = self.compute_rotated_energy_gradient(new_partial_unitary) # calculates gradf(U_1) and stores it in temporary variable
        new_rotated_energy_gradient = self.gradient_function(new_partial_unitary) # calculates gradf(U_1) and stores it in temporary variable

        self._previous_partial_unitary = self._current_partial_unitary # copies U_1 into self._previous_partial_unitary for k=2 iteration
        self._previous_rotated_energy_gradient = self._current_rotated_energy_gradient # copies gradf(U_1) into self._previous_rotated_energy_gradient for k=2 iteration
        self._current_partial_unitary = new_partial_unitary # override value of self._current_partial_unitary to make it U_2
        self._current_rotated_energy_gradient = new_rotated_energy_gradient # override value of self._current_rotated_energy_gradient to make it gradf(U_2)
        self._iteration_number += 1 # update iteration number to make it 2

        
        self._P4_array[0] = self.compute_rotated_energy(self._current_partial_unitary) # calculates and stores f(U_2) in P4_array[0].
        if self._callback is not None:
            self._callback(self._iteration_number, self._P4_array[0].item())

        #print(f'Iteration: {self._iteration_number}, energy: {self._P4_array[0]}')
        self._St_array = np.roll(self._St_array, shift = 1) # roll self._St_array so that S_1 is stored in self._St_array[1]
        self._St_array[0] = (1 - self._decay_factor)*torch.abs(self._P4_array[0] - self._P4_array[1]) + self._decay_factor*self._St_array[1] # calculate S_2 and store it in self._St_array[0]

        new_partial_unitary = self.compute_updated_partial_unitary() # calculates U_3 (from U_2 and gradf(U_2)) and stores it in temporary variable
        #new_rotated_energy_gradient = self.compute_rotated_energy_gradient(new_partial_unitary) # calculates gradf(U_2) and stores it in temporary variable
        new_rotated_energy_gradient = self.gradient_function(new_partial_unitary) # calculates gradf(U_2) and stores it in temporary variable

        self._previous_partial_unitary = self._current_partial_unitary # copies U_2 into self._previous_partial_unitary for k=3 iteration
        self._previous_rotated_energy_gradient = self._current_rotated_energy_gradient # copies gradf(U_2) into self._previous_rotated_energy_gradient for k=3 iteration
        self._current_partial_unitary = new_partial_unitary # override value of self._current_partial_unitary to make it U_3
        self._current_rotated_energy_gradient = new_rotated_energy_gradient # override value of self._current_rotated_energy_gradient to make it gradf(U_3)
        self._iteration_number += 1 # update iteration number to make it 3

        # if everything is done properly:
        # self._St_array = [S_2, S_1]
        # self._P4_array = [f(U_2), f(U_1), f(U_0)]
        # self._current_partial_unitary = U_3
        # self._previous_partial_unitary = U_2
        # self._current_rotated_energy_gradient = gradf(U_3)
        # self._previous_rotated_energy_gradient = gradf(U_2)
        # everything should now be properly initialized to proceed with general iteration k loop

        while self.St_array[0] > self._stopping_tolerance and self._iteration_number <= self._maxiter:
            
            self._P4_array = np.roll(self._P4_array, shift = 1) # roll self._P4_array so that f(U_k-1) is stored in self._P4_array[1] and f(U_k-2) is stored in self._P4_array[2]
            self._P4_array[0] = self.compute_rotated_energy(self._current_partial_unitary) # set self._P4_array[0] to f(U_k)

            if self._callback is not None:
                self._callback(self._iteration_number, self._P4_array[1].item())

            #print(f'Iteration: {self._iteration_number}, energy: {self._P4_array[0]}')
            # self._P4_array should now be [f(U_k), f(U_k-1), f(U_k-2)]
            self._St_array = np.roll(self._St_array, shift = 1) # roll self._St_array so that S_k-1 is stored in self._St_array[0]
            self._St_array[0] = (1 - self._decay_factor)*torch.abs(self._P4_array[1] - self._P4_array[2]) + self._decay_factor*self._St_array[1] # set self._St_array[0] to S_k
            # self._St_array should now be [S_k, S_k-1]

            new_partial_unitary = self.compute_updated_partial_unitary() # calculates U_k+1 (from U_k and gradf(U_k)) and stores it in temporary variable
            #new_rotated_energy_gradient = self.compute_rotated_energy_gradient(new_partial_unitary) # calculates gradf(U_k+1) and stores it in temporary variable
            new_rotated_energy_gradient = self.gradient_function(new_partial_unitary) # calculates gradf(U_k+1) and stores it in temporary variable

            self._previous_partial_unitary = self._current_partial_unitary # copies U_k into self._previous_partial_unitary for k=k+1 iteration
            self._previous_rotated_energy_gradient = self._current_rotated_energy_gradient # copies gradf(U_k) into self._previous_rotated_energy_gradient for k=k+1 iteration
            self._current_partial_unitary = new_partial_unitary # override value of self._current_partial_unitary to make it U_K+1
            self._current_rotated_energy_gradient = new_rotated_energy_gradient # override value of self._current_rotated_energy_gradient to make it gradf(U_K+1)
            self._iteration_number += 1 # update iteration number to make it k+1

        self._current_partial_unitary = self._current_partial_unitary.to(torch.device('cpu'))
        self.two_body_integral = self.two_body_integral.to(torch.device('cpu'))
        self.one_body_integral = self.one_body_integral.to(torch.device('cpu'))
        self.oneRDM = self.oneRDM.to(torch.device('cpu'))
        self.twoRDM = self.twoRDM.to(torch.device('cpu'))

        return self._current_partial_unitary, self._P4_array[0]
            