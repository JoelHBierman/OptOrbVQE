import numpy as np
from time import perf_counter
from typing import Optional, Callable, Tuple
import torch
from functools import partial

class PartialUnitaryProjectionOptimizer():

    def __init__(self,
        initial_BBstepsize: float,
        stopping_tolerance: float,
        maxiter: int,
        callback: Optional[Callable] = None,
        decay_factor: int = 0.8,
        gradient_method: Optional[str] = 'autograd',
        device: Optional[str] = 'cpu'
        ) -> None:
        """
            Args:
                initial_BBstepsize: The initial stepsize to be used in the optimization.
                stopping_tolerance: The stopping tolerance that determines when to end the optimization.
                maxiter: The maximum number of optimization iterations.
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
        self._callback = callback
        self.stopping_tolerance = stopping_tolerance
        self.maxiter = maxiter
        self._BBstepsize = initial_BBstepsize
        self.decay_factor = decay_factor
        self.device = device
        self.gradient_method = gradient_method
        
    @property
    def callback(self) -> Callable:
        """Returns the callback function."""
        return self._callback

    @callback.setter
    def callback(self, func: Callable) -> None:
        """Sets the callback function."""
        self._callback = func

    @property
    def BBstepsize(self) -> float:
        """Returns the current BB step size."""
        return self._BBstepsize

    @BBstepsize.setter
    def BBstepsize(self, stepsize: float) -> None:
        """Sets the current BB step size."""
        self._BBstepsize = stepsize

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
    
    def compute_rotated_energy_automatic_gradient(self, partial_unitary: torch.Tensor,
                                                        func: Callable) -> torch.Tensor:

        """
        Calculates the gradient of the energy function for a given partial unitary matrix U using Pytorch automatic differentiation.

        Args:
            partial_unitary: The partial unitary matrix U at which the gradient is being evaluated.

        Returns: The gradient at the point U.
        """

        partial_unitary.requires_grad = True

        rotated_gradient = torch.autograd.grad([func(partial_unitary)], inputs=[partial_unitary])[0]

        partial_unitary.requires_grad = False
        
        return rotated_gradient

    def compute_rotated_energy_gradient(self, partial_unitary: torch.Tensor,
                                              func: Callable) -> torch.Tensor:

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
                rotated_gradient[i,j] = (func(point + temp_tensor) - func(point - temp_tensor))/(2*10**-8)

        return rotated_gradient

    def compute_updated_partial_unitary(self, iteration_number: int,
                                              current_partial_unitary: torch.Tensor,
                                              previous_partial_unitary: torch.Tensor,
                                              current_rotated_energy_gradient: torch.Tensor,
                                              previous_rotated_energy_gradient: torch.Tensor) -> torch.Tensor:

        """
        Computes the updated partial unitary using the stored values of the current and previous unitaries and gradients.

        Returns: The updated partial unitary matrix.
        """

        if iteration_number == 0:
            pass
        if iteration_number % 2 != 0: # iteration number is odd

            delta_U = current_partial_unitary - previous_partial_unitary
            delta_G = current_rotated_energy_gradient - previous_rotated_energy_gradient

            self._BBstepsize = (torch.trace(torch.t(delta_U) @ delta_U))/torch.abs(torch.trace(torch.t(delta_U) @ delta_G))

        if iteration_number % 2 == 0 and iteration_number != 0: # iteration number is even, but not zero.

            delta_U = current_partial_unitary - previous_partial_unitary
            delta_G = current_rotated_energy_gradient - previous_rotated_energy_gradient

            self._BBstepsize = torch.abs(torch.trace(torch.t(delta_U) @ delta_G))/(torch.trace(torch.t(delta_G) @ delta_G))
            
        updated_partial_unitary = self.orth(current_partial_unitary - torch.mul(current_rotated_energy_gradient, self._BBstepsize))
        
        return updated_partial_unitary

    def compute_optimal_rotation(self, fun: Callable,
                                       initial_partial_unitary: torch.Tensor,
                                       oneRDM: torch.Tensor,
                                       twoRDM: torch.Tensor,
                                       one_body_integrals: torch.Tensor,
                                       two_body_integrals: torch.Tensor) -> Tuple[torch.Tensor, float]:

        """
        Performs the optimization with fixed wavefunction and varied partial unitary U in order
            to find the optimal rotation and energy. The first few iterations are written out explicitly
            in order to first fill the arrays Pt_array and St_array with values that are not of type ``None``.
            Then the rest of the optimization is handled in a while loop.

        Returns: A tuple of the form optimal_partial_unitary, optimal_energy.
        """
        objfunc = partial(fun, oneRDM=oneRDM, twoRDM=twoRDM, one_body_integrals=one_body_integrals, two_body_integrals=two_body_integrals)
        P4_array = np.array([None, None, None])
        St_array = np.array([None, 1.5*self.stopping_tolerance])
        current_partial_unitary = initial_partial_unitary.to(self.device)
        previous_partial_unitary = None
        current_rotated_energy_gradient = None 
        previous_rotated_energy_gradient = None

        #two_body_integrals = two_body_integrals.to(self.device)
        #one_body_integrals = one_body_integrals.to(self.device)
        #oneRDM = oneRDM.to(self.device)
        #twoRDM = twoRDM.to(self.device)
        iteration_number = 0

        if self.gradient_method == 'finite_difference':
            gradient_function = self.compute_rotated_energy_gradient
        if self.gradient_method == 'autograd':
            gradient_function = self.compute_rotated_energy_automatic_gradient

        #self._current_partial_unitary = self._current_partial_unitary.to(self.device)
        #self._current_partial_unitary
        P4_array[2] = objfunc(partial_unitary=current_partial_unitary) # calculates f(U_0) and stores it in self._P4_array[2]
        if self._callback is not None:
            self._callback(iteration_number, P4_array[2].item())

        #print(f'Iteration: {self._iteration_number}, energy: {self._P4_array[2]}')
        #self._current_rotated_energy_gradient = self.compute_rotated_energy_gradient(self._current_partial_unitary) # computes gradf(U_0) and stores it in self._current_rotated_energy_gradient
        current_rotated_energy_gradient = gradient_function(current_partial_unitary, objfunc) # computes gradf(U_0) and stores it in self._current_rotated_energy_gradient

        new_partial_unitary = self.compute_updated_partial_unitary(iteration_number=iteration_number,
                                                                   current_partial_unitary=current_partial_unitary,
                                                                   previous_partial_unitary=previous_partial_unitary,
                                                                   current_rotated_energy_gradient=current_rotated_energy_gradient,
                                                                   previous_rotated_energy_gradient=previous_rotated_energy_gradient) # calculates U_1 (from U_0 and gradf(U_0)) and stores it in temporary variable
        #new_rotated_energy_gradient = self.compute_rotated_energy_gradient(new_partial_unitary) # calculates gradf(U_1)
        new_rotated_energy_gradient = gradient_function(new_partial_unitary, objfunc) # calculates gradf(U_1)

        previous_partial_unitary = current_partial_unitary # copies U_0 into self._previous_partial_unitary for k=1 iteration
        previous_rotated_energy_gradient = current_rotated_energy_gradient # copies gradf(U_0) into self._previous_rotated_energy_gradient for k=1 iteration
        current_partial_unitary = new_partial_unitary # sets the value of self._current_partial_unitary to U_1 for k=1 iteration
        current_rotated_energy_gradient = new_rotated_energy_gradient # sets the value of self._current_rotated_energy_gradient to gradf(U_1) for k=1 iteration
        iteration_number += 1 # updates iteration number to 1


        P4_array[1] = objfunc(partial_unitary=current_partial_unitary) # calculates f(U_1) and stores it in self._P4_array[1].
        if self._callback is not None:
            self._callback(iteration_number, P4_array[1].item())

        #print(f'Iteration: {self._iteration_number}, energy: {self._P4_array[1]}')
        St_array[0] = (1 - self.decay_factor)*torch.abs(P4_array[1] - P4_array[2]) + self.decay_factor*St_array[1] # calculate S_1 and store it in self._St_array[0]

        new_partial_unitary = self.compute_updated_partial_unitary(iteration_number=iteration_number,
                                                                   current_partial_unitary=current_partial_unitary,
                                                                   previous_partial_unitary=previous_partial_unitary,
                                                                   current_rotated_energy_gradient=current_rotated_energy_gradient,
                                                                   previous_rotated_energy_gradient=previous_rotated_energy_gradient) # calculates U_2 (from U_1 and gradf(U_1)) and stores it in temporary variable
        #new_rotated_energy_gradient = self.compute_rotated_energy_gradient(new_partial_unitary) # calculates gradf(U_1) and stores it in temporary variable
        new_rotated_energy_gradient = gradient_function(new_partial_unitary, objfunc) # calculates gradf(U_1) and stores it in temporary variable

        previous_partial_unitary = current_partial_unitary # copies U_1 into self._previous_partial_unitary for k=2 iteration
        previous_rotated_energy_gradient = current_rotated_energy_gradient # copies gradf(U_1) into self._previous_rotated_energy_gradient for k=2 iteration
        current_partial_unitary = new_partial_unitary # override value of self._current_partial_unitary to make it U_2
        current_rotated_energy_gradient = new_rotated_energy_gradient # override value of self._current_rotated_energy_gradient to make it gradf(U_2)
        iteration_number += 1 # update iteration number to make it 2

        
        P4_array[0] = objfunc(partial_unitary=current_partial_unitary) # calculates and stores f(U_2) in P4_array[0].
        if self._callback is not None:
            self._callback(iteration_number, P4_array[0].item())

        #print(f'Iteration: {self._iteration_number}, energy: {self._P4_array[0]}')
        St_array = np.roll(St_array, shift = 1) # roll self._St_array so that S_1 is stored in self._St_array[1]
        St_array[0] = (1 - self.decay_factor)*torch.abs(P4_array[0] - P4_array[1]) + self.decay_factor*St_array[1] # calculate S_2 and store it in self._St_array[0]

        new_partial_unitary = self.compute_updated_partial_unitary(iteration_number=iteration_number,
                                                                   current_partial_unitary=current_partial_unitary,
                                                                   previous_partial_unitary=previous_partial_unitary,
                                                                   current_rotated_energy_gradient=current_rotated_energy_gradient,
                                                                   previous_rotated_energy_gradient=previous_rotated_energy_gradient) # calculates U_3 (from U_2 and gradf(U_2)) and stores it in temporary variable
        #new_rotated_energy_gradient = self.compute_rotated_energy_gradient(new_partial_unitary) # calculates gradf(U_2) and stores it in temporary variable
        new_rotated_energy_gradient = gradient_function(new_partial_unitary, objfunc) # calculates gradf(U_2) and stores it in temporary variable

        previous_partial_unitary = current_partial_unitary # copies U_2 into self._previous_partial_unitary for k=3 iteration
        previous_rotated_energy_gradient = current_rotated_energy_gradient # copies gradf(U_2) into self._previous_rotated_energy_gradient for k=3 iteration
        current_partial_unitary = new_partial_unitary # override value of self._current_partial_unitary to make it U_3
        current_rotated_energy_gradient = new_rotated_energy_gradient # override value of self._current_rotated_energy_gradient to make it gradf(U_3)
        iteration_number += 1 # update iteration number to make it 3

        # if everything is done properly:
        # self._St_array = [S_2, S_1]
        # self._P4_array = [f(U_2), f(U_1), f(U_0)]
        # self._current_partial_unitary = U_3
        # self._previous_partial_unitary = U_2
        # self._current_rotated_energy_gradient = gradf(U_3)
        # self._previous_rotated_energy_gradient = gradf(U_2)
        # everything should now be properly initialized to proceed with general iteration k loop

        while St_array[0] > self.stopping_tolerance and iteration_number <= self.maxiter:
            
            P4_array = np.roll(P4_array, shift = 1) # roll self._P4_array so that f(U_k-1) is stored in self._P4_array[1] and f(U_k-2) is stored in self._P4_array[2]
            P4_array[0] = objfunc(partial_unitary=current_partial_unitary) # set self._P4_array[0] to f(U_k)

            if self._callback is not None:
                self._callback(iteration_number, P4_array[1].item())

            #print(f'Iteration: {self._iteration_number}, energy: {self._P4_array[0]}')
            # self._P4_array should now be [f(U_k), f(U_k-1), f(U_k-2)]
            St_array = np.roll(St_array, shift = 1) # roll self._St_array so that S_k-1 is stored in self._St_array[0]
            St_array[0] = (1 - self.decay_factor)*torch.abs(P4_array[1] - P4_array[2]) + self.decay_factor*St_array[1] # set self._St_array[0] to S_k
            # self._St_array should now be [S_k, S_k-1]

            new_partial_unitary = self.compute_updated_partial_unitary(iteration_number=iteration_number,
                                                                       current_partial_unitary=current_partial_unitary,
                                                                       previous_partial_unitary=previous_partial_unitary,
                                                                       current_rotated_energy_gradient=current_rotated_energy_gradient,
                                                                       previous_rotated_energy_gradient=previous_rotated_energy_gradient) # calculates U_k+1 (from U_k and gradf(U_k)) and stores it in temporary variable
            #new_rotated_energy_gradient = self.compute_rotated_energy_gradient(new_partial_unitary) # calculates gradf(U_k+1) and stores it in temporary variable
            new_rotated_energy_gradient = gradient_function(new_partial_unitary, objfunc) # calculates gradf(U_k+1) and stores it in temporary variable

            previous_partial_unitary = current_partial_unitary # copies U_k into self._previous_partial_unitary for k=k+1 iteration
            previous_rotated_energy_gradient = current_rotated_energy_gradient # copies gradf(U_k) into self._previous_rotated_energy_gradient for k=k+1 iteration
            current_partial_unitary = new_partial_unitary # override value of self._current_partial_unitary to make it U_K+1
            current_rotated_energy_gradient = new_rotated_energy_gradient # override value of self._current_rotated_energy_gradient to make it gradf(U_K+1)
            iteration_number += 1 # update iteration number to make it k+1

        current_partial_unitary = current_partial_unitary.to(torch.device('cpu'))
        #two_body_integrals = two_body_integrals.to(torch.device('cpu'))
        #one_body_integrals = one_body_integrals.to(torch.device('cpu'))
        #oneRDM = oneRDM.to(torch.device('cpu'))
        #twoRDM = twoRDM.to(torch.device('cpu'))

        return current_partial_unitary, P4_array[0]
            