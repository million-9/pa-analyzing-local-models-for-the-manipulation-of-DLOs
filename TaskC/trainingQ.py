# Class for training-related functionalities
import numpy as np
from scipy.optimize import minimize
from sofa_env.utils.math_helper import euler_to_rotation_matrix, quaternion_to_euler_angles, multiply_quaternions,quaternion_to_euler_angles,conjugate_quaternion

def compute_rotations_between_consecutive_quaternions(quaternions):
    """Computes the relative rotations between consecutive quaternions."""
    n = quaternions.shape[0]
    rotations = []

    for i in range(n - 1):
        q1 = quaternions[i]
        q2 = quaternions[i + 1]

        # Step 1: Compute the relative quaternion
        q1_conj = conjugate_quaternion(q1)
        q_rel = multiply_quaternions(q2, q1_conj)

        # Step 2: Convert the relative quaternion to Euler angles
        euler_angles = quaternion_to_euler_angles(q_rel[::-1])

        # Store the result (you can choose to store quaternions or Euler angles)
        rotations.append(euler_angles[2])

    return np.array(rotations)

class QTraining:
    def __init__(self, feature_points, gripper_points):
        """
        Initialize the Training class with feature points and gripper points.

        Parameters:
        feature_points (numpy.ndarray): A 3D array of shape (time, mass_points, coordinates).
        gripper_points (numpy.ndarray): A 3D array of shape (time, 2, coordinates).
        """
        self.feature_points = feature_points  # Store feature points
        self.gripper_points = gripper_points    # Store gripper points

    def calculate_differences(self):
        """
        Calculate the differences between consecutive feature points and gripper points.

        Returns:
        tuple: A tuple containing:
            - A 3D array of shape (time-1, mass_points, coordinates) for feature differences.
            - A 3D array of shape (time-1, 2, coordinates) for gripper differences.
        """
        feature_differences = np.diff(self.feature_points, axis=0)  # Calculate differences along axis 0
        gripper_differences = np.diff(self.gripper_points[:,:,:3], axis=0) # Calculate differences along axis 0

        print(self.gripper_points[:,0,3:].shape)

        rotation_differences = compute_rotations_between_consecutive_quaternions(self.gripper_points[:,0,3:]).reshape(gripper_differences.shape[0],1,1)
        print(rotation_differences.shape)

        gripper_differences= np.concatenate((gripper_differences, rotation_differences), axis=2)

        return feature_differences, gripper_differences

    def objective_function(self, G_flat, delta_c, delta_r):
        """
        Define the objective function for optimization.

        Parameters:
        G_flat (numpy.ndarray): Flattened matrix G.
        delta_c (numpy.ndarray): Differences in feature points.
        delta_r (numpy.ndarray): Differences in gripper points.

        Returns:
        float: The computed cost (sum of squared residuals).
        """
        G = G_flat.reshape((delta_c.shape[1], delta_r.shape[1]))  # Reshape the flattened G
        # Assuming the calculation involves some model prediction using G
        # Here you would replace the following line with the actual computation
        predicted_delta_r = delta_c @ G  # This is just a placeholder

        # Return the sum of squared residuals
        return np.sum((predicted_delta_r - delta_r) ** 2)

    def optimize_G(self, delta_cn, delta_rn):
        """
        Optimize the G matrix to minimize the objective function.

        Parameters:
        delta_cn (numpy.ndarray): Differences in feature points.
        delta_rn (numpy.ndarray): Differences in gripper points.

        Returns:
        numpy.ndarray: The optimized G matrix.
        """
        delta_cn = delta_cn.reshape(-1, delta_cn.shape[1] * delta_cn.shape[2])
        delta_rn = delta_rn.reshape(-1, delta_rn.shape[1] * delta_rn.shape[2])

        # Set a random seed for reproducibility
        np.random.seed(42)
        # Initialize random values for G
        G_init = np.random.rand(delta_cn.shape[1], delta_rn.shape[1])
        # Flatten G
        G_flat_init = G_init.flatten()

        # Use scipy's minimize to find the best value for G
        result = minimize(self.objective_function, G_flat_init, args=(delta_cn, delta_rn), method='l-bfgs-b')

        # Store the result we obtained to matrix G
        G_optimized = result.x.reshape((delta_cn.shape[1], delta_rn.shape[1]))
        return G_optimized