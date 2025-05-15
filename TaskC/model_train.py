import numpy as np
from trainingQ import QTraining

def model_train(feature_points, gripperwithu):
    """Process rope and gripper data to train the model.

    Args:
        feature_points (np.ndarray): The positions of the rope feature points.
        gripperwithu (np.ndarray): The gripper points with additional rotation data.

    Returns:
        np.ndarray: The optimized G matrix.
    """
    # Use all data for training
    training_c, training_r = feature_points, gripperwithu

    # Train model
    modeltrain = QTraining(training_c, training_r)
    delta_c, delta_r = modeltrain.calculate_differences()
    optimized_G = modeltrain.optimize_G(delta_c, delta_r)

    return optimized_G,delta_c
