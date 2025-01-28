import numpy as np
import pandas as pd


class BAM:
    def __init__(self, input_size, output_size):
        """
        Initialize the Enhanced BAM with given input and output sizes.
        :param input_size: Number of input neurons.
        :param output_size: Number of output neurons.
        """
        self.weights = np.zeros((input_size, output_size))

    def train(self, inputs, outputs):
        """
        Train the BAM network using pairs of binary input and output vectors.
        :param inputs: List of binary input vectors (lists of 0 and 1).
        :param outputs: List of binary output vectors (lists of 0 and 1).
        """
        # Convert inputs and outputs to numpy arrays.
        inputs = np.array(inputs)
        outputs = np.array(outputs)

        # Update the weights using standard matrix multiplication.
        for x, y in zip(inputs, outputs):
            x_transformed = 2 * x - 1  # Transform input to values between -1 and 1.
            y_transformed = 2 * y - 1  # Transform output to values between -1 and 1.
            self.weights += x_transformed[:, np.newaxis] @ y_transformed[np.newaxis, :]  # Standard outer product.

    def classify(self, input_vector, max_iterations=10000):
        """
        Classify an input vector by iterating until convergence.
        :param input_vector: Binary input vector (list of 0 and 1).
        :param max_iterations: Maximum number of iterations for the convergence loop.
        :return: Stable output vector (list of 0 and 1).
        """
        input_vector = 2 * np.array(input_vector) - 1
        output_vector = np.sign(np.dot(input_vector, self.weights))
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            new_input = np.sign(np.dot(output_vector, self.weights.T))
            new_output = np.sign(np.dot(new_input, self.weights))
            if np.array_equal(input_vector, new_input) and np.array_equal(output_vector, new_output):
                break
            input_vector = new_input
            output_vector = new_output

        output_vector = np.where(output_vector == -1, 0, output_vector)
        return output_vector

    def recall_backward(self, output_pattern):
        """
        Perform a backward recall (output to input).
        :param output_pattern: Output pattern to recall.
        :return: Recalled input pattern.
        """
        y_transformed = 2 * output_pattern - 1  # Transform output to values between -1 and 1.
        input_hat = np.sign(np.dot(y_transformed, self.weights.T))  # Compute the input vector.
        return np.where(input_hat == -1, 0, 1)  # Convert back to binary values (0 and 1).
