import numpy as np

class EnhancedBAM:
    def __init__(self, input_size, output_size):
        """
        Initialize the Enhanced BAM with given input and output sizes.
        :param input_size: Number of input neurons.
        :param output_size: Number of output neurons.
        """
        self.weights = np.zeros((input_size, output_size))  # Initialize weights to zeros.
        self.known_outputs = None  # To store valid output patterns.

    def train_usa_bam(self, inputs, outputs):
        """
        Train the BAM using the USA-BAM method (Unlearning Spurious Attractors).
        This method not only updates weights for correct patterns but also actively removes
        weights caused by spurious attractors (incorrect intermediate patterns).

        :param inputs: Array of input patterns (binary values).
        :param outputs: Array of output patterns (binary values).
        """
        # Store valid outputs for validation later
        self.known_outputs = np.array(outputs)

        # Transform patterns to [-1, 1]
        inputs_transformed = 2 * np.array(inputs) - 1
        outputs_transformed = 2 * np.array(outputs) - 1

        # Step 1: Standard weight update for correct associations
        for x, y in zip(inputs_transformed, outputs_transformed):
            self.weights += np.outer(x, y)  # Strengthen correct associations

        # Step 2: Identify and weaken spurious attractors
        # Generate all possible combinations of incorrect inputs/outputs
        num_patterns = len(inputs)
        for i in range(num_patterns):
            for j in range(num_patterns):
                if i != j:
                    incorrect_input = inputs_transformed[i]
                    incorrect_output = outputs_transformed[j]
                    predicted_output = np.sign(np.dot(incorrect_input, self.weights))
                    predicted_input = np.sign(np.dot(incorrect_output, self.weights.T))
                    self.weights -= 0.1 * np.outer(predicted_input, predicted_output)

    def train_bdr_bam(self, inputs, outputs, learning_rate, epochs):
        """
        Train the BAM using the BDR-BAM method (Bidirectional Delta Rule).
        :param inputs: Array of input patterns.
        :param outputs: Array of output patterns.
        :param learning_rate: Learning rate for weight updates.
        :param epochs: Number of training epochs.
        """
        self.known_outputs = np.array(outputs)  # Store valid outputs for later validation.
        for epoch in range(epochs):  # Repeat training for the specified number of epochs.
            for x, y in zip(inputs, outputs):
                x_transformed = 2 * x - 1
                y_transformed = 2 * y - 1

                # Forward pass: Predict the output
                output_hat = np.sign(np.dot(x_transformed, self.weights))

                # Check if the prediction is correct
                if np.array_equal(output_hat, y_transformed):  # If prediction is correct, skip update
                    continue

                # Backward pass: Predict the input from the predicted output
                input_hat = np.sign(np.dot(output_hat, self.weights.T))

                # Update the weights using the Bidirectional Delta Rule
                self.weights += learning_rate * (
                        np.outer(x_transformed, y_transformed) -  # Correct association.
                        np.outer(input_hat, output_hat)  # Incorrect association.
                )

    def find_closest_output(self, output_hat):
        """
        Find the closest valid output to the given output_hat.
        :param output_hat: Output pattern to validate.
        :return: Closest valid output pattern.
        """
        # Calculate Hamming distances between the predicted output and all known outputs.
        distances = [np.sum(output_hat != valid_output) for valid_output in self.known_outputs]
        # Find the index of the known output with the smallest distance.
        closest_idx = np.argmin(distances)
        return self.known_outputs[closest_idx]  # Return the closest valid output pattern.

    def recall_forward(self, input_pattern):
        """
        Perform a forward recall (input to output), ensuring output matches a known pattern.
        :param input_pattern: Input pattern to recall.
        :return: Recalled output pattern.
        """
        # Transform the input pattern to [-1, 1].
        x_transformed = 2 * input_pattern - 1
        # Predict the output using the weights.
        output_hat = np.sign(np.dot(x_transformed, self.weights))
        # Convert the predicted output back to [0, 1].
        output_hat = np.where(output_hat == -1, 0, 1)
        # Find the closest valid output to ensure the output is valid.
        return self.find_closest_output(output_hat)

    def recall_backward(self, output_pattern):
        """
        Perform a backward recall (output to input).
        :param output_pattern: Output pattern to recall.
        :return: Recalled input pattern.
        """
        # Transform the output pattern to [-1, 1].
        y_transformed = 2 * output_pattern - 1
        # Predict the input using the weights (transpose for backward recall).
        input_hat = np.sign(np.dot(y_transformed, self.weights.T))
        # Convert the predicted input back to [0, 1].
        return np.where(input_hat == -1, 0, 1)
