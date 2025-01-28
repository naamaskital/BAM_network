import numpy as np
import pandas as pd

from BAM import BAM

# Load the training dataset
train_data = pd.read_csv('English_Uppercase_8x8_ASCII_Complete.csv')

# Prepare inputs (64 pixel columns) and outputs (7 ASCII bit columns)
train_inputs = train_data.iloc[:, :64].values
train_outputs = train_data.iloc[:, 64:71].values

# Train the BAM network
bam = BAM(64, 7)
bam.train(train_inputs, train_outputs)

# Function to add noise to the inputs
def add_noise(data, noise_level):
    """
    Add noise to binary data.
    :param data: Original binary data (numpy array)
    :param noise_level: Percentage of bits to flip
    :return: Data with noise
    """
    noisy_data = data.copy()
    n_samples, n_features = data.shape
    n_noisy_bits = int(noise_level * n_features)

    for i in range(n_samples):
        noisy_indices = np.random.choice(n_features, n_noisy_bits, replace=False)
        noisy_data[i, noisy_indices] = 1 - noisy_data[i, noisy_indices]  # Flip bits

    return noisy_data

# Print all letters using the recall_backward function, labeled by ASCII character
print("\nRecalled letters using recall_backward function:")

for i in range(len(train_outputs)):
    output_pattern = train_outputs[i]
    recalled_input = bam.recall_backward(output_pattern)

    # Convert output pattern (7 bits) to ASCII character
    binary_string = "".join(map(str, output_pattern))
    ascii_character = chr(int(binary_string, 2))  # Convert binary to decimal, then to ASCII

    # Format the recalled input as an 8x8 "image"
    print(f"Letter '{ascii_character}':")
    for j in range(8):
        row = recalled_input[j * 8:(j + 1) * 8]  # Every 8 bits form a row
        print("".join(str(bit) for bit in row))
    print("-" * 16)

# Test the BAM network with different noise levels
noise_levels = [0.05, 0.10, 0.20]  # 5%, 10%, 20% noise
for noise_level in noise_levels:
    print(f"\nTesting with {int(noise_level * 100)}% noise:")
    noisy_inputs = add_noise(train_inputs, noise_level)

    correct_count = 0
    for i in range(len(noisy_inputs)):
        sample_input = noisy_inputs[i]
        predicted_output = bam.classify(sample_input)
        expected_output = train_outputs[i]

        # Convert predicted_output and expected_output to binary strings
        predicted_binary = "".join(map(str, predicted_output.astype(int)))
        expected_binary = "".join(map(str, expected_output.astype(int)))

        # Convert binary strings to ASCII characters
        predicted_letter = chr(int(predicted_binary, 2))
        expected_letter = chr(int(expected_binary, 2))

        # Check if the prediction matches the expected output
        if np.array_equal(predicted_output, expected_output):
            correct_count += 1
            print(f"Correct: {expected_letter}")
        else:
            print(f"Incorrect: Predicted {predicted_letter}, Expected {expected_letter}")

    noise_accuracy = correct_count / len(noisy_inputs)
    print(f"Accuracy at {int(noise_level * 100)}% noise: {noise_accuracy:.2%}")

# Load test dataset
test_data = pd.read_csv('letters_data_test.csv')

# Prepare inputs (64 pixel columns) and outputs (7 ASCII bit columns)
test_inputs = test_data.iloc[:, :64].values
test_outputs = test_data.iloc[:, 64:71].values
print("\nTesting with new data:")
# Evaluate accuracy on the test set
correct_count = 0
for i in range(len(test_inputs)):
    sample_input = test_inputs[i]
    predicted_output = bam.classify(sample_input)
    expected_output = test_outputs[i]

    # Convert predicted_output and expected_output to binary strings
    predicted_binary = "".join(map(str, predicted_output.astype(int)))
    expected_binary = "".join(map(str, expected_output.astype(int)))

    # Convert binary strings to ASCII characters
    predicted_letter = chr(int(predicted_binary, 2))
    expected_letter = chr(int(expected_binary, 2))

    if np.array_equal(predicted_output, expected_output):
        correct_count += 1
        print(f"Correct: {expected_letter}")
    else:
        print(f"Incorrect: Predicted {predicted_letter}, Expected {expected_letter}")

test_accuracy = correct_count / len(test_inputs)
print(f"Test accuracy: {test_accuracy:.2%}")
