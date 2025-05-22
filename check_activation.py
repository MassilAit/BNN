import json
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_activations(path, function : str = "ReLu"):    
    with open(path, 'r') as file:
        data = json.load(file)


    # Dictionaries to store extracted activations
    layer1_activations = {}  # Stores activations after ReLU for each input
    layer3_activations = {}  # Stores activations after Sigmoid for each input

    # Extract activations for each input
    for input_key, epochs_data in data.items():
        layer1_activations[input_key] = [entry["act_0"][0] for entry in epochs_data]  # Layer 1 (ReLU)
        layer3_activations[input_key] = [entry["act_1"][0] for entry in epochs_data]  # Layer 3 (Sigmoid)



    # Create a figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Iterate over each input and assign it to a subplot
    for idx, (input_key, relu_activations) in enumerate(layer1_activations.items()):
        sigmoid_activations = layer3_activations[input_key]  # Get sigmoid activations for the same input
        ax = axes[idx]  # Select corresponding subplot

        # Convert list of lists into separate lists for each neuron in ReLU
        relu_values = list(zip(*relu_activations))  # Transpose to get activations per neuron
        #colors = ["red", "blue", "orange", "purple", "cyan"]  # Add more colors if needed

        # Plot ReLU activations for each neuron
        for i, neuron_values in enumerate(relu_values):
            ax.plot(range(len(neuron_values)), neuron_values, linestyle="-", 
                     label=f"{function} Neuron {i + 1}")

        # Plot Sigmoid activations (always green)
        ax.plot(range(len(sigmoid_activations)), sigmoid_activations, linestyle="--", color="green", label="layer2")

        # Titles and labels
        ax.set_title(f"Input {input_key}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Activation Value")
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_layer_data(data, layers, data_type='value', title="Model Data Over Epochs"):
    """
    Plots weights, biases, or gradients for each neuron in the specified layers over multiple epochs.
    
    Args:
        data (str): Path to the JSON file containing model parameters or gradients.
        layers (list): List of layer names to plot (e.g., ["hidden", "output"]).
        data_type (str): Type of data to plot. Options: "value" or "gradient" (default: "value").
        title (str): Title for the plots (default: "Model Data Over Epochs").
    """
    # Load the data from the JSON file
    with open(data, 'r') as file:
        data = json.load(file)


    # Initialize dictionaries to store weights and biases value or gradient for each layer
    layers_weights = {}
    layers_biases = {}

    # Extract weights and biases value or gardient for the layers
    for item in data:
        for layer in layers:
            # Generate the key for weights (e.g., 'hidden.weights', 'output.weights')
            weight_key = f"{layer}.weight"
            bias_key = f"{layer}.bias"
            
            # Handle weights
            if weight_key in item:
                if weight_key not in layers_weights:
                    layers_weights[weight_key] = {}
                for i, neuron_weights in enumerate(item[weight_key]):
                    if f'neuron{i+1}' not in layers_weights[weight_key]:
                        layers_weights[weight_key][f'neuron{i+1}'] = []
                    layers_weights[weight_key][f'neuron{i+1}'].append(np.array(neuron_weights))
            
            # Handle biases
            if bias_key in item:
                if bias_key not in layers_biases:
                    layers_biases[bias_key] = {}
                for i, bias in enumerate(item[bias_key]):
                    if f'neuron{i+1}' not in layers_biases[bias_key]:
                        layers_biases[bias_key][f'neuron{i+1}'] = []
                    layers_biases[bias_key][f'neuron{i+1}'].append(np.array(bias))

    # Plot the weights and biases for each layer
    for layer in layers:
        # Plot Weights
        weight_key = f"{layer}.weight"
        if weight_key in layers_weights:
            plt.figure(figsize=(12, 6))  # Create a new figure for each layer's weight plot
            for neuron, neuron_weights in layers_weights[weight_key].items():
                for weight_idx in range(len(neuron_weights[0])):
                    plt.plot(range(len(neuron_weights)), [w[weight_idx] for w in neuron_weights], label=f'{neuron} weight {weight_idx + 1}')
            plt.title(f'{title} - {layer} Weights')
            plt.xlabel('Epoch')
            plt.ylabel(f'Weight {data_type}')
            plt.legend()
            plt.grid(True)
            plt.show()

        # Plot Biases
        bias_key = f"{layer}.bias"
        if bias_key in layers_biases:
            plt.figure(figsize=(12, 6))  # Create a new figure for each layer's bias plot
            for neuron, neuron_biases in layers_biases[bias_key].items():
                plt.plot(range(len(neuron_biases)), neuron_biases, label=f'{neuron} Bias')
            plt.title(f'{title} - {layer} Biases')
            plt.xlabel('Epoch')
            plt.ylabel(f'Bias {data_type}')
            plt.legend()
            plt.grid(True)
            plt.show()

