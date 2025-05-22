from check_activation import *


for i in range(10):
    plot_activations(f"Result/{i}_activations.json","layer1")
#plot_layer_data("Result/1_params.json", ["model.linear_0", "model.linear_1"])