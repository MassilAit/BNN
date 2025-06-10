from check_activation import plot_activations, plot_layer_data, plot_gradients
N_INPUTS = 2
N_HIDDEN = 2
bias = False 
RESULT = "-1,-1,1,1" 
is_bias = "bias" if bias else "no_bias"

for i in range(1,6):
    params = f"Activations/Binarized/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/{RESULT}/params/{i}_params.json"
    grads = f"Activations/Binarized/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/{RESULT}/grads/{i}_grads.json"
    activation = f"Activations/Binarized/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/{RESULT}/activation/{i}_activation.json"

    #plot_gradients(grads)
    plot_layer_data(params, ["model.linear_0", "model.linear_1"])


#for name in ["FullyBinarizedModel", "BinarizedModel"]:
#    params = f"Comparaison/params/{name}_params.json"
#    grads = f"Comparaison/grads/{name}_grads.json"
#    activation =  f"Comparaison/activation/{name}_activation.json"
#    plot_layer_data(params, ["model.linear_0", "model.linear_1"])
#    plot_gradients(grads)
