import torch
import copy
import numpy as np

def extract_params(model):
    """
    Extract the parameter values of a model and return them in a dictionary.

    Args:
        model (torch.nn.Module): The model whose parameters will be extracted.

    Returns:
        dict: A dictionary containing parameter names as keys and their values as lists.
    """
    params_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_dict[name] = copy.deepcopy(param.data.cpu().numpy().tolist())
    return params_dict

def extract_gradients(model):
    """
    Extract the gradients of each parameter of the model and return them in a dictionary.

    Args:
        model (torch.nn.Module): The model whose gradients will be extracted.

    Returns:
        dict: A dictionary containing parameter names as keys and their gradients as lists.
    """
    gradients_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients_dict[name] = copy.deepcopy(param.grad.cpu().numpy().tolist())
    return gradients_dict


class EarlyStopping:
    def __init__(self, patience=500, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



def train_model(model, train_loader, optimizer, num_epochs, criterion, verbose=False, record_params=False, record_gradients=False, early_stop=False, delta=0.001, patience=500):
    """
    Train the model, with options to record parameter values and gradients at each epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The DataLoader providing training data.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        num_epochs (int): The number of epochs to train for.
        criterion (callable): The loss function to use.
        record_params (bool, optional): Whether to record parameter values during training. Defaults to False.
        record_gradients (bool, optional): Whether to record gradient values during training. Defaults to False.

    Returns:
        tuple: A tuple containing two lists: recorded parameters and gradients.
    """
    params = []
    grads = []
    average_loss = 0
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if record_gradients:
                grads.append(extract_gradients(model))
    
            if record_params:
                params.append(extract_params(model))


        average_loss = running_loss / len(train_loader)
        early_stopping(average_loss)
        
        if verbose:
            if epoch%100==0 : 
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

        if early_stop and early_stopping.early_stop :
            print(f"Early stopping triggered : {average_loss}")
            break

    return average_loss, params, grads


def train_model_activations(model, train_loader, optimizer, num_epochs, criterion, record_params=False, record_gradients=False):

    params = []
    grads = {}
    activation = {}
    loss_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            
            if record_params:
                params.append(extract_params(model))

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            tuple_inputs=tuple(inputs.squeeze(0).tolist())

            activations_serializable = {key: value.tolist() for key, value in model.activations.items()}

            if str(tuple_inputs) not in activation:
                activation[str(tuple_inputs)]=[]
            
            activation[str(tuple_inputs)].append(copy.deepcopy(activations_serializable))

            if record_gradients:
                if str(tuple_inputs) not in grads:
                    grads[str(tuple_inputs)]=[]
                grads[str(tuple_inputs)].append(extract_gradients(model))
    

        average_loss = running_loss / len(train_loader)

        loss_list.append(average_loss)

        #early_stopping(average_loss)
        if epoch%100==0 : 
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

        #if early_stopping.early_stop:
        #    print(f"Early stopping triggered : {average_loss}")
        #    break

    return params, grads, activation, loss_list