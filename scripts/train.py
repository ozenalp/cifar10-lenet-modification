from scripts.utils import set_seed, calculate_accuracy
from tqdm import tqdm
import torch

# Set a seed value to ensure the reproducibility of the experiments
SEED = 1773

# Pass the SEED value to the set_seed function to set the seed for Python's random module, 
# NumPy's random number generator, PyTorch's random number generator for CPU and CUDA
set_seed(SEED)


def train_model(model, iterator, optimizer, criterion, device):
    # Initialize the total loss and accuracy for the epoch
    epoch_loss = 0
    epoch_acc = 0

    # Set the model to training mode
    model.train()

    # Loop over each batch from the iterator
    for (x, y) in tqdm(iterator, desc="Training", leave=False):
        # Move the batch tensors to the device the model is on
        x = x.to(device)
        y = y.to(device)

        # Reset the gradients from the last iteration
        optimizer.zero_grad()

        # Run the forward pass of the model
        y_pred, _ = model(x)

        # Compute the loss between the predictions and the true labels
        loss = criterion(y_pred, y)

        # Compute the accuracy of the predictions
        acc = calculate_accuracy(y_pred, y)

        # Perform backpropagation to compute gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Accumulate the loss and accuracy values for the epoch
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # Return the average loss and accuracy over the epoch
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model, iterator, criterion, device):
    # Initialize the total loss and accuracy for the epoch
    epoch_loss = 0
    epoch_acc = 0

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations to save memory and computation during evaluation
    with torch.no_grad():
        # Loop over each batch from the iterator
        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
            # Move the batch tensors to the device the model is on
            x = x.to(device)
            y = y.to(device)

            # Run the forward pass of the model
            y_pred, _ = model(x)

            # Compute the loss between the predictions and the true labels
            loss = criterion(y_pred, y)

            # Compute the accuracy of the predictions
            acc = calculate_accuracy(y_pred, y)

            # Accumulate the loss and accuracy values for the epoch
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    # Return the average loss and accuracy over the epoch
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
