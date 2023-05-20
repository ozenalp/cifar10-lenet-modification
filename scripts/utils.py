import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def set_seed(seed):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for NumPy's random number generator
    np.random.seed(seed)
    
    # Set the seed for PyTorch's random number generator for CPU and CUDA (GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Ensures that CUDA selects the same algorithm each time an operation is run on a tensor with the same shape
    torch.backends.cudnn.deterministic = True



def imshow(img):
    # Unnormalize the input image tensor. This reverses the normalization operation 
    # where we previously subtracted 0.5 and divided by 0.5 during preprocessing.
    img = img / 2 + 0.5
    
    # Convert the PyTorch tensor image to a NumPy array so that matplotlib can display it
    npimg = img.numpy()
    
    # Transpose the image dimensions from (channels, height, width) to (height, width, channels)
    # Matplotlib requires this shape to correctly display images
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



def count_parameters(model):
    # Sum up the number of elements (parameters) in each tensor of the model's parameters
    # Only consider parameters that require gradients (i.e., trainable parameters)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def calculate_accuracy(y_pred, y):
    # Identify the predicted class for each example in the batch by finding the index (class number)
    # with the highest prediction score. y_pred has shape [batch_size, num_classes].
    top_pred = y_pred.argmax(1, keepdim=True)

    # Check which predictions match the true classes. This line of code gives a boolean array, 
    # where an element is True if the prediction and true class match, and False otherwise.
    correct = top_pred.eq(y.view_as(top_pred)).sum()

    # Calculate the accuracy by taking the mean of the 'correct' boolean array.
    # We need to convert the correct tensor to float to perform floating point division.
    acc = correct.float() / y.shape[0]

    return acc



def epoch_time(start_time, end_time):
    # Calculate the total elapsed time between start_time and end_time in seconds
    elapsed_time = end_time - start_time

    # Calculate the number of whole minutes that the elapsed time comprises
    elapsed_mins = int(elapsed_time / 60)
    
    # Calculate the remaining number of seconds after subtracting the minutes
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    # Return the elapsed time in minutes and seconds
    return elapsed_mins, elapsed_secs



def plot_loss(num_epochs, training_losses, validation_losses):
    # Create a numpy array representing epoch numbers starting from 1 till num_epochs
    epochs = np.arange(1, num_epochs + 1)

    # Plot the training losses with respect to epochs; depict these as blue lines
    plt.plot(epochs, training_losses, 'b', label='Training Loss')

    # Plot the validation losses with respect to epochs; depict these as red lines
    plt.plot(epochs, validation_losses, 'r', label='Validation Loss')

    # Set the title for the graph
    plt.title('Training Loss vs. Validation Loss')

    # Label the x-axis as 'Epochs' 
    plt.xlabel('Epochs')

    # Label the y-axis as 'Loss'
    plt.ylabel('Loss')

    # Place a legend on the plot to distinguish between training and validation loss
    plt.legend()

    # Enable the grid for better visualization
    plt.grid(True)

    # Display the plot
    plt.show()



def df_results(num_epochs, train_losses, valid_losses, train_accs, valid_accs, model_name = "LeNet-xx"):
    # Create a numpy array representing epoch numbers starting from 1 till num_epochs
    epochs = np.arange(1, num_epochs + 1)
    
    # Create a dictionary with data for each epoch
    data = {
        'epoch': epochs, 
        'train_loss': train_losses, 
        'valid_loss': valid_losses,
        'train_acc': train_accs, 
        'valid_acc': valid_accs
    }

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a .csv file for future reference
    # The name of the .csv file is based on the model's name
    df.to_csv('../results/'+ model_name +'.csv', index=False)

    # Return the created DataFrame
    return df