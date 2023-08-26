---

## PyTorch MNIST Classifier

This project is a simple example of a neural network implemented using PyTorch to classify handwritten digits from the MNIST dataset. The code demonstrates the essential steps for building, training, and evaluating a neural network, including data loading, model architecture definition, and the training loop.

### Code Overview

1. **Import Libraries**: The code starts by importing all the necessary libraries such as PyTorch, torchvision, and others.

2. **Device Configuration**: Checks if a GPU is available and sets it for computation; otherwise, it uses a CPU.

3. **Data Preprocessing**: 
    - Uses torchvision's transforms to convert images to PyTorch tensors and normalize them.
    - Loads the MNIST training and test datasets.
    - Creates DataLoader instances for batching, shuffling, and loading data in parallel.

4. **Model Architecture**: 
    - Defines a simple feed-forward neural network with three fully connected layers.
    - Uses ReLU (Rectified Linear Unit) as the activation function for the hidden layers.

5. **Model Initialization**: 
    - Initializes the neural network, loss function (Cross-Entropy Loss), and optimizer (Adam).

6. **Training Loop**: 
    - Iterates through the training dataset multiple times (epochs).
    - In each epoch, it performs a forward pass to compute predictions and loss, followed by a backward pass to update the model parameters.
    - Prints the average loss for each epoch.

7. **Model Evaluation**: 
    - After training, the model is evaluated on the test dataset.
    - Prints the overall test accuracy.

### Running the Code

To run the code, simply execute the script. Make sure you have installed PyTorch and torchvision before running.

```bash
python your_script_name.py
```

### Output

The output will display the device being used (CPU or GPU), the average loss for each training epoch, and the final test accuracy.

---
