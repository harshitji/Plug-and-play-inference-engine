import torch
import torch.nn as nn
import struct

class BasicMLP(nn.Module):
    def __init__(self):
        super(BasicMLP, self).__init__()
        # A simple model: Input(10) -> Linear(10, 6) -> ReLU -> Linear(6, 4) -> ReLU
        self.fc1 = nn.Linear(10, 6)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(6, 4)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x

def main():
    model = BasicMLP()
    model.eval()

    # Generate dummy input to test the model
    # Seed for reproducibility if needed, but random is fine for a single test
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 10)
    
    # Run forward pass to get expected output
    with torch.no_grad():
        expected_output = model(dummy_input)
    
    print("Dummy Input:\n", dummy_input)
    print("Expected Output:\n", expected_output)

    # Function to export a PyTorch tensor to a binary file as sequentially packed floats
    def export_tensor_to_bin(tensor, file_handle):
        # Detach and convert to numpy array
        np_array = tensor.detach().numpy()
        flattened = np_array.flatten()
        # Pack floats ('f' is standard size float, typically 32-bit)
        for val in flattened:
            file_handle.write(struct.pack('f', val))

    # 1. Export the Model Weights and Biases
    # The C++ engine needs to load these in the exact same order!
    # Order: FC1 Weights, FC1 Bias, FC2 Weights, FC2 Bias
    with open('model_weights.bin', 'wb') as f:
        export_tensor_to_bin(model.fc1.weight, f)
        export_tensor_to_bin(model.fc1.bias, f)
        export_tensor_to_bin(model.fc2.weight, f)
        export_tensor_to_bin(model.fc2.bias, f)
    print("Exported model_weights.bin")

    # 2. Export the Test Input
    with open('test_input.bin', 'wb') as f:
        export_tensor_to_bin(dummy_input, f)
    print("Exported test_input.bin")

    # 3. Export the Expected Test Output
    with open('test_output.bin', 'wb') as f:
        export_tensor_to_bin(expected_output, f)
    print("Exported test_output.bin")

if __name__ == '__main__':
    main()
