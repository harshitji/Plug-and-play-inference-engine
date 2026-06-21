import torch
import torchvision.models as models

def main():
    print("Downloading pre-trained ResNet50...")
    # ResNet50 is the perfect "Hello World" model for benchmarking inference engines
    model = models.resnet50(pretrained=True)
    model.eval()

    # Create dummy input: Batch size 1, 3 channels (RGB), 224x224 pixels
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    onnx_path = "resnet50.onnx"
    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,                       # Model being run
        dummy_input,                 # Model input
        onnx_path,                   # Where to save the output file
        export_params=True,          # Store the trained parameter weights inside the model file
        opset_version=12,            # The ONNX version to export the model to
        do_constant_folding=True,    # Optimizes constants
        input_names = ['input'],     # The model's input names
        output_names = ['output'],   # The model's output names
        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}} # Allow dynamic batch sizes later
    )
    print("Export complete! ResNet50 ONNX model is ready.")

if __name__ == "__main__":
    main()
