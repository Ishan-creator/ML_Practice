import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Define the transformation to be applied to the image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

def load_model(model_path):
    # Define the model architecture
    class ANN(torch.nn.Module):
        def __init__(self):
            super(ANN, self).__init__()
            self.fc1 = torch.nn.Linear(32 * 32, 256)
            self.fc2 = torch.nn.Linear(256, 10)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize the model
    model = ANN()

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def predict_image(model, image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Apply the transformation
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

if __name__ == "__main__":
    # Load the model
    model_path = os.path.join("model", "test1.pth")
    model = load_model(model_path)
    
    # Specify the path to the image you want to predict
    image_path = "/home/ishan-pc/Desktop/InternshipRepo/pytorch/images/unnamed.png"
    
    # Get the ground truth label (assuming it's known)
    # Replace the next line with the code to get the ground truth label
    ground_truth_label = 0# Replace 0 with the actual ground truth label
    
    # Perform prediction
    predicted_label = predict_image(model, image_path)
    
    # Compare predicted label with ground truth label
    if predicted_label == ground_truth_label:
        print("Prediction is correct!")
    else:
        print("Prediction is incorrect!")
