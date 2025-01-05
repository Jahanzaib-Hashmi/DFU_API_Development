# import os
# from flask import Flask, request, jsonify, redirect
# import torch
# from torchvision import transforms
# from PIL import Image
# import torch.nn as nn
# import torchvision.models as models

# # Initialize the Flask application
# app = Flask(__name__)

# # Model setup function
# def build_model(pretrained=True, fine_tune=True, num_classes=4):
#     if pretrained:
#         print('[INFO]: Loading pre-trained weights')
#     else:
#         print('[INFO]: Not loading pre-trained weights')
#     model = models.efficientnet_b1(weights='DEFAULT')  # Load the base model
#     if fine_tune:
#         print('[INFO]: Fine-tuning all layers...')
#         for params in model.parameters():
#             params.requires_grad = True
#     else:
#         print('[INFO]: Freezing hidden layers...')
#         for params in model.parameters():
#             params.requires_grad = False
#     # Update the final classifier head
#     model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
#     return model

# # Initialize and load the model
# model_path = 'effnetb1_best_model.pth'  # Path to your trained model weights
# model = build_model()
# try:
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
#     print("[INFO]: Model weights loaded successfully.")
# except Exception as e:
#     print(f"[ERROR]: Failed to load model weights.")

# model.eval()  # Set the model to evaluation mode

# # Define image transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to match model input
#     transforms.ToTensor(),          # Convert image to tensor
# ])

# @app.route('/')
# def home():
#     return redirect('/predict')  # Redirect to the /predict endpoint

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'GET':
#         return "Use a POST request to this endpoint to upload an image for prediction."
    
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     try:
#         # Process the uploaded image
#         img = Image.open(file.stream).convert('RGB')
#         img = transform(img).unsqueeze(0)  # Add batch dimension
        
#         # Perform the prediction
#         with torch.no_grad():
#             output = model(img)
#         _, predicted = torch.max(output, 1)  # Get the predicted class index
#         return jsonify({'prediction': int(predicted.item())})
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
