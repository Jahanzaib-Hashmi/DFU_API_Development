from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np

# Define constants for image processing
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240

# Load your pre-trained model
model = tf.keras.models.load_model('DFU_v1_sgd_BCE_100.keras')

# Define class names
class_names = ['Abnormal(Ulcer)', 'Normal(Healthy skin)']

# Initialize FastAPI app
app = FastAPI()

def decode_img(img_data):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img_data, channels=3)
    # Resize the image to the desired size
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    # Normalize the image to [0, 1]
    img = img / 255.0
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()

    # Process the image
    try:
        image = decode_img(contents)  # Decode and preprocess the image
        # Add batch dimension
        image = tf.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_index]
        
        return {"prediction": predicted_class}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
