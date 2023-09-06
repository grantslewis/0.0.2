from flask import Flask, render_template, request
import base64
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import model

# Initialize the Flask application
app = Flask(__name__)

caption_path = "Salesforce/blip-image-captioning-large"

# Load the HuggingFace model outside the route so it's loaded only once when the Flask app starts
cap_processor = BlipProcessor.from_pretrained(caption_path)
cap_model = BlipForConditionalGeneration.from_pretrained(caption_path, torch_dtype=torch.float16).to("cuda")


# Define a route for the default URL, which loads the form
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transform', methods=['POST'])
def transform_image():
    # Get the data from POST request
    data_url = request.values['imageBase64']
    image_data = base64.b64decode(data_url.split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    
    prompt = model.caption_image(cap_processor, cap_model, image, "a picture of ")
    
    # Run your deep learning pipeline here
    # For example: transformed_image = model(image)
    # Note: you might need to preprocess the image to be compatible with your model and also handle the output as required.
    # For demonstration purposes in this code, let's just save the image as-is
    image.save("received_image.png")
    
    return "Image transformation done!"

if __name__ == '__main__':
    app.run(debug=True)
