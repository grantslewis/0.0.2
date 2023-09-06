from flask import Flask, render_template, request
import base64
import io
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)

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
    
    # Run your deep learning pipeline here
    # transformed_image = your_pipeline(image)
    
    # For demonstration, let's just save the image as-is
    image.save("received_image.png")
    
    return "Image transformation done!"

if __name__ == '__main__':
    app.run(debug=True)
