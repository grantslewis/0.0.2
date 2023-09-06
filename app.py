from flask import Flask, render_template, request
import base64
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import torch
import avatar_generation

port = 5000

# Initialize the Flask application
app = Flask(__name__)

caption_path = "Salesforce/blip-image-captioning-large"
caption_text = "a picture of "

control_net_path = "diffusers/controlnet-canny-sdxl-1.0"
vae_path = "madebyollin/sdxl-vae-fp16-fix"
diffuser_path = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_conditioning_scale = 0.5
device = "cuda"

# Load the HuggingFace model outside the route so it's loaded only once when the Flask app starts
cap_processor = BlipProcessor.from_pretrained(caption_path)
cap_model = BlipForConditionalGeneration.from_pretrained(caption_path, torch_dtype=torch.float16).to(device)

controlnet = ControlNetModel.from_pretrained(
    control_net_path, torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    diffuser_path, controlnet=controlnet, vae=vae, torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()



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
    
    prompt = avatar_generation.caption_image(cap_processor, cap_model, image, text=caption_text, device=device)
    
    result_image = avatar_generation.generate_avatar(pipe, prompt, controlnet_conditioning_scale, image)
    result_image.save("result_image.png")
    # Run your deep learning pipeline here
    # For example: transformed_image = model(image)
    # Note: you might need to preprocess the image to be compatible with your model and also handle the output as required.
    # For demonstration purposes in this code, let's just save the image as-is
    image.save("received_image.png")
    
    return result_image #"Image transformation done!"

if __name__ == '__main__':
    app.run(debug=True, port=port)
