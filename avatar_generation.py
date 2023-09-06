import torch
from PIL import Image
import cv2
import numpy as np

def caption_image(cap_process, cap_model, image, text):
    inputs = cap_process(image, text, return_tensors="pt").to("cuda", torch.float16)

    out = cap_model.generate(**inputs)
    prompt = cap_process.decode(out[0], skip_special_tokens=True)
    return prompt

def canny_generation(image, t_lower=100, t_upper=200):
    image = np.array(image)
    image = cv2.Canny(image, t_lower, t_upper)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def generate_avatar(pipe, prompt, controlnet_conditioning_scale, image):
    control_image = canny_generation(image)
    
    image = pipe(
        prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=control_image
    ).images[0]
    return image


# def generate_image(prompt, control_net, control_net_scale, vae, pipe)

# def run(model_path, caption_path, raw_image, text):

#     processor = BlipProcessor.from_pretrained(caption_path)
#     model = BlipForConditionalGeneration.from_pretrained(caption_path, torch_dtype=torch.float16).to("cuda")

#     # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'


#     # text = "a picture of "
#     inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

#     out = model.generate(**inputs)
#     prompt = processor.decode(out[0], skip_special_tokens=True)
    

# if __name__ == '__main__':
#     model_path = ""
#     caption_path = "Salesforce/blip-image-captioning-large"
    