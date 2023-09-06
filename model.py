import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
# import argparse


def caption_image(cap_process, cap_model, image, text):
    inputs = cap_process(image, text, return_tensors="pt").to("cuda", torch.float16)

    out = cap_model.generate(**inputs)
    prompt = cap_process.decode(out[0], skip_special_tokens=True)
    return prompt



def run(model_path, caption_path, raw_image, text):

    processor = BlipProcessor.from_pretrained(caption_path)
    model = BlipForConditionalGeneration.from_pretrained(caption_path, torch_dtype=torch.float16).to("cuda")

    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'


    # text = "a picture of "
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    prompt = processor.decode(out[0], skip_special_tokens=True)
    

if __name__ == '__main__':
    model_path = ""
    caption_path = "Salesforce/blip-image-captioning-large"
    