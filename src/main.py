from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import asyncio
import random
import hashlib
import cv2
from PipelineWrapper import PipelineWrapper
from diffusers import DiffusionPipeline
import torch
import os


def translate_text(source : str) -> str:
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    
    translation_pipe = pipeline("translation", model=model, tokenizer=tokenizer)
    translated_text = translation_pipe(source)[0]['translation_text']
    return translated_text

def get_keywords_from(en_text: str) -> list:
    keywords_pipe = pipeline("summarization", model="transformer3/H2-keywordextractor")
    keywords = keywords_pipe(en_text)[0]['summary_text'].split(',  ')
    return keywords

def get_thesis_of(en_text : str) -> str:
    thesis_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    summ_output = thesis_pipeline(en_text, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
    return summ_output

def generate_image_desc(thesis : str, keywords : list, style_pos : str, image_type):
    neg_promt = '''ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft'''
    
    keywords = ','.join(keywords)
    compressed_text = f'promt:{thesis}; keywords:{keywords}'

    wrapper = PipelineWrapper(style_pos)
    pipeline = wrapper.get_pipeline_with_style()
    
    pipeline.to("cuda")
    if image_type == 1:
        image = pipeline('Video preview.' + compressed_text, negative_prompt=neg_promt).images[0]
        return image
    elif image_type == 2:
        image = pipeline('Channel Banner.' + compressed_text, negative_prompt=neg_promt, height=800, width=800).images[0]
        return image
    elif image_type == 3:
        image = pipeline('Channel Avatar.' + compressed_text, negative_prompt=neg_promt, height=864, width=2204).images[0]
        return image
    

def generate_images_video(thesis, keywords, video_path):
    neg_promt = '''ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft'''
    
    device = "cuda"
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, safety_checker = None,
        requires_safety_checker = False
    )
    pipeilne = pipe.to(device)
    
    images = []
    frames = get_frames(video_path)
    keywords = ','.join(keywords)
    compressed_text = f'promt:{thesis}; keywords:{keywords}'
    
    pipeline.to("cuda")
    
    for frame in frames:
        image = pipeline(prompt=prompt, negative_prompt=neg_promt, image=init_image, strength=0.75, guidance_scale=5).images[0]
        images.append(image)
    return images
    
    
def get_frames(vid_path) -> list:
    def extract_frames(video_path, key=1):
        a = []
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            if not count % key and key != 1:
                a.append(image)
            success, image = vidcap.read()
            count += 1
        os.chdir('..')
        return a, count

    a, num = extract_frames(vid_path)
    mas_of_frames, num = extract_frames(vid_path, key=num//5)
    return mas_of_frames
    

print('''
Choose image type: 
1. Video Preview
2. Channel Banner
3. Channel Avatar
''')
image_type = int(input('Number: '))


promt = input("Hello! It's incerdible video preview generator, write your promt here: ")
print('''
Choose style:
1. Cyborg
2. Pixel Art
3. Realistic
4. Dark Darker
5. Comics
6. Herge
7. Pencil draw
8. Default
''')
style_pos = int(input('Number: '))


print('''
Choose mode type:
1. Descripton
2. Video + Description
''')
mode_type = int(input('Number: '))

print('\033[96m' + '|> Start text translation' + '\033[0m')
translated_text = translate_text(promt)
print('|> Translated text: ' + translated_text)

print('\033[96m' + '|> Start generating thesis of text'+ '\033[0m')
thesis = get_thesis_of(translated_text)
print('|> Result: ' + thesis)

print('\033[96m' + '|> Start generating keywords'+ '\033[0m')
keywords = get_keywords_from(translated_text)
print('|> Result: ' + str(keywords))

print('\033[96m' + '|> Start generating image'+ '\033[0m')

if mode_type == 1:
    image = generate_image_desc(thesis, keywords, style_pos, image_type)
    
    path = hashlib.md5(str(random()).encode('utf-8')).hexdigest()
    print(f'|> Image saved in ./{path}') 
    image.save(f'./images/{path}.jpg')
elif mode_type == 2:
    video_path = input('Enter video path: ')
    images = generate_images_video(thesis, keywords, video_path)
    
    for image in images:
        path = hashlib.md5(str(random()).encode('utf-8')).hexdigest()
        print(f'|> Image saved in ./{path}') 
        image.save(f'./images/{path}.jpg')