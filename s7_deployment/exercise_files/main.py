from fastapi import FastAPI, Query, File, UploadFile
from fastapi.responses import FileResponse
from http import HTTPStatus
import re
from typing import Optional
#from pydantic import BaseModel
import json
import os
from contextlib import asynccontextmanager
# import cv2


# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# def predict_step(image_paths, gen_kwargs):
#     images = []
#     for image_path in image_paths:
#         i_image = Image.open(image_path)
#         if i_image.mode != "RGB":
#             i_image = i_image.convert(mode="RGB")

#         images.append(i_image)
#     pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device)
#     output_ids = model.generate(pixel_values, **gen_kwargs)
#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

#     preds = [pred.strip() for pred in preds]
    
#     return preds

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Hello please input image")
    yield
    os.remove('tmp_image.jpg')
    os.remove('image.jpg')
    print("World")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.get("/text_model/")
def contains_email():
    json_path = os.path.join(os.path.dirname(__file__), "mail.json")

    with open(json_path, "r") as file:
        data = json.load(file)

    email = data.get("email")
    domain_match = data.get("domain_match")

    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    response = {
        "email": email,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, email) is not None,
        "is-correct-domain": domain_match == email.split('@')[1].split('.')[0]
    }
    return response

# @app.post("/cv_model/")
# async def cv_model(data: UploadFile = File(...), h: None | int = 28, w: None | int = 28):
#     with open('image.jpg', 'wb') as image:
#         content = await data.read()
#         image.write(content)
#     img = cv2.imread('image.jpg')
#     res = cv2.resize(img, (h, w))
#     cv2.imwrite('resized_image.jpg', res)

#     response = FileResponse('resized_image.jpg')
#     return response

# @app.post("/good_cv_model/")
# async def good_cv_model(data: UploadFile = File(...), max_length: Optional[int] = Query(16)):
    

#     
#     model.to(device)

#     tmp_image_path = "tmp_image.jpg"

#     with open('image.jpg', 'wb') as image:
#         content = await data.read()
#         image.write(content)
#     img = cv2.imread('image.jpg')
#     res = cv2.resize(img, (32, 32))
#     cv2.imwrite(tmp_image_path, res)

#     gen_kwargs = {"max_length": max_length, "num_beams": 8, "num_return_sequences": 1}
    
#     preds = predict_step([tmp_image_path], gen_kwargs=gen_kwargs)
#     response = preds[0]
#     return response
