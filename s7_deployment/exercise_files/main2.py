from fastapi import UploadFile, File
from typing import Optional
from http import HTTPStatus
import re
from pydantic import BaseModel
import json
import os

app = FastAPI()

@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...)):
    with open('image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)
        image.close()

    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response