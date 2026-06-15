from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File

import os
import shutil

from services.breed_service import (
    predict_breed
)

from services.age_service import (
    predict_age
)

app = FastAPI(
    title="Pet AI Backend"
)


@app.get("/")
def home():

    return {
        "message":
        "Pet AI Backend Running"
    }


@app.post("/analyze-pet")
async def analyze_pet(
    file: UploadFile = File(...)
):

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    file_path = (
        f"uploads/{file.filename}"
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    breed_result = predict_breed(
        file_path
    )

    age_result = predict_age(
        file_path
    )

    return {
        "breed":
            breed_result,
        "age":
            age_result
    }