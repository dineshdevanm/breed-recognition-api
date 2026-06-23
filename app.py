from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi.middleware.cors import CORSMiddleware

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://zest.pet",
        "https://www.zest.pet",
        "https://api.zest.pet",
        "http://localhost:5173",
        "https://zest-a27d3.web.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
