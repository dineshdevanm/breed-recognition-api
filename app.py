from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
from PIL import Image
import numpy as np
import json

app = FastAPI(
    title="Dog Breed Recognition API"
)

session = ort.InferenceSession(
    "model/breed_model.onnx"
)

with open(
    "model/classes.json",
    "r"
) as f:
    classes = json.load(f)


def preprocess(image):

    image = image.resize((224,224))

    image = np.array(image).astype(np.float32)

    image /= 255.0

    image = np.transpose(
        image,
        (2,0,1)
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


@app.get("/")
def home():
    return {
        "status": "running"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image = Image.open(
        file.file
    ).convert("RGB")

    image = preprocess(image)

    input_name = (
        session.get_inputs()[0].name
    )

    output = session.run(
        None,
        {
            input_name: image
        }
    )

    pred = int(
        np.argmax(output[0])
    )

    return {
        "breed": classes[pred]
    }