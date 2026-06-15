import json
import cv2
import numpy as np
import onnxruntime as ort

BREED_MODEL_PATH = "models/breed_model.onnx"
BREED_CLASSES_PATH = "classes/breed_classes.json"

session = ort.InferenceSession(
    BREED_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

with open(
    BREED_CLASSES_PATH,
    "r"
) as f:
    BREED_CLASSES = json.load(f)


def preprocess(image_path):

    img = cv2.imread(image_path)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (224, 224)
    )

    img = img.astype(
        np.float32
    ) / 255.0

    img = np.transpose(
        img,
        (2, 0, 1)
    )

    img = np.expand_dims(
        img,
        axis=0
    )

    return img


def predict_breed(image_path):

    image = preprocess(
        image_path
    )

    input_name = (
        session.get_inputs()[0].name
    )

    outputs = session.run(
        None,
        {input_name: image}
    )

    probs = outputs[0][0]

    idx = int(
        np.argmax(probs)
    )

    confidence = float(
        probs[idx]
    )

    return {
        "name":
            BREED_CLASSES[idx],
        "confidence":
            round(
                confidence * 100,
                2
            )
    }