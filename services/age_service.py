import json
import cv2
import numpy as np
import onnxruntime as ort

AGE_MODEL_PATH = "models/age_model.onnx"
AGE_CLASSES_PATH = "classes/age_classes.json"

session = ort.InferenceSession(
    AGE_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

with open(
    AGE_CLASSES_PATH,
    "r"
) as f:
    AGE_CLASSES = json.load(f)


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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict_age(image_path):

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
    logits = outputs[0][0]

    probs = softmax(logits)

    idx = np.argmax(probs)

    confidence = float(probs[idx])
    

    return {
        "group":
            AGE_CLASSES[idx],
        "confidence":
            round(
                confidence * 100,
                2
            )
    }