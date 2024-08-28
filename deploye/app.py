from fastapi import FastAPI,File,UploadFile
from typing import Annotated
import uvicorn
import tensorflow as tf
import cv2
import numpy as np

# define fastapi instance
app = FastAPI()

# convert image to numpy array and resize to (64 x 64 x 1) used for ml model to predict
def convert_image_to_nparray(data):
    try:
        img = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        if (img.shape is None) or (img.shape != (64, 64)):
            raise Exception("Invalid image. please input image of size 64 x 64 dimension only.")
        img = img / 255.0
        img = np.array(img)
        img = img.reshape(-1, 64, 64, 1)
        return img
    except Exception as e:
           print("Exception raised message :{}".format(e))
    return None

# allow user to upload or input an image of 64 x 64 in dimensions
@app.post('/upload/')
# async def _file_upload(my_file: UploadFile = File(...) ):
async def _file_upload(my_file: Annotated[UploadFile, File(description="An image (.jpg) of 64 x 64 in dimensions only.")]):
    try:
        img = convert_image_to_nparray(await my_file.read())
        if img is not None:
           return predict(img)
    except Exception as e:
        print("Exception raised :{}".format(e))

# ML model prediction
def predict(image):
    model = tf.keras.models.load_model("pneumonia.h5")
    if model.predict(image) > 0.5:
        return "Pneumonia"
    else:
        return "Normal"

if __name__ == "__main__":
    uvicorn.run(app)