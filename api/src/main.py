from fastapi import FastAPI, File, UploadFile
from classifier import Classifier

app = FastAPI()
classifier = Classifier()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
async def upload_file(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png."

    input_data = classifier.preprocessing(await file.read())
    label, score = classifier.predict(input_data)

    return {'predictions':[{'class':[label], 'score':[score]}]}
