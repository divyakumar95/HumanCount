import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, Response, url_for, send_file, flash
import pickle
import jsonpickle
import cv2
from skimage.feature import hog
from sklearn.metrics import accuracy_score
import base64, os, io, sys
from PIL import Image
from werkzeug.utils import secure_filename
from base64 import encodebytes
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

#create an instance of Flask
app = Flask(__name__)

#Load the model
model = pickle.load(open('../Models/rf.pkl','rb'))

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route('/predict/', methods=['GET','POST'])
def getImage():
    if request.method == 'POST':
        print(request.form)
        r = request
        file = request.files['filename'].read()
        npimg = np.fromstring(file, np.uint8)
        img2 = cv2.imdecode(npimg,cv2.IMREAD_GRAYSCALE)
        X_test, hog_img, img1 = Feat(img2)
        y_pred = predict(X_test)
        mpimg.imsave("img.png", img1)
        mpimg.imsave("hog.png", hog_img)
        #encoded_img = encodebytes(img1.getvalue()).decode('ascii')
        response = {'Y_pred': str(y_pred),'hog_img':'./hog.png','image':'./img.png'}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")
        

def predict(X):
    if request.method == 'POST':
        print('hello there')
        prediction = model.predict(X)
        return prediction

def Feat(FileName):
    img1 = cv2.resize(FileName, (256, 256))
    _, hog_image = hog(img1, orientations=16, pixels_per_cell=(5, 5),
                    cells_per_block=(4, 4), visualize=True)#, multichannel=True)
    
    #img = cv2.resize(hog_image, (256, 256))
    new = hog_image.flatten()
    mag = np.array(new, dtype='float32')
    return [mag], hog_image,img1


@app.route('/', methods=['GET','POST'])
def home():
    return render_template('ImageCount.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)