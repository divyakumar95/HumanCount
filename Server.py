import numpy as np
from flask import Flask, json, render_template, request
import pickle
import cv2
from skimage.feature import hog
from sklearn.metrics import accuracy_score
import matplotlib.image as mpimg

#create an instance of Flask
app = Flask(__name__)

#Load the model
model = pickle.load(open('Models/rf.pkl','rb'))

@app.route('/disp', methods=['GET','POST'])
def getImage():
    if request.method == 'POST':
        print(request.form)
        r = request
        file = request.files['filename'].read()
        npimg = np.fromstring(file, np.uint8)
        img2 = cv2.imdecode(npimg,cv2.IMREAD_GRAYSCALE)
        X_test, hog_img, img1 = Feat(img2)
        y_pred = predict(X_test)
        mpimg.imsave('images/img.png', img1)
        mpimg.imsave('images/hog.png', hog_img)
        responseData = {'Y_pred': str(y_pred),'hog_img':'images/hog.png','image':'images/img.png'}
        response = app.response_class(
            response = json.dumps(responseData),
            mimetype='application/json'
            )
        #return response
        return render_template('predict.html', prediction = str(y_pred))

def predict(X):
    if request.method == 'POST':
        prediction = model.predict(X)
        return prediction

def Feat(FileName):
    img1 = cv2.resize(FileName, (256, 256))
    _, hog_image = hog(img1, orientations=16, pixels_per_cell=(5, 5),
                    cells_per_block=(4, 4), visualize=True)#, multichannel=True)
    new = hog_image.flatten()
    mag = np.array(new, dtype='float32')
    return [mag], hog_image,img1


@app.route('/', methods=['GET','POST'])
def home():
    return render_template('ImageCount.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)