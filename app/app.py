import os
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
import cv2
import sys
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config.from_object('config')

# recognizer = cv2.createLBPHFaceRecognizer()
recognizer_modi = cv2.face.LBPHFaceRecognizer_create()
recognizer_kejru = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# global globe

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getImagesAndLabels(path):
	#get the path of all the files in the folder
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
	#create empth face list
	faceSamples=[]
	#create empty ID list
	Ids=[]
	#now looping through all the image paths and loading the Ids and the images
	for imagePath in imagePaths:
		#loading the image and converting it to gray scale
		pilImage=Image.open(imagePath).convert('L')
		#Now we are converting the PIL image into numpy array
		imageNp=np.array(pilImage,'uint8')
		#getting the Id from the image
		Id=int(os.path.split(imagePath)[-1].split(".")[0])
		# extract the face from the training image sample
		faces=detector.detectMultiScale(imageNp)
		# print faces
		#If a face is there then append that in the list as well as Id of it
		for (x,y,w,h) in faces:
			faceSamples.append(imageNp[y:y+h,x:x+w])
			# print Id
			if Id>100:
				Ids.append(0)
			else:
				Ids.append(1)
	return faceSamples,Ids


@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return redirect(url_for('uploaded_file',filename=filename))
	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form method=post enctype=multipart/form-data>
	  <p><input type=file name=file>
		 <input type=submit value=Upload>
	</form>
	'''

@app.route('/<filename>', methods=['GET'])
def uploaded_file(filename):

	faces,Ids = getImagesAndLabels('Train/')
	x=np.array(Ids)
	y=[]
	recognizer_modi.train(faces, x)
	for i in range(len(x)):
		if x[i]==0:
			y.append(1)
		else:
			y.append(0)
	y=np.array(y)
	recognizer_kejru.train(faces, y)
	# print x
	# print filename
	imagePath= "uploads/"+str(filename)
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Detect faces in the image
	faces = detector.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		# flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		# flags = cv2.CV_HAAR_SCALE_IMAGE
	)

	# print("Found {0} faces!".format(len(faces)))
	facefound=False
	modifound=False
	kejrufound=False
	
	if len(faces)>0:
		facefound=True
	m1,sm1=recognizer_modi.predict(gray)
	m2,sm2=recognizer_kejru.predict(gray)

	if facefound==True:
		if m1==1:
			modifound=True
		if m2==0:
			kejrufound=True

		# Draw a rectangle around the Faces
		for (x, y, w, h) in faces:
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

		facefound=str(facefound)
		# print type(facefound)
	cv2.imwrite("static/results/display.jpg", image)
	image=cv2.imread("static/results/display.jpg",0)

	return render_template("result.html",f=str(facefound),m=str(modifound),k=str(kejrufound),img="static/results/display.jpg")
