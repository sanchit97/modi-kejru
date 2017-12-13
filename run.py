import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

UPLOAD_FOLDER = '/home/unholy-me/Desktop/webapp/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config.from_object('config')
recognizer = cv2.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
globe=""

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
			Ids.append(Id)
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
			globe=filename
			print "             \n\n\n"
			print globe
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

@app.route('/upload', methods=['GET', 'POST'])
def uploaded_file():

	# recognizer = cv2.createLBPHFaceRecognizer()
	# detector= cv2.CascadeClassifier("/home/unholy-me/Desktop/haarcascade_frontalface_default.xml")

	faces,Ids = getImagesAndLabels('/home/unholy-me/Desktop/webapp/static/Narendra/')
	x=[]
	for i in range(len(Ids)):
		x.append(1)
	x=np.array(x)
	# print faces
	# print x
	recognizer.train(faces, x)
	# recognizer.save('trainner.yml')
	# imagePath = sys.argv[1]
	print globe
	print type(globe)
	imagePath= "/home/unholy-me/Desktop/webapp/uploads/"+"modi.png"
	print imagePath
	print "\n\n\n\n\n\n"
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print recognizer.predict(gray)
	# Detect faces in the image
	faces = detector.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30),
	    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the Faces
	for (x, y, w, h) in faces:
	    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow("Faces found", image)
	return '''
	<!doctype html>
	<title>Results</title>
	<h2>Detected Faces:</h2>
	'''
