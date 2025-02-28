# Overview
This project demonstrates the applications of face landmark detections. 
The main product is a real-time face swapping application. 
The side product is a real-time eye-closing detection application which serves the purpose of drowsiness detection system of a driver to promote safe driving. </br>

# Exectution
To run the real-time face swapping application, execute the command "python face_swap_app.py" in terminal. </br>
To run the real-time drowsiness detection application, execute the command "python EyeDetect_v2.py" </br>

# Details on Face Swap
This app relies on the shape_predictor_68_face_landmarks detector which is called by the dlib package. The app takes one source image as the input face and a sequence of real-time video captured image as the output faces. The input face is mapped to each output face and consequently the real-time video will display with the output face swapped with the input face. The construction of face map consists of the following sequential steps. <br>
## Steps to construct the face swapping 
### Step 1
Identify the face landmark points of the input face and the output face. 
### Step 2
Segment the input face and output face with triangulation based on the face landmarks established in Step 1
### Step 3
Map, via affine transformseach, triangulated subset of the input face to the corresponding triangulated region of the output face identified by the face landmark point indices. (Note that when the word "region" is used to replace the word "subset", it means that the image contained in the region need not be the same image as it originally was) 
### Step 4
Piece together the mapped triangulated regions to reconstruct the output face
### Step 5
Use inpaint, median blur and seamlesscloning to remove/deviate the boundaries of the triangulated regions.














