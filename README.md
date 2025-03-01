# 1.Overview
This project demonstrates the applications of face landmark detections. 
The main product is a real-time face swapping application. 
The side product is a real-time eye-closing detection application which serves the purpose of drowsiness detection system of a driver to promote safe driving. </br>

# 2.Execution
To run the real-time face swapping application, execute the command "python face_swap_app.py" in terminal. </br>
To run a modified version of real-time face swapping which updates the face every 5 seconds, execute the command "python face_swap_app_multiple.py" in terminal </br>
To run the real-time drowsiness detection application, execute the command "python EyeDetect_v2.py" in terminal </br>

# 3.Details on Face Swap
This app relies on the shape_predictor_68_face_landmarks detector which is called by the dlib package. The app takes one source image as the input face and a sequence of real-time video captured images as the output faces. The input face is mapped to each output face and consequently the real-time video will display the output face mapped from the input face. The construction of face map consists of the following sequential steps. <br>
## 3.1 Steps(Methodology) to construct the face swapping application
### Step 1
Identify the face landmark points of the input face and the output face. 
### Step 2
Segment the input face and output face with triangulation based on the face landmarks established in Step 1.
### Step 3
Map, via affine transforms, each triangulated subset of the input face to the corresponding triangulated region of the output face identified by the face landmark point indices. (Note that when the word "region" is used to replace the word "subset", it means that the image contained in the region need not be the same image as it originally was). 
### Step 4
Piece-stitch together the mapped triangulated regions to reconstruct the output face.
### Step 5
Use inpaint, median blur and seamlesscloning to remove/deviate the boundaries of the triangulated regions.
### Step 6
The main application uses one single source input face throughout one application run. To allow input face updating, the "threading" package is used and the resulted "face_swap_app_multiple.py" updates the output face every 5 seconds as a consequence of input update. 

## 3.2 Result Display
### Results/FaceSwap/Images 
contains images to showchase the one-to-one map from the input face to the output.
### Results/FaceSwap/Videos/Single_Face 
contains videos to showcase some results generated by "face_swap_app.py".
### Results/FaceSwap/Videos/Multi_Faces 
contains one video to showcase a result generated by "face_swap_app_multiple.py".

# 4.Details on Eye Closing/Drowsiness Detection 
This app relies on the 400+ face landmarks generated by the mediapipe package. It is crucial to accurately get the locations of landmark points of eyes for real-time eye-closing detection and for this task mediapipe package does a better job than the shape_predictor_68_face_landmarks detector does. For the alert sound, the pyttsx3 package is invoked. The default languages for alert messages are set be Mandarin/Cantonese, with the alert message being "疲劳驾驶请注意！", meaning "be cautious of tired driving". Other languages can also be selected upon users' request.</br>

## 4.1 Steps(Methodology) to construct the drowsiness detection application
### Step 1
Identify the eye landmark points of the person in camera (usually the driver).
### Step 2
Calculate the widths and lengths of eyes and conseqeuntly the average width/length ratio, denoted by the eye closing ratio.
### Step 3
Set a threshold of 0.2. If the eye closing ratio is below the threshold, it is considered to be one occurence of eye closing. The system will not alert for one single eye closing occurence for reasons to be described in Section 4.2.
### Step 4 
Set a threshold of drowsiness level of 5 and initialize the drowsiness level to be 0. For each of consecutive occurences of eye closing, the drowsiness level is increased by 1; at the end of consecutive occurences of eye closing which means there exist a none-eye-closing status(defined by eye closing ratio>0.2) that ends the consecutive sequence of eye closing, e.g., closed, closed, ..., closed, open, the drowsiness level is reset to 0. If the drowsiness level is greater than 5, the threshold, a warning messsage will be displayed on screen and an alert voice (currently in Chinese) will be raised and heard by the user(ususally the driver)

## 4.2 Why use accumulative count of eye closing instead of instant alert of a single detection
The detection of eye closing is acute. An unfavorable side effect is the system will raise false alarm when the driver blinks.
To solve this problem, it is reasonable to declare drowsiness only if there exist a signficant length of consective eye closing occurences. Once the consecutive occurences end, it is fair to claim that the driver is awake.

## 4.3 Result Display
### Results/Drowsiness/Detection 
contains three videos of real-time drowsiness detection with alert voice being spoken in Cantonese/English/Mandarin














