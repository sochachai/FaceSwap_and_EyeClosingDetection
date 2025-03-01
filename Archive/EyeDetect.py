import cv2
import mediapipe as mp
class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_con=0.5,
                 min_tracking_con=0.5):
        # Initialize the parameters for face mesh detection
        self.static_image_mode = static_image_mode  # Whether to process images (True) or video stream (False)
        self.max_num_faces = max_num_faces  # Maximum number of faces to detect
        self.refine_landmarks = refine_landmarks  # Whether to refine iris landmarks for better precision
        self.min_detection_con = min_detection_con  # Minimum confidence for face detection
        self.min_tracking_con = min_tracking_con  # Minimum confidence for tracking

        # Initialize Mediapipe FaceMesh solution
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode,
                                                 self.max_num_faces,
                                                 self.refine_landmarks,
                                                 self.min_detection_con,
                                                 self.min_tracking_con)

        # Store the landmark indices for specific facial features
        # These are predefined Mediapipe indices for left and right eyes, iris, nose, and mouth

        self.LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                                   380, 381, 382, 362]  # Left eye landmarks

        self.RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
                                    144, 163, 7]  # Right eye landmarks

        self.LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]  # Left iris landmarks
        self.RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]  # Right iris landmarks

        self.NOSE_LANDMARKS = [193, 168, 417, 122, 351, 196, 419, 3, 248, 236, 456, 198, 420, 131, 360, 49, 279, 48,
                               278, 219, 439, 59, 289, 218, 438, 237, 457, 44, 19, 274]  # Nose landmarks

        self.MOUTH_LANDMARKS = [0, 267, 269, 270, 409, 306, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39,
                                37]  # Mouth landmarks

    def findMeshInFace(self, img):
        # Initialize a dictionary to store the landmarks for facial features
        landmarks = {}

        # Convert the input image to RGB as Mediapipe expects RGB images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to find face landmarks using the FaceMesh model
        results = self.faceMesh.process(imgRGB)

        # Check if any faces were detected
        if results.multi_face_landmarks:
            # Iterate over detected faces (here, max_num_faces = 1, so usually one face)
            for faceLms in results.multi_face_landmarks:
                # Initialize lists in the landmarks dictionary to store each facial feature's coordinates
                landmarks["left_eye_landmarks"] = []
                landmarks["right_eye_landmarks"] = []
                landmarks["left_iris_landmarks"] = []
                landmarks["right_iris_landmarks"] = []
                landmarks["nose_landmarks"] = []
                landmarks["mouth_landmarks"] = []
                landmarks["all_landmarks"] = []  # Store all face landmarks for complete face mesh

                # Loop through all face landmarks
                for i, lm in enumerate(faceLms.landmark):
                    h, w, ic = img.shape  # Get image height, width, and channel count
                    x, y = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values

                    # Store the coordinates of all landmarks
                    landmarks["all_landmarks"].append((x, y))

                    # Store specific feature landmarks based on the predefined indices
                    if i in self.LEFT_EYE_LANDMARKS:
                        landmarks["left_eye_landmarks"].append((x, y))  # Left eye
                    if i in self.RIGHT_EYE_LANDMARKS:
                        landmarks["right_eye_landmarks"].append((x, y))  # Right eye
                    if i in self.LEFT_IRIS_LANDMARKS:
                        landmarks["left_iris_landmarks"].append((x, y))  # Left iris
                    if i in self.RIGHT_IRIS_LANDMARKS:
                        landmarks["right_iris_landmarks"].append((x, y))  # Right iris
                    if i in self.NOSE_LANDMARKS:
                        landmarks["nose_landmarks"].append((x, y))  # Nose
                    if i in self.MOUTH_LANDMARKS:
                        landmarks["mouth_landmarks"].append((x, y))  # Mouth

        # Return the processed image and the dictionary of feature landmarks
        return img, landmarks


# Initialize the FaceMeshDetector with refined iris landmarks for better precision
detector = FaceMeshDetector(refine_landmarks=True)

# Define the facial features (eyes, nose, mouth, iris, and all landmarks) we are interested in
face_parts = ["left_eye_landmarks", "right_eye_landmarks", "nose_landmarks",
              "mouth_landmarks", "all_landmarks", "left_iris_landmarks",
              "right_iris_landmarks"]

# Specify which facial feature to detect (index 2 refers to the nose landmarks here)
face_part = 1

video = cv2.VideoCapture(0)
while True:
    ret, image = video.read()
    # Use the FaceMeshDetector to find facial landmarks in the current frame
    image, landmarks = detector.findMeshInFace(image)

    # Try to draw the landmarks for the specified face part (nose, in this case)
    try:
        for landmark in landmarks[face_parts[face_part]]:
            # Draw a small green circle at each landmark coordinate
            cv2.circle(image, (landmark[0], landmark[1]), 3, (0, 255, 0), -1)  # Circle parameters: center, radius, color, thickness
    except KeyError:
        # If the landmark for the specified part is not found, skip drawing
        pass

    # Display the name of the facial feature being detected (e.g., "nose_landmarks") on the frame
    cv2.putText(image, f"{face_parts[face_part]}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
    # cv2.putText parameters: image, text, position, font, font size, color, thickness

    # Show the modified frame with the detected landmarks in a window titled "Image"
    cv2.imshow("Image", image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()