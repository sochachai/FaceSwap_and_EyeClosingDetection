import cv2
import dlib
import numpy as np

img = cv2.imread("bradley_cooper.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray) # 0 means black

img2 = cv2.imread("jim_carrey.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def extract_index_from_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

faces = detector(img_gray)
for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x,y))
        #cv2.circle(img,(x,y),3,(0,0,225),-1)
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    #cv2.polylines(img,[convexhull], True, (255,0,0),3) # draws the boundary of convex hull on the image
    cv2.fillConvexPoly(mask, convexhull, 255)
    face_image_1 = cv2.bitwise_and(img,img,mask=mask)
    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    #(x,y,w,h) = rect
    #cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0))
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype = np.int32) # gives the coordinates of the triangles, 6 integers
    #print(triangles)

    indices_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_from_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_from_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_from_nparray(index_pt3)
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indices_triangles.append(triangle)
    #print(indices_triangles)

# Find landmarks on the second face
faces2 = detector(img2_gray)
for face in faces2:
    landmarks2 = predictor(img2_gray, face)
    landmarks_points2 = []
    for n in range(68):
        x = landmarks2.part(n).x
        y = landmarks2.part(n).y
        landmarks_points2.append((x,y))
        #cv2.circle(img2,(x,y),3,(0,0,225),-1)

# Triangulation of both faces
for triangle_index in indices_triangles[0:1]:
    # Triangulation on second face
    tr1_pt1 = landmarks_points[triangle_index[0]]
    tr1_pt2 = landmarks_points[triangle_index[1]]
    tr1_pt3 = landmarks_points[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    rect1 = cv2.boundingRect(triangle1)
    (x,y,w,h) = rect1
    cropped_triangle = img[y:y+h, x:x+w]
    #cropped_tr1_mask = np.zeros_like(cropped_triangle) #black image
    cropped_tr1_mask = np.zeros((h,w), np.uint8)
    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1]-y],
                      [tr1_pt2[0] - x, tr1_pt2[1]-y],
                      [tr1_pt3[0] - x, tr1_pt3[1]-y]], np.int32)
    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
    cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)
    # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1) draw the rectangle
    cv2.line(img, tr1_pt1, tr1_pt2, (0, 0, 255), 1)
    cv2.line(img, tr1_pt2, tr1_pt3, (0, 0, 255), 1)
    cv2.line(img, tr1_pt3, tr1_pt1, (0, 0, 255), 1)

    # Triangulation on second face
    tr2_pt1 = landmarks_points2[triangle_index[0]]
    tr2_pt2 = landmarks_points2[triangle_index[1]]
    tr2_pt3 = landmarks_points2[triangle_index[2]]
    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
    rect2 = cv2.boundingRect(triangle2)
    (x, y, w, h) = rect2
    cropped_triangle2 = img2[y:y + h, x:x + w]
    cropped_tr2_mask = np.zeros((h, w), np.uint8)
    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                       [tr2_pt2[0] - x, tr2_pt2[1] - y],
                       [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
    cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_tr2_mask)
    cv2.line(img2, tr2_pt1, tr2_pt2, (0, 0, 255), 1)
    cv2.line(img2, tr2_pt2, tr2_pt3, (0, 0, 255), 1)
    cv2.line(img2, tr2_pt3, tr2_pt1, (0, 0, 255), 1)

    # Warp triangles
    points =  np.float32(points)
    points2 = np.float32(points2)
    M = cv2.getAffineTransform(points, points2)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w,h))




#cv2.imshow("Zidane", img)
#cv2.imshow("Ronaldinho", img2)
cv2.imshow("cropped triangle 1", cropped_triangle)
cv2.imshow("cropped triangle 2", cropped_triangle2)
cv2.imshow("Warped triangle", warped_triangle)
#cv2.imshow("mask cropped triangle", cropped_tr1_mask)

#cv2.imshow("jim", imgq)
#cv2.imshow("face_1", face_image_1)
#cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
