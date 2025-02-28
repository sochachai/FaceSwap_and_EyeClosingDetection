import cv2
import dlib
import numpy as np

name = 'Zidane'
img = cv2.imread(name+".jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)  # 0 means black
cv2.imshow('img',img)

img2 = cv2.imread("So_mouth_closed.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2_new_face = np.zeros_like(img2)

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
        landmarks_points.append((x, y))
        # cv2.circle(img,(x,y),3,(0,0,225),-1)
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    # cv2.polylines(img,[convexhull], True, (255,0,0),3) # draws the boundary of convex hull on the image
    cv2.fillConvexPoly(mask, convexhull, 255)
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)
    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    # (x,y,w,h) = rect
    # cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0))
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)  # gives the coordinates of the triangles, 6 integers
    # print(triangles)

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
    # print(indices_triangles)

# Find landmarks on the second face
faces2 = detector(img2_gray)
for face in faces2:
    landmarks2 = predictor(img2_gray, face)
    landmarks_points2 = []
    for n in range(68):
        x = landmarks2.part(n).x
        y = landmarks2.part(n).y
        landmarks_points2.append((x, y))
        # cv2.circle(img2,(x,y),3,(0,0,225),-1)

# Triangulation of both faces
for triangle_index in indices_triangles:
    # Triangulation on second face
    tr1_pt1 = landmarks_points[triangle_index[0]]
    tr1_pt2 = landmarks_points[triangle_index[1]]
    tr1_pt3 = landmarks_points[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    cropped_triangle = img[y:y + h, x:x + w]
    # cropped_tr1_mask = np.zeros_like(cropped_triangle) #black image
    cropped_tr1_mask = np.zeros((h, w), np.uint8)
    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
    cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)
    # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1) draw the rectangle
    #cv2.line(img, tr1_pt1, tr1_pt2, (0, 0, 255), 1)
    #cv2.line(img, tr1_pt2, tr1_pt3, (0, 0, 255), 1)
    #cv2.line(img, tr1_pt3, tr1_pt1, (0, 0, 255), 1)

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
    #cv2.line(img2, tr2_pt1, tr2_pt2, (0, 0, 255), 1)
    #cv2.line(img2, tr2_pt2, tr2_pt3, (0, 0, 255), 1)
    #cv2.line(img2, tr2_pt3, tr2_pt1, (0, 0, 255), 1)

    # Warp triangles
    points = np.float32(points)
    points2 = np.float32(points2)
    M = cv2.getAffineTransform(points, points2)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))

    # Reconstruct output face by stitching the warped triangles
    triangle_area = img2_new_face[y:y+h, x:x+w]
    triangle_area = cv2.add(triangle_area, warped_triangle)
    #cv2.imshow("tri", triangle_area)
    #cv2.waitKey(0)
    img2_new_face[y:y+h, x:x+w] = triangle_area


# Face swap: map face1 to face2
img2_new_face_gray = cv2.cvtColor(img2_new_face,cv2.COLOR_BGR2GRAY)
_, background = cv2.threshold(img2_new_face_gray, 0, 255, cv2.THRESH_BINARY_INV)
background = cv2.bitwise_and(img2, img2, mask=background)
result = cv2.add(background, img2_new_face)

#print(landmarks_points2)

img2_face_mask = np.zeros_like(img2_new_face_gray)
points2 = np.array(landmarks_points2, np.int32)
convexhull2 = cv2.convexHull(points2)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
#blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0) # Adjust kernel size as needed
(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
#seamlessclone = cv2.seamlessClone(seamlessclone, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)





# Setting parameter values
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold

# Applying the Canny Edge filter
edge = cv2.Canny(seamlessclone, t_lower, t_upper)

#img_gray = cv2.cvtColor(seamlessclone, cv2.COLOR_BGR2GRAY)
#seamlessclone2 = cv2.seamlessClone(seamlessclone, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
#blurred_image = cv2.GaussianBlur(seamlessclone, (5,5),0)
#blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0) # Adjust kernel size as needed
#output = cv2.seamlessClone(src, dst, blurred_mask, center, cv2.NORMAL_CLONE)

# cv2.imshow("Zidane", img)
# cv2.imshow("Ronaldinho", img2)
#cv2.imshow("cropped triangle 1", cropped_triangle)
#cv2.imshow("cropped triangle 2", cropped_triangle2)
#cv2.imshow("Warped triangle", warped_triangle)
#cv2.imshow("cropped_tr2_mask", cropped_tr2_mask)
cv2.imshow("Image 2 new face", img2_new_face)
#cv2.imshow("background",background)
cv2.imshow("NewDihno", result)
cv2.imshow("seamless", seamlessclone)
cv2.imshow("edge", edge)


########
#img_no_triangles = seamlessclone.copy()
for face in faces2:
    landmarks2 = predictor(img2_gray, face)
    landmarks_points2 = []
    for n in range(68):
        x = landmarks2.part(n).x
        y = landmarks2.part(n).y
        landmarks_points2.append((x, y))
    points = np.array(landmarks_points2, np.int32)
    # Create Delaunay triangulation
    rect = (0, 0, img2_gray.shape[1], img2_gray.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points2)
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        index_pt1 = np.where((points == pt1).all(axis=1))[0][0]
        index_pt2 = np.where((points == pt2).all(axis=1))[0][0]
        index_pt3 = np.where((points == pt3).all(axis=1))[0][0]
        delaunayTri.append([index_pt1, index_pt2, index_pt3])

    # Create a mask for the triangles to remove (e.g., the first triangle)
    mask = np.zeros_like(img2_gray)
    #white_mask = np.full(mask.shape, 120, dtype=np.uint8)
    for i in range(len(delaunayTri)):
        #print(delaunayTri[i])
        triangle_to_remove = delaunayTri[i]
        pt1 = points[triangle_to_remove[0]]
        pt2 = points[triangle_to_remove[1]]
        pt3 = points[triangle_to_remove[2]]
        pts = np.array([pt1, pt2, pt3],np.int32)
        #print(pts)
        cv2.line(mask, pt1, pt2, 255, 1)
        cv2.line(mask, pt2, pt3, 255, 1)
        cv2.line(mask, pt3, pt1, 255, 1)
        #cv2.fillConvexPoly(mask, pts, 255)
    # Inpaint the masked region
    #seamlessclone_gray = cv2.cvtColor(seamlessclone, cv2.COLOR_BGR2GRAY)
    #img_no_triangles = cv2.inpaint(seamlessclone, mask, 1000, cv2.INPAINT_TELEA)
    #alpha = 0.5
    #beta = 1 - alpha
    #img_no_triangles = cv2.addWeighted(seamlessclone_gray, alpha, white_mask, beta, 0.0)
    # Remove small noise
    inp_mask = cv2.morphologyEx(mask,
                                cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))

    # Dilate mask
    inp_mask = cv2.dilate(inp_mask,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))
    img_no_triangles = cv2.inpaint(result, inp_mask, 10, cv2.INPAINT_TELEA) #0.1
########

img2_face_mask = np.zeros_like(img2_new_face_gray)
points2 = np.array(landmarks_points2, np.int32)
convexhull2 = cv2.convexHull(points2)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
#blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0) # Adjust kernel size as needed
(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
#seamlessclone2 = cv2.seamlessClone(img_no_triangles, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
median = cv2.medianBlur(img_no_triangles, 3) # 5
seamlessclone2 = cv2.seamlessClone(median, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)


#median_inpaint1 = cv2.medianBlur(seamlessclone, 15)
#median_inpaint2 = cv2.inpaint(median_inpaint1, mask, 100, cv2.INPAINT_NS)
#gaussian = cv2.GaussianBlur(seamlessclone2, (7, 7), 0)
#seamlessclone3 = cv2.seamlessClone(seamlessclone2, median, img2_head_mask, center_face2, cv2.MIXED_CLONE)
#seamlessclone4 = cv2.seamlessClone(seamlessclone2, gaussian, img2_head_mask, center_face2, cv2.MIXED_CLONE)
#dst = cv2.fastNlMeansDenoisingColored(img_no_triangles, None, 10, 10, 7, 21)


cv2.imshow('mask',mask)
cv2.imshow('inp_mask',inp_mask)
cv2.imshow("inpainted_img", img_no_triangles)
cv2.imshow("median", median)
cv2.imshow(name, seamlessclone2)
#cv2.imshow("median_inpaint1", median_inpaint1)
#cv2.imshow("median_inpaint2", median_inpaint2)
#cv2.imshow("gaussian", gaussian)
#cv2.imshow("seamlessclone3", seamlessclone3)
#cv2.imshow("seamlessclone4", seamlessclone4)
#cv2.imshow("dst", dst)

#inpainted_img2 = cv2.seamlessClone(img_no_triangles, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
#cv2.imshow("inpainted_img2", inpainted_img2)
#cv2.imshow("seamless2", seamlessclone2)
#cv2.imshow("blurred_image", blurred_image)
# cv2.imshow("mask cropped triangle", cropped_tr1_mask)

# cv2.imshow("jim", img)
# cv2.imshow("face_1", face_image_1)
# cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
