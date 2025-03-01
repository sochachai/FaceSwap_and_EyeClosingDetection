import cv2
import dlib
import numpy as np
import random
import threading
picture_directory = 'Pictures/'
names = ['Jackie_Cheung','Aaron_Kwok','Leon_Lai','Andy_Lau','Adam_Cheng','Michael_Hui','Chow_Runfat','Stephen_Chow']
name_index = 0
def update_global():
    global name
    global name_index
    # name = random.choice(names)
    name_index = int((name_index + 1)) % int(len(names))
    name = names[name_index]
    print(f"Updated global_variable: {name}")
    threading.Timer(5, update_global).start()

update_global()
video = cv2.VideoCapture(0)
while True:
    img = cv2.imread(picture_directory + name + ".jpg")
    cv2.imshow('img',img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)  # 0 means black

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

    ret, img2 = video.read()
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_new_face = np.zeros_like(img2)

    # Find landmarks on the second face
    faces2 = detector(img2_gray)
    if len(faces)==0:
        pass
    else:
        landmarks_points2 = []
        for face in faces2:
            landmarks2 = predictor(img2_gray, face)
            #landmarks_points2 = []
            for n in range(68):
                x = landmarks2.part(n).x
                y = landmarks2.part(n).y
                landmarks_points2.append((x, y))
        #print(landmarks_points2)

    #if len(landmarks_points2) == 0:
    #    pass
    #else:
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

        # Triangulation on second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
        #print(x,y,w,h)
        cropped_triangle2 = img2[y:y + h, x:x + w]
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
        #cv2.imshow('image', cropped_tr2_mask)
        #cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_tr2_mask)


        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))

        # Reconstruct output face by stitching the warped triangles
        triangle_area = img2_new_face[y:y+h, x:x+w]
        triangle_area = cv2.add(triangle_area, warped_triangle)
        img2_new_face[y:y+h, x:x+w] = triangle_area

    # Face swap: map face1 to face2
    img2_new_face_gray = cv2.cvtColor(img2_new_face,cv2.COLOR_BGR2GRAY)
    _, background = cv2.threshold(img2_new_face_gray, 0, 255, cv2.THRESH_BINARY_INV)
    background = cv2.bitwise_and(img2, img2, mask=background)
    result = cv2.add(background, img2_new_face)

    #for face in faces2:
    face = faces2[0]
    print(face)
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
        triangle_to_remove = delaunayTri[i]
        pt1 = points[triangle_to_remove[0]]
        pt2 = points[triangle_to_remove[1]]
        pt3 = points[triangle_to_remove[2]]
        pts = np.array([pt1, pt2, pt3],np.int32)
        cv2.line(mask, pt1, pt2, 255, 1)
        cv2.line(mask, pt2, pt3, 255, 1)
        cv2.line(mask, pt3, pt1, 255, 1)
    # Inpaint the masked region
    # Remove small noise
    inp_mask = cv2.morphologyEx(mask,
                                cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))

    # Dilate mask
    inp_mask = cv2.dilate(inp_mask,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)))
    img_no_triangles = cv2.inpaint(result, inp_mask, 10, cv2.INPAINT_TELEA)

    img2_face_mask = np.zeros_like(img2_new_face_gray)
    points2 = np.array(landmarks_points2, np.int32)
    convexhull2 = cv2.convexHull(points2)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    median = cv2.medianBlur(img_no_triangles, 3) # 5
    final_result = cv2.seamlessClone(median, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    cv2.imshow('new', final_result)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

