import numpy as np
import cv2
import glob


def get_points(image_dir, chessboard_dims):
    # Creating vector to store vectors of 3D points for each chessboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each chessboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, chessboard_dims[0] * chessboard_dims[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:chessboard_dims[0], 0:chessboard_dims[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob(
        image_dir + "/*.png")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, chessboard_dims, None)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of chess board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), None)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, chessboard_dims, corners2, ret)

        cv2.imshow(fname, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return objpoints, imgpoints, gray.shape[::-1]

chessboard_dims = (7, 11)
image_dir="Week 1/Homework/chessboard_images"
image_path="Week 1/Homework/chessboard_images/Im_L_1.png"
objpoints, imgpoints, gray_shape = get_points(image_dir, chessboard_dims=chessboard_dims)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

img = cv2.imread(image_path)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imshow("Calibration Result", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Pose estimation
def draw_lines(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

objp = np.zeros((7*11,3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_dims[0], 0:chessboard_dims[1]].T.reshape(-1, 2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)


for fname in glob.glob('Week 1/Homework/chessboard_images/*.png'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray,
                                             chessboard_dims,
                                             None
                                             )
    if ret:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), None)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # Convert float np array to int

        corners2 = np.int32(corners2)
        imgpts = np.int32(imgpts)
        img = draw_lines(img, corners2, imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            cv2.imwrite(fname[:6] + "_pose"+'.png', img)


# Render a cube on the image
def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])


objp = np.zeros((7*11,3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_dims[0], 0:chessboard_dims[1]].T.reshape(-1, 2)

for fname in glob.glob('Week 1/Homework/chessboard_images/*.png'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray,
                                             chessboard_dims,
                                             None
                                             )
    if ret:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), None)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # Convert float np array to int

        corners2 = np.int32(corners2)
        imgpts = np.int32(imgpts)
        img = draw_cube(img, corners2, imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            cv2.imwrite(fname[:6] + "_pose"+'.png', img)
