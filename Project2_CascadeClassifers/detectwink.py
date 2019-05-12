import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import sys

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
LINE_TYPE = 2

FACE_SCALE = 1.05
FACE_MIN_NB = 4
FACE_FLAGS = 0|cv2.CASCADE_SCALE_IMAGE
FACE_MINSIZE = (40, 40)

# Black face for eyes
EYE_SCALE = 1.15
EYE_MIN_NB = 4
EYE_FLAGS = 0 | cv2.CASCADE_SCALE_IMAGE
EYE_MINSIZE = (10, 10)
EYE_MAXSIZE = (150, 150)

EYE_SCALE2 = 1.02
EYE_MIN_NB2 = 1
EYE_FLAGS2 = 0 | cv2.CASCADE_SCALE_IMAGE
EYE_MINSIZE2 = (10, 10)
EYE_MAXSIZE2 = (150, 150)

SMALL = 0
LARGE = 1

def preprocessing_faces(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(3,3))
    img = clahe.apply(img)

    # Other poor preprocessing possibilities:
    #img = cv2.equalizeHist(img) # poor result, too much contrast
    #img = cv2.GaussianBlur(img, (5, 5), 0) # Good for too sharp image
    #img = cv2.medianBlur(img, 5) # Good for salt-n-pepper
    #img = cv2.bilateralFilter(img, 9, 75, 75)
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return img

def preprocessing_eyes(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5,5))
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(7,7))
    img = clahe.apply(img)
    return img

def preprocessing_eyes2(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(6,6))
    img = clahe.apply(img)
    return img

def isOverlapping(x1, y1, w1, h1, x2, y2, w2, h2):
    # Rectangle 1
    topleft_x1 = x1
    topleft_y1 = y1
    bottomright_x1 = x1 + w1
    bottomright_y1 = y1 + h1

    # Rectangle 2
    topleft_x2 = x2
    topleft_y2 = y2
    bottomright_x2 = x2 + w2
    bottomright_y2 = y2 + h2

    # First check if rectangles are adjacent but not overlapping
    if topleft_x1 > bottomright_x2 or bottomright_x1 < topleft_x2:
        return False
    # Then check if rectangles are above/below but not overlapping
    elif bottomright_y1 < topleft_y2 or topleft_y1 > bottomright_y2:
        return False
    else:
        return True

def removeDups(boxes, keep=SMALL):

    unique = []
    while len(boxes) > 0:
        box = boxes.pop(0)

        overlapping = False
        for i in range(len(unique)):
            if isOverlapping(box[0],  # x1
                             box[1],  # y1
                             box[2],  # width1
                             box[3],  # height1
                             unique[i][0],  # x2
                             unique[i][1],  # y2
                             unique[i][2],  # width2
                             unique[i][3]): # height2
                    overlapping = True

                    if keep == SMALL:
                        if box[2]*box[3] < unique[i][2]*unique[i][3]:

                            unique[i] = box
                    else:

                        if box[2]*box[3] > unique[i][2]*unique[i][3]:
                            unique[i] = box
                    break
        # If it is overlapping, drop the box
        # Otherwise, add it to the unique boxes
        if not overlapping:
            unique.append(box)
    return unique

# Detect Eyes
def detect_eyes(frame, location, ROI, cascade):

    x = location[0]
    y = location[1]
    w = location[2]
    h = location[3]

    ROI1 = preprocessing_eyes(ROI)

    # Detect Multi-scale for EYES
    eyes = cascade.detectMultiScale(
        ROI1,
        EYE_SCALE,
        EYE_MIN_NB,
        EYE_FLAGS,
        EYE_MINSIZE,
        EYE_MAXSIZE)

    if len(eyes) > 0:
        eyes = removeDups(eyes.tolist(), keep=SMALL)
    else:
        eyes = []

    midx = x + int(round(w*0.5))
    midy = y + int(round(h*0.4))
    cv2.putText(frame, ".", (midx, midy), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)

    count = 0
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]

        if y < midy:
            print("y: ", y)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
            count += 1


    if count > 0:
        return count == 1    # number of eyes is one

    ROI2 = preprocessing_eyes2(ROI)
    # Detect Multi-scale for EYES
    eyes = cascade.detectMultiScale(
        ROI2,
        EYE_SCALE2,
        EYE_MIN_NB2,
        EYE_FLAGS2,
        EYE_MINSIZE2,
        EYE_MAXSIZE2)

    if len(eyes) > 0:
        eyes = removeDups(eyes.tolist(), keep=SMALL)
    else:
        eyes = []

    cv2.putText(frame, ".", (midx, midy), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)

    count = 0
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]

        if y < midy:
            print("y: ", y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            count += 1

    return count == 1

# Detect Faces
def detect_faces(frame, faceCascade, eyesCascade):
    gray_frame = preprocessing_faces(frame)

    faces = faceCascade.detectMultiScale(
        gray_frame, 
        FACE_SCALE,
        FACE_MIN_NB,
        FACE_FLAGS,
        FACE_MINSIZE)

    detected = 0
    if len(faces) > 0:
        faces = removeDups(faces.tolist(), keep=LARGE)
    else:
        faces = []

    # For each of the detected faces, process the region of interest
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        # Define the region of interest
        faceROI = frame[y:y+h, x:x+w]

        if isWinking(frame, x, y, w, h, faceROI, eyesCascade):
             detected = detected + 1

    return detected, len(faces)

# Check if the region of interest is a wink or simply a face
def isWinking(frame, x, y, w, h, faceROI, eyesCascade):

    if detect_eyes(frame, (x, y, w, h), faceROI, eyesCascade):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return True
    else:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        return False

def run_on_folder(cascade1, cascade2, folder):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]

    windowName = None
    winks = 0
    faces = 0
    for f in files:
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            ctwink, ctface = detect_faces(img, cascade1, cascade2)
            winks += ctwink
            faces += ctface
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return winks, faces

def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video 1"
    showlive = True

    while(showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect_faces(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)

        if cv2.waitKey(30) >= 0:
            showlive = False
    
    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if(len(sys.argv) == 2): # one argument
        folderName = sys.argv[1]
        winks, faces = run_on_folder(face_cascade, eye_cascade, folderName)
        print("Total number of faces: ", faces)
        print("Total number of winks: ", winks)
    else: # no arguments
        runonVideo(face_cascade, eye_cascade)

