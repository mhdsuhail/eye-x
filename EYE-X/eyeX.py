from imutils import face_utils
from allFunctions import *
import numpy as np
import pyautogui as pag
import imutils
import dlib
import cv2

# initialise aspect ratio threshold values
MOUTH_AR_THRESH = 0.4
EYE_AR_THRESH = 0.20
WINK_AR_DIFF_THRESH = 0.01
WINK_AR_CLOSE_THRESH = 0.19

# initialise consecutive frame counters
MOUTH_AR_CONSECUTIVE_FRAMES = 15
EYE_AR_CONSECUTIVE_FRAMES = 15
WINK_CONSECUTIVE_FRAMES = 10

# all counters are initialised
input_run = 0
scroll_run = 0
mouth_counter = 0
eye_counter = 0
wink_counter = 0

# initially all mode's are deactivated
INPUT_MODE = False
EYE_CLICK = False
LEFT_WINK = False
RIGHT_WINK = False
SCROLL_MODE = False

# initialised color variable
LIGHT_BLUE = (176,196,222)
LAVENDER_COLOR = (230,230,250)
LINEN_COLOR = (250,240,230)
THISTLE_COLOR = (216,191,216)


# Initialize Dlib's face detector (HOG-based) and then create
# the facial landmark predictor
shape_predictor = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# Grab the indexes of the facial landmarks for the left and
# right eye, nose and mouth respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# capture the video with resolution of device screen
vid = cv2.VideoCapture(0)
SCREEN_RESOLUTION = pag.size()
cam_w = SCREEN_RESOLUTION.width
cam_h = SCREEN_RESOLUTION.height


while True:
    # Grab the frame from the threaded video file stream,
    # resize it, and convert it to grayscale channels
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    if len(rects) > 0:
        rect = rects[0]
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue

    # Determine the facial landmarks for the face region, then convert
    # the facial landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    mouth = shape[mStart:mEnd]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    nose = shape[nStart:nEnd]

    # Because of we flipped the frame, left is right, right is left.
    temp = leftEye
    leftEye = rightEye
    rightEye = temp

    # Average the mouth aspect ratio together for both eyes
    mar = mouth_aspect_ratio(mouth)
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    diff_ear = np.abs(leftEAR - rightEAR)

    #initialize the nose point
    nose_point = (nose[3, 0], nose[3, 1])

    # Compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [mouthHull], -1, LINEN_COLOR, 1)
    cv2.drawContours(frame, [leftEyeHull], -1, LINEN_COLOR, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, LINEN_COLOR, 1)


    # Check to see if the eye aspect ratio is below the blink
    # threshold, and if so, increment the blink frame counter
    if diff_ear > WINK_AR_DIFF_THRESH:

        if leftEAR < rightEAR:
            if leftEAR < EYE_AR_THRESH:
                wink_counter += 1

                if wink_counter > WINK_CONSECUTIVE_FRAMES:
                    pag.click(button='left')

                    wink_counter = 0

        elif leftEAR > rightEAR:
            if rightEAR < EYE_AR_THRESH:
                wink_counter += 1

                if wink_counter > WINK_CONSECUTIVE_FRAMES:
                    pag.click(button='right')

                    wink_counter = 0
        else:
            wink_counter = 0
    else:
        if ear <= EYE_AR_THRESH:
            eye_counter += 1

            if eye_counter > EYE_AR_CONSECUTIVE_FRAMES:
                # activate and deactivate scroll mode
                SCROLL_MODE = not SCROLL_MODE

                eye_counter = 0

        else:
            eye_counter = 0
            wink_counter = 0

    if mar > MOUTH_AR_THRESH:
        mouth_counter += 1

        if mouth_counter >= MOUTH_AR_CONSECUTIVE_FRAMES:
            # activate and deactivate input mode
            INPUT_MODE = not INPUT_MODE

            mouth_counter = 0


    else:
        mouth_counter = 0
    # notification over input mode activation
    if INPUT_MODE:
        if input_run == 0:
            notify("Input mode on" ,"Cursor synched with eyes")
            input_run = 1

        # finding mid point of the screen to draw circle and line
        x = int(cam_w/2)
        y = int(cam_h/2)
        ANCHOR_POINT = (x,y)
        r = 60
        cv2.circle(frame, (x,y), r, LIGHT_BLUE, 4)
        cv2.line(frame, ANCHOR_POINT, nose_point, LAVENDER_COLOR, 4)


        # drag the mouse according to head movements
        dir = direction(nose_point, ANCHOR_POINT, r)
        drag = 10
        if dir == 'right':
            if SCROLL_MODE:
                pag.hscroll(10)
            else:
                pag.moveRel(drag, 0)
        elif dir == 'left':
            if SCROLL_MODE:
                pag.hscroll(-10)
            else:
                pag.moveRel(-drag, 0)
        elif dir == 'up':
            if SCROLL_MODE:
                pag.scroll(5)
            else:
                pag.moveRel(0, -drag)
        elif dir == 'down':
            if SCROLL_MODE:
                pag.scroll(-5)
            else:
                pag.moveRel(0, drag)

    # notification over input mode deactivation
    if not INPUT_MODE:
        if input_run ==1:
            notify("INPUT MODE OF" ,"Cursor movement released from eyes")
            input_run = 0

    # notification over scroll mode activation
    if SCROLL_MODE:
        if scroll_run == 0:
            notify("SCROLL MODE ON" ,"Scroll using yor eyes")
            scroll_run = 1

    # notification over scroll mode deactivation
    if not SCROLL_MODE:
        if scroll_run == 1:
            notify("SCROLL MODE OF" ,"Scroll synch release from eyes" )
            scroll_run = 0


    # to display the frame from webcam
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # to exit from running, uses "esc" key
    if key == 27:
        break

# remove all the widows which are created
cv2.destroyAllWindows()
vid.release()
