# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:42:05 2021

@author: Karansinh Padhiar
"""

# USAGE
# python selectROI.py --image F:/Project Images/Project Deep Blue/karan_select_roi --yolo G:/CHANGA/SEM 5/Project Deep Blue/version_1.5/ccc_deep_blue_project/proj_deep_blue/yolo_v4

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#     help="path to input image FOLDER")
# ap.add_argument("-y", "--yolo", required=True,
#     help="base path to YOLO directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#     help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3,
#     help="threshold when applyong non-maxima suppression")
# args = vars(ap.parse_args())

TEST_DIR = r"F:/Project Images/Project Deep Blue/karan_select_roi/test_images"
YOLO_PATH = r"F:/Project Images/Project Deep Blue/karan_select_roi/yolo_v4"
CONFIDENCE = 0.5
NMS_THRESH = 0.3


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([YOLO_PATH, "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([YOLO_PATH, "custom_best.weights"])
configPath = os.path.sep.join([YOLO_PATH, "custom.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
#net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net = cv2.dnn_DetectionModel(configPath, weightsPath)

for imageName in os.listdir(TEST_DIR):
    # load our input image and grab its spatial dimensions
    image = cv2.imread(os.path.join(TEST_DIR, imageName))
    (H, W) = image.shape[:2]
    
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
    
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
    
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
    
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, NMS_THRESH)
    
    # Select ROI
    text = ""
    roi = None
    # text = "Press 's' to select region to be blurred..."
    # cv2.putText(image, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (0, 0, 255), 2)
    cv2.imshow("Image", image)
    key = cv2.waitKey(0)
    if key == ord('s'):
        # Select ROI
        roi = cv2.selectROI(windowName='Image', img=image, showCrosshair=True, fromCenter=False)
        image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :] = cv2.blur(src=image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :], ksize=(7, 7))
    
        # text = "Press ENTER or SPACE to continue..."
        # cv2.putText(image, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (0, 0, 255), 2)
        cv2.imshow("Image", image)
    
    elif key == ord('q'):
        cv2.destroyAllWindows()
        print("Quitting...")
        break
    else:
        cv2.destroyWindow(winname="Image")
        pass
    
    # ensure at least one detection exists
    if len(idxs) > 0:        
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            if roi is not None:
                if (x > roi[0] and y > roi[1]) and (x < roi[0]+roi[2] and y < roi[1]+roi[3]):
                    continue
            
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
    
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # save the image
    cv2.imwrite(filename=os.path.join(TEST_DIR, "output", imageName), img=image)


"""
import cv2
import numpy as np

roi = None

imagePath = r"F:/Project Images/Project Deep Blue/test_images/EnfntsTerribles-How-Uniqlo.jpg"

image = cv2.imread(imagePath)
print("Image.shape = ", image.shape)

cv2.imshow(winname="image", mat=image)
key = cv2.waitKey(0)
if key == ord('s'):
    print("Select ROI")
    roi = cv2.selectROI(windowName='image', img=image, showCrosshair=True, fromCenter=False)
    print("ROI = ", roi)
    
    pt1 = (roi[0], roi[1])
    pt2 = (roi[0]+roi[2], roi[1]+roi[3])
    color=(0, 0, 255)
    cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=color, thickness=2)
    cv2.imshow(winname="image", mat=image)
    
    image[roi[1]:, roi[0]:, :] = cv2.blur(src=image[roi[1]:, roi[0]:, :], ksize=(7, 7))
    cv2.imshow(winname='blurred', mat=image)
    
    if cv2.waitKey(0) == ord('q'):
        print("Quit")
        cv2.destroyAllWindows()

if key == ord('q'):
    print("Quit")
cv2.destroyAllWindows()
"""