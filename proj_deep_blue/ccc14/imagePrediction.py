import cv2
import os
import time
import numpy as np

YOLO_V4_PATH = os.path.join(os.getcwd(), "yolo_v4")
CONF_THRES = 0.25  # minimum probability to filter weak detections
NMS_THRES = 0.30   # threshold when applyong non-maxima suppression
LABELS = "obj.names"
CFG = "karan_custom_23_01_21.cfg"
WEIGHTS = "karan_custom_best_23_01_21.weights"

class ImagePrediction:
    def __init__(self):
        self.predictedPersonCount = 0
        self.predictedMannequinCount = 0
        self.labels = self.get_labels(LABELS)
        self.cfg = self.get_config(CFG) # config file path
        self.weights = self.get_weights(WEIGHTS) # weights file path
        self.net = self.load_model()
        self.colors = self.get_colors()

    def getPredictedPersonCount(self):
        return self.predictedPersonCount

    def setPredictedPersonCount(self, personCount):
        self.predictedPersonCount = personCount

    def getPredictedMannequinCount(self):
        return self.predictedMannequinCount

    def setPredictedMannequinCount(self, mannequinCounts):
        self.predictedMannequinCount = mannequinCounts
    
    def get_predection(self, imagePath):
        imagePath = imagePath.replace('\\','/')
        image = cv2.imread(imagePath)
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        # print(layerOutputs)
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
                # print(scores)
                classID = np.argmax(scores)
                # print(classID)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONF_THRES:
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
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRES,
                                NMS_THRES)

        person_counter = 0
        mannequin_count = 0
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}".format(self.labels[classIDs[i]])
                # print(boxes)
                # print(classIDs)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                if (self.labels[classIDs[i]] == "Person"):
                    person_counter += 1
                if(self.labels[classIDs[i]] == "Mannequin"):
                    mannequin_count += 1

        person_count_txt = "Person Count: {}".format(person_counter)
        mannequin_count_txt = "Mannequin Count: {}".format(mannequin_count)

        self.setPredictedPersonCount(person_counter)
        self.setPredictedMannequinCount(mannequin_count)

        cv2.putText(image,person_count_txt, (2,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
        cv2.putText(image,mannequin_count_txt, (2,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
        
        cv2.imwrite(imagePath, image)

    def get_labels(self, labels_path):
        # load the class labels our YOLO model was trained on.
        labels_path = os.path.sep.join([YOLO_V4_PATH, labels_path])
        labels = open(labels_path).read().strip().split("\n")
        return labels

    def get_config(self, config_path):
        # path to the YOLO config file
        config_path = os.path.sep.join([YOLO_V4_PATH, config_path])
        return config_path

    def get_weights(self, weights_path):
        # derive the paths to the YOLO weights
        weights_path = os.path.sep.join([YOLO_V4_PATH, weights_path])
        return weights_path

    def load_model(self):
        # load our YOLO object detector
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn_DetectionModel(self.cfg, self.weights)
        return net

    # [TODO]: Fix a color for Person and Mannequin so that there is mo need to import numpy
    def get_colors(self):
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(self.labels), 3),dtype="uint8")
        return COLORS
