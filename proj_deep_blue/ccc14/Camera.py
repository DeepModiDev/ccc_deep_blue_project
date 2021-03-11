import cv2
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
import time
confthres = 0.30 # confidence threshold value
nmsthres = 0.30
yolo_path = os.path.join(os.getcwd(), "yolo_v4")
from .centroidtracker import CentroidTracker


class VideoCamera(object):
    def __init__(self,url):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(url)
        self.ct = CentroidTracker(maxDisappeared=30, maxDistance=70)
        self.total_person_count = 0
        self.live_person_count = 0
        self.frame_counter = 0
        self.rects = []
        self.person_id_list = []
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        self.labelsPath = "11_03_2021/obj.names"
        self.cfgpath = "11_03_2021/custom.cfg"
        self.wpath = "11_03_2021/custom_best.weights"
        self.Lables = self.get_labels(self.labelsPath)
        self.CFG = self.get_config(self.cfgpath)
        self.Weights = self.get_weights(self.wpath)
        self.nets = self.load_model(self.CFG, self.Weights)
        self.Colors = self.get_colors(self.Lables)

    def get_colors(self, LABELS):
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
        return COLORS

    def get_weights(self, weights_path):
        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([yolo_path, weights_path])
        return weightsPath

    def get_config(self, config_path):
        configPath = os.path.sep.join([yolo_path, config_path])
        return configPath

    def load_model(self, configpath, weightspath):
        # load our YOLO object detector trained on COCO dataset (80 classes)
        # print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return net

    def get_labels(self, labels_path):
        # load the COCO class labels our YOLO model was trained on
        # labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
        lpath = os.path.sep.join([yolo_path, labels_path])
        LABELS = open(lpath).read().strip().split("\n")
        return LABELS

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        W = None
        H = None

        if success != False:
            image = cv2.resize(image, (1080, 720))

            if W is None or H is None:
                (H, W) = image.shape[:2]

            if (self.frame_counter % 15 == 0):
                (image, self.rects) = self.get_prediction(image, self.nets, W, H, self.Colors, self.Lables)
            else:
                (image, self.person_id_list, self.live_person_count, self.total_person_count) = self.tracking(image)

            text_live_person_count = "Live Person Count: " + str(self.live_person_count)
            text_total_person_count = "Total Person Count: " + str(self.total_person_count)

            cv2.putText(image, text_live_person_count, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, text_total_person_count, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.frame_counter += 1

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def get_prediction(self, frame, net, W, H, COLORS, LABELS):

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        rects = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                rects.append([x, y, x + w, y + h])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, rects

    def tracking(self,frame):
        objects = self.ct.update(self.rects)
        self.live_person_count = len(objects)
        for (objectID, centroid) in objects.items():
            if objectID not in self.person_id_list:
                self.person_id_list.append(objectID)
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            #cv2.rectangle(frame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (255, 0, 0), 2)

        return frame, self.person_id_list, self.live_person_count, len(self.person_id_list)