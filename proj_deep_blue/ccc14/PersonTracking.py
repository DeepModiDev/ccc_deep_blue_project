import cv2
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
import time
from queue import Queue
from threading import Thread, Event
from .models import DetectionVideos
confthres = 0.30 # confidence threshold value
nmsthres = 0.30
yolo_path = os.path.join(os.getcwd(), "yolo_v4")
import datetime
from .centroidtracker import CentroidTracker
from .trackableobject import TrackableObject

# karan_custom_23_01_21.cfg
# karan_custom_best_23_01_21.weights
# obj.names

# 16_02_2021/proj_deep_blue_16_02_2021_names.names
# 16_02_2021/proj_deep_blue_16_02_2021_cfg_upto_3000.cfg
# 16_02_2021/proj_deep_blue_16_02_2021_cfg_best.weights

class PersonTracking:

    def __init__(self):
        self.feedURL = ""
        self.videoURL = ""
        self.videoTitle = ""
        self.detectedVideoUrl = ""
        self.newName = ""
        self.userId = None
        self.frameQueue = Queue()
        self.detectionQueue = Queue()
        self.finalQueue = Queue()
        self.labelsPath = "11_03_2021/obj.names"
        self.cfgpath = "11_03_2021/custom.cfg"
        self.wpath = "11_03_2021/custom_best.weights"
        self.Lables = self.get_labels(self.labelsPath)
        self.CFG = self.get_config(self.cfgpath)
        self.Weights = self.get_weights(self.wpath)
        self.nets = self.load_model(self.CFG, self.Weights)
        self.Colors = self.get_colors(self.Lables)

    def setDetectedVideoUrl(self,detectedVideoUrl):
        self.detectedVideoUrl = detectedVideoUrl

    def getDetectedVideoUrl(self):
        return self.detectedVideoUrl

    def setfeedURL(self, feedURL):
        self.feedURL = feedURL

    def setvideoURL(self, videoURL):
        self.videoURL = videoURL

    def setuserId(self, userId):
        self.userId = userId

    def getuserId(self):
        return self.userId

    def getVideoURL(self):
        return self.videoURL

    def getfeedURL(self):
        return self.feedURL

    def setVideoTitle(self, videoTitle):
        self.videoTitle = videoTitle

    def getVideoTitle(self):
        return self.videoTitle

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

    def caller(self):
        videoUrl = self.getVideoURL().replace('\\','/')
        video = cv2.VideoCapture(videoUrl)
        currentTime = datetime.datetime.now()
        newName = str(currentTime.day)+"_"+str(currentTime.month)+"_"+str(currentTime.year)+"_"+str(currentTime.second)+"_"+self.getVideoTitle()
        # http://cam6284208.miemasu.net/nphMotionJpeg?Resolution=640x480&Quality=Clarity
        out = cv2.VideoWriter('media/videos/detections/'+newName,-1,30.0, (1080,720))

        totalUp = 0
        totalDown = 0
        frame_counter = 0
        rects = []

        ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        trackableObjects = {}

        W = None
        H = None

        while(True):
            check, frame = video.read()
            if check != False:
                frame = cv2.resize(frame,(1080,720))

                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                cv2.line(frame, (0, H // 2), (W, H // 2), (255, 255, 0), 2)

                if frame_counter == 0:
                    cv2.imwrite("media/videos/detections/thumbnails/"+newName.split('.')[0]+".jpg", frame)

                if frame_counter % 10 == 0:
                    status = "Detecting"
                    frame, rects = self.get_prediction(frame, self.nets, W, H, self.Colors, self.Lables)

                else:
                    status = "Tracking"
                    frame, trackableObjects, totalUp, totalDown = self.tracking(frame, rects, ct, H, trackableObjects,
                                                                           totalUp, totalDown)

                text_up_person_count = "Up: " + str(totalUp)
                text_down_person_count = "Down: " + str(totalDown)
                text_status = "Down: " + str(status)
                cv2.putText(frame, text_up_person_count, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, text_down_person_count, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, text_status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                frame_counter += 1

                out.write(frame)
            else:
                break

        video.release()
        out.release()
        print("Video title issssssssss:",self.getVideoTitle())
        detectedVideo = DetectionVideos(videoTitle=newName,user_id=self.getuserId(),video=os.path.join('videos/detections/',newName),thumbnail="videos/detections/thumbnails/"+newName.split('.')[0]+".jpg",date=datetime.datetime.now())
        detectedVideo.save()
        self.setDetectedVideoUrl(detectedVideo.video)

    def feedVideo(self):
        video = cv2.VideoCapture(self.getfeedURL())
        # http://cam6284208.miemasu.net/nphMotionJpeg?Resolution=640x480&Quality=Clarity
        start_time = time.time()
        total_frames = 0
        frame_count = 0
        person_count = 0

        if self.getVideoTitle() is not None:
            frameOut = cv2.VideoWriter("media/videos/" + str(time.time()) + ".mp4", -1, 20.0, (1080, 720))
        else:
            frameOut = cv2.VideoWriter(time.time(), -1, 20.0, (1080, 720))

        while (True):
            check, image = video.read()
            image = cv2.resize(image, (1080, 720))
            #res = self.get_predection(image, self.nets, self.Lables, self.Colors)
            total_frames = total_frames + 1

            if(frame_count%15 == 0):
                image,person_count = self.get_predection(image, self.nets, self.Lables, self.Colors)
            frame_count = frame_count + 1

            end_time = time.time()
            time_diff = end_time - start_time
            if time_diff == 0:
                fps = 0
            else:
                fps = total_frames/time_diff
            fps_text = "FPS: {:.2f}".format(fps)

            cv2.putText(image, fps_text, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            count_txt = "Person Count: {}".format(person_count)
            cv2.putText(image,count_txt, (10,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)

            cv2.imshow("capture", image)
            frameOut.write(image)
            key = cv2.waitKey(1)
            if (key == ord('q')):
                break

        video.release()
        cv2.destroyAllWindows()

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

    def tracking(self,frame, rects, ct, H, trackableObjects, totalUp, totalDown):
        objects = ct.update(rects)
        live_person_count = len(objects)
        for (objectID, centroid) in objects.items():

            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        to.counted = True

                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        to.counted = True

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            #cv2.rectangle(frame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (255, 0, 0), 2)

        return frame, trackableObjects, totalUp, totalDown
