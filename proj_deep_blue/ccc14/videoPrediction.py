import cv2
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
import time
from queue import Queue
from threading import Thread, Event
from .models import DetectionVideos
confthres = 0.60 # confidence threshold value
nmsthres = 0.30
yolo_path = os.path.join(os.getcwd(), "yolo_v4")
import datetime
from .centroidtracker import CentroidTracker

# karan_custom_23_01_21.cfg
# karan_custom_best_23_01_21.weights
# obj.names

# 16_02_2021/proj_deep_blue_16_02_2021_names.names
# 16_02_2021/proj_deep_blue_16_02_2021_cfg_upto_3000.cfg
# 16_02_2021/proj_deep_blue_16_02_2021_cfg_best.weights

# 28_03_2021/all_img_xml_head_3717_mannequin_head_3219.cfg
# 28_03_2021/all_img_xml_head_3717_mannequin_head_3219_names.names
# 28_03_2021/all_img_xml_head_3717_mannequin_head_3219_last.weights

# 11_03_2021/custom_best.weights
# 11_03_2021/custom.cfg
# 11_03_2021/obj.names

class VideoPrediction:

    def __init__(self):
        self.feedURL = ""
        self.videoURL = ""
        self.videoTitle = ""
        self.detectedVideoUrl = ""
        self.newName = ""
        self.userId = None
        self.totalCountSet = {0}
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
        #http://cam6284208.miemasu.net/nphMotionJpeg?Resolution=640x480&Quality=Clarity
        out = cv2.VideoWriter('media/videos/detections/'+newName,-1,30.0, (1080,720))

        total_person_count = 0
        live_person_count = 0
        frame_counter = 0
        rects = []
        person_id_list = []

        ct = CentroidTracker(maxDisappeared=30, maxDistance=70)

        W = None
        H = None

        while(True):
            check, image = video.read()
            if check != False:
                image = cv2.resize(image,(1080,720))

                if W is None or H is None:
                    (H, W) = image.shape[:2]

                if frame_counter == 0:
                    cv2.imwrite("media/videos/detections/thumbnails/"+newName.split('.')[0]+".jpg", image)

                if(frame_counter%5 == 0):
                    image, rects = self.get_prediction(image, self.nets, W ,H , self.Colors, self.Lables)
                else:
                    image, person_id_list, live_person_count, total_person_count = self.tracking(image, rects, ct, person_id_list)

                text_live_person_count = "Live Person Count: " + str(live_person_count)
                text_total_person_count = "Total Person Count: " + str(total_person_count)
                self.totalCountSet.add(live_person_count)
                cv2.putText(image, text_live_person_count, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
                #cv2.putText(image, text_total_person_count, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                frame_counter += 1
                out.write(image)
            else:
                break

        video.release()
        out.release()

        max_count = max(self.totalCountSet)
        average = 0
        i = 0
        sorted_set = sorted(self.totalCountSet)
        if len(sorted_set) > 0:
            min_count = sorted_set[1]
        else:
            min_count = sorted_set[0]

        for count in self.totalCountSet:
            i += 1
            average += count
        average = average // i
        median = 0
        if len(sorted_set) % 2 == 0:
            median += sorted_set[len(sorted_set) // 2]
        else:
            sum_near_median = sorted_set[len(sorted_set) // 2 - 1] + sorted_set[len(sorted_set) // 2]
            median += (sum_near_median // 2)

        print("Min:", min_count, "Max: ", max_count, "Average: ", average,"Median: ",median,"Range: ", min_count, "-", max_count,"Median: ")

        detectedVideo = DetectionVideos(videoTitle=newName,user_id=self.getuserId(),video=os.path.join('videos/detections/',newName),thumbnail="videos/detections/thumbnails/"+newName.split('.')[0]+".jpg",date=datetime.datetime.now(),min_count=min_count,max_count=max_count,average_count=average,median_count=median)
        detectedVideo.save()
        self.setDetectedVideoUrl(detectedVideo.video)


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
                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                # cv2.putText(frame, text, (x, y - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, rects

    def tracking(self,frame, rects, ct, person_id_list):
        objects = ct.update(rects)
        live_person_count = len(objects)
        for (objectID, centroid) in objects.items():
            if objectID not in person_id_list:
                person_id_list.append(objectID)
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 220, 225), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (102, 220, 225), -1)
            #cv2.rectangle(frame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (255, 0, 0), 2)

        return frame, person_id_list, live_person_count, len(person_id_list)



    def preprocessing(self, video):

        start_time = datetime.datetime.now()
        total_frames = 0

        while video.isOpened():
            hasFrame, image = video.read()
            evtPreprocessing = Event()
            if not hasFrame:
                break
            image = cv2.resize(image, (1080, 720))

            if total_frames == 1:
                cv2.imwrite("media/videos/detections/thumbnails/"+self.newName.split('.')[0]+".jpg", image)

            total_frames = total_frames + 1
            end_time = datetime.datetime.now()
            time_diff = end_time - start_time
            if time_diff.seconds == 0:
                fps = 0
            else:
                fps = total_frames / time_diff.seconds
            fps_text = "FPS: {:.2f}".format(fps)
            cv2.putText(image, fps_text, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            # cv2.imshow('demo', image)
            self.frameQueue.put((image, evtPreprocessing))
            print('item added: ', self.frameQueue.qsize())
            # time.sleep(0.01)
            evtPreprocessing.wait()
        video.release()

    def inference(self, video, net):
        while video.isOpened():
            # determine only the *output* layer names that we need from YOLO
            evtInference = Event()
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            dataFrame, evtPreprocessing = self.frameQueue.get()
            blob = cv2.dnn.blobFromImage(dataFrame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            self.detectionQueue.put((dataFrame, layerOutputs, evtInference))
            print("detection item added: ",self.detectionQueue.qsize())
            print("item removed: ", self.frameQueue.qsize())
            # time.sleep(0.0001)
            evtPreprocessing.set()
            self.frameQueue.task_done()
            evtInference.wait()
            # sprint(layerOutputs)
        video.release()

    def display(self, video,frameOut):

        (H, W) = (720, 1080)
        while video.isOpened():
            boxes = []
            confidences = []
            classIDs = []

            dataFrame, layerOutputs, evtInference = self.detectionQueue.get()
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
                    if confidence > confthres:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                                    nmsthres)
            person_counter = 0
            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color = [int(c) for c in self.Colors[classIDs[i]]]
                    cv2.rectangle(dataFrame, (x, y), (x + w, y + h), color, 2)
                    text = "{}".format(self.Lables[classIDs[i]])
                    cv2.putText(dataFrame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if (self.Lables[classIDs[i]] == "head"):
                        person_counter += 1

            count_txt = "Person Count: {}".format(person_counter)
            cv2.putText(dataFrame, count_txt, (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            # cv2.imshow("demo", dataFrame)
            print("detection item removed: ",self.detectionQueue.qsize())
            evtInference.set()
            self.detectionQueue.task_done()
            frameOut.write(dataFrame)
        video.release()
        frameOut.release()

        detectedVideo = DetectionVideos(videoTitle=self.newName,user_id=self.getuserId(),video=os.path.join('videos/detections/',self.newName),thumbnail="videos/detections/thumbnails/"+self.newName.split('.')[0]+".jpg",date=datetime.datetime.now())
        detectedVideo.save()
        self.setDetectedVideoUrl(detectedVideo.video)

    def ControlledThreading(self):
        currentTime = datetime.datetime.now()
        newName = str(currentTime.day)+"_"+str(currentTime.month)+"_"+str(currentTime.year)+"_"+str(currentTime.second)+"_"+self.getVideoTitle()
        self.newName = newName
        if self.getVideoTitle() != '':
            videoUrl = self.getVideoURL().replace('\\','/')
            frameOut = cv2.VideoWriter("media/videos/detections/" + self.newName, -1, 30.0, (1080, 720))

            video = cv2.VideoCapture(videoUrl)

            t1 = Thread(target=self.preprocessing, args=(video,))
            t2 = Thread(target=self.inference, args=(video, self.nets,))
            t3 = Thread(target=self.display, args=(video,frameOut))

            t1.start()
            t2.start()
            t3.start()
            print("t1 started")
            t1.is_alive()
            t2.is_alive()
            t3.is_alive()
            t1.join()
            t2.join()
            t3.join()
            t1.is_alive()
            t2.is_alive()
            t3.is_alive()
        else:
            print("Something went wrong.")
