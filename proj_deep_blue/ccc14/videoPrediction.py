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

# karan_custom_23_01_21.cfg
# karan_custom_best_23_01_21.weights
# obj.names

# 16_02_2021/proj_deep_blue_16_02_2021_names.names
# 16_02_2021/proj_deep_blue_16_02_2021_cfg_upto_3000.cfg
# 16_02_2021/proj_deep_blue_16_02_2021_cfg_best.weights

class VideoPrediction:

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
        self.labelsPath = "16_02_2021/proj_deep_blue_16_02_2021_names.names"
        self.cfgpath = "16_02_2021/proj_deep_blue_16_02_2021_cfg_upto_3000.cfg"
        self.wpath = "16_02_2021/proj_deep_blue_16_02_2021_cfg_best.weights"
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

        start_time = time.time()
        total_frames = 0
        person_count = 0

        while(True):
            check, image = video.read()
            if check != False:
                image = cv2.resize(image,(1080,720))
                total_frames = total_frames + 1

                if total_frames == 1:
                    cv2.imwrite("media/videos/detections/thumbnails/"+newName.split('.')[0]+".jpg", image)

                if(total_frames%1 == 0):
                    image,person_count = self.get_predection(image, self.nets, self.Lables, self.Colors)

                end_time = time.time()
                time_diff = end_time - start_time
                if time_diff == 0:
                    fps = 0
                else:
                    fps = total_frames/time_diff
                fps_text = "FPS: {:.2f}".format(fps)
                cv2.putText(image,fps_text,(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                #frame_count = frame_count + 1
                count_txt = "Person Count: {}".format(person_count)
                cv2.putText(image,count_txt, (10,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                #cv2.imshow("capture", image)
                out.write(image)
            else:
                break

        video.release()
        out.release()

        detectedVideo = DetectionVideos(user_id=self.getuserId(),video=os.path.join('videos/detections/',newName),thumbnail="videos/detections/thumbnails/"+newName.split('.')[0]+".jpg",date=datetime.datetime.now())
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

    def get_predection(self, image, net, LABELS, COLORS):
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
        # print(layerOutputs)
        end = time.time()

        # show timing information on YOLO
        # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

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
                if confidence > confthres:
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

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                text = "{}".format(LABELS[classIDs[i]])
                # print(boxes)
                # print(classIDs)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if (LABELS[classIDs[i]] == "head"):
                    person_counter += 1
        # count_txt = "Person Count: {}".format(person_counter)
        # cv2.putText(image, count_txt, (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        return image,person_counter

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
                    if (self.Lables[classIDs[i]] == "Person"):
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

        detectedVideo = DetectionVideos(user_id=self.getuserId(),video=os.path.join('videos/detections/',self.newName),thumbnail="videos/detections/thumbnails/"+self.newName.split('.')[0]+".jpg",date=datetime.datetime.now())
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
