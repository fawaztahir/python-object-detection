import cv2

classNames = []
threshold = 0.7
classNamesFile = 'coco.names'
weightsFile = 'frozen_inference_graph.pb'
confsFile = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# img = cv2.imread('sample.jpg')
camera = cv2.VideoCapture(0)
camera.set(3, 1280)
camera.set(4, 720)

with open(classNamesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

totalNames = len(classNames)

net = cv2.dnn_DetectionModel(weightsFile, confsFile)
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.7, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = camera.read()
    classIds, confs, bbox = net.detect(img, threshold)

    if (len(classIds) > 0):
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                print(classId, classId-1, confidence, box)
                if (classId-1 <= totalNames):
                    rounded_confidence = round(confidence * 100, 2)
                    cv2.rectangle(img, box, color=(0,255,0), thickness=2)
                    cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0,255,0))
                    cv2.putText(img, str(rounded_confidence), (box[0]+10, box[1]+65), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0,255,0))
        
    cv2.imshow('Output', img)
    cv2.waitKey(1)