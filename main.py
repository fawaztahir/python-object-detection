import cv2
import glob

classNames = []
threshold = 0.5
classNamesFile = 'coco.names'
weightsFile = 'frozen_inference_graph.pb'
confsFile = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# img = cv2.imread('sample.jpg')
# camera = cv2.VideoCapture(0)
# camera.set(3, 1280)
# camera.set(4, 720)

PERSON_ID = 0

with open(classNamesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

totalNames = len(classNames)

net = cv2.dnn_DetectionModel(weightsFile, confsFile)
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.7, 127.5, 127.5))
net.setInputSwapRB(True)

def isObjectPerson(class_id: int) -> bool:
    return class_id - 1 == PERSON_ID

def findPersonInVideo(path: str) -> tuple:
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        return;

    person_found = False
    # print(f"DETECTING PERSON IN: {path}")
    while video.isOpened():
        if person_found:
            break

        check, frame = video.read()
        if check == False:
            break
        
        class_ids, confs, bbox = net.detect(frame, threshold)
        if (len(class_ids) == 0):
            continue

        for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
            # print(f"CLASS_ID: {class_id}")
            if (isObjectPerson(class_id)):
                # print(f"PERSON FOUND IN VIDEO AT GIVEN PATH: {path}")
                person_found = True
                break
    
    return person_found, class_id, classNames[class_id-1]


root_dir = 'videos/'

for filename in glob.iglob(f"{root_dir}/**/*.*", recursive=True):
    print(f"CHECKING :: {filename}")
    found, class_id, class_name = findPersonInVideo(filename)
    if found == True:
        print(f"{class_name.upper()} FOUND IN VIDEO: {filename}")
    else:
        print(f"NO OBJECT FOUND IN VIDEO: {filename}")

# videoPath = 'videos/video3.mp4'
# video = cv2.VideoCapture(videoPath)
# check, frame = video.read()

# if check == False:
#     print('Video not found. Please enter a valid full path to video.')
#     exit(1)

# break_while = False
# while video.isOpened():
#     check, frame = video.read()
#     if break_while:
#         break

#     if check:
#         classIds, confs, bbox = net.detect(frame, threshold)
#         if (len(classIds) > 0):
#             for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
#                 print(f"ClassId: {classId}")
#                 if (classId-1 == PERSON_ID):
#                     print(f'PERSON FOUND IN VIDEO AT {videoPath}')
#                     break_while = True
#                     break
#     else:
#         break


# while True:
#     success, img = camera.read()
#     classIds, confs, bbox = net.detect(img, threshold)

#     if (len(classIds) > 0):
#         for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
#                 print(classId, classId-1, confidence, box)
#                 if (classId-1 <= totalNames):
#                     rounded_confidence = round(confidence * 100, 2)
#                     cv2.rectangle(img, box, color=(0,255,0), thickness=2)
#                     cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0,255,0))
#                     cv2.putText(img, str(rounded_confidence), (box[0]+10, box[1]+65), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0,255,0))
        
#     cv2.imshow('Output', img)
#     cv2.waitKey(1)