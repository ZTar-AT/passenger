import cv2
import pandas as pd
import time
import threading
import numpy as np
from hubconf import custom
from models.yolo import Model
from config import config



# =======================
model=None
frame = None
previous_results = None
lock = threading.Lock()
desired_width = config['video_size'][0]
desired_height = config['video_size'][1]
frame_interval_seconds = config['frame_interval_seconds']
confidence_threshold = config['confidence_threshold']
start_time = time.time()
points = config['scope'] 
is_scope = config['is_scope'] 
class_type=config['class']
color_class=config['color_class']
detection=config['detection']
color_level=config['color']
max_seat=config['max_seat']
rating=config['rating']

def calculate_rating(score):
    if score >= rating[2]:
        return color_level[3]
    elif score >= rating[1]:
        return color_level[2]
    elif score >= rating[0]:
        return color_level[1]
    else:
        return color_level[0]
    
def draw_test(image,body=0,face=0):
    overlay = image.copy()
    rectangle_position = (image.shape[1] - 100 - 20, 20)
    cv2.rectangle(overlay, rectangle_position, (rectangle_position[0] + 100, rectangle_position[1] + 60), (0, 0, 0), -1)
    tab=20
    person=0
    if detection[0]:
        text = f"{class_type[0]} : {body}"
        person=body
        text_position = (rectangle_position[0] + 10, rectangle_position[1] + tab)
        cv2.putText(overlay, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tab=50
    if detection[1]:
        text = f"{class_type[1]} : {face}"
        text_position = (rectangle_position[0] + 10, rectangle_position[1] + tab)
        cv2.putText(overlay, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        opacity = 0.5
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
        if detection[0]:
            tab=80
        else:
            person=face
            tab=50

    rectangle_position = (image.shape[1] - 100 - 20, tab)
    score = (person / max_seat) * 100 

    
    cv2.rectangle(image, rectangle_position, (rectangle_position[0] + 100, rectangle_position[1] + 30), calculate_rating(score), -1)
    cv2.putText(image, f"{max_seat-person}", (image.shape[1] - 80, tab+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


    if is_scope:
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], (0, 255, 0), 2)
            cv2.circle(image, points[i], 5, (0, 0, 255), -1)
        cv2.line(image, points[-1], points[0], (0, 255, 0), 2)
        cv2.circle(image, points[-1], 5, (0, 0, 255), -1)


    return image




def inference_thread_1():
    global frame, previous_results,model
    while True:
        start_time = time.time()

        if frame is not None:
            image=frame.copy()
            if is_scope:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))
                image = cv2.bitwise_and(frame, frame, mask=mask)

            results = model(image)
            if results is not None and results.xyxy is not None and len(results.xyxy) > 0:
                df_prediction = pd.DataFrame(results.xyxy[0][:, :6], columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
                df_prediction['name'] = df_prediction['class'].map(class_type)

                with lock:
                    previous_results = df_prediction
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_interval_seconds:
            time.sleep(frame_interval_seconds - elapsed_time)

def load_model():
    global model
    print("Loading Model...")
    model = custom(path_or_model=config['model'])
    print("Succeed...")


def detect():
    global frame
    if model is None: 
        print("Model Error...")
        return
    print("Detect...")

    cap=None
    if (config['mode']==0):
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(config['video'])
        
    if cap is not None and not cap.isOpened():
        print("Unable to turn on camera")
        exit()

    global start_time
    start_time = time.time()

    inference_thread = threading.Thread(target=inference_thread_1)
    inference_thread.start()



    class_counts=[0,0]

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame = cv2.resize(frame, (desired_width, desired_height))

            
            if previous_results is not None:
                        body,face=0,0
                        for index, row in previous_results.iterrows():
                            x_min, y_min, x_max, y_max = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']])
                            confidence = row['confidence']
                            if confidence >confidence_threshold:
                                
                                if detection[0] and row['class'] == 0:
                                    body += 1
                                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_class[0], 2)
                                    cv2.putText(frame, row['name'],
                                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_class[0], 2)
                                elif detection[1] and row['class'] == 1:
                                    face += 1
                                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_class[1], 2)
                                    cv2.putText(frame, row['name'],
                                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_class[1], 2)
                        class_counts=[body,face]

            cv2.imshow(config["name_project"], draw_test(frame,class_counts[0],class_counts[1]))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    inference_thread.join()

    cap.release()
    cv2.destroyAllWindows()
