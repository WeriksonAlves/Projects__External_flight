#!/usr/bin/env python
import cv2
import time
import os
from modules import BebopROS
from modules import myYolo
from modules import DroneCamera
import numpy as np
import rospy

def my_imshow(camera: DroneCamera):
    time.sleep(2)
    start_time = time.time() - 5
    while not rospy.is_shutdown() and time.time() - start_time < 120:  
        if camera.success_compressed_image:
            try:
                # Detect people in the frame
                results_people, results_identifies = track.detects_people_in_frame(camera.image_compressed)

                # Identify operator
                boxes, track_ids = track.identifies_operator(results_people)

                # Crop operator in frame
                cropped_image, _ = track.crop_operator_in_frame(boxes, track_ids, results_identifies, camera.image_compressed)
                
                # Centralize person in frame , 
                dist_center_h, dist_center_v =  track.centralize_person_in_frame(camera.image_compressed, boxes[0])
                
                # Signal control
                gain = [45, 45]
                sc_pitch = np.tanh(dist_center_h)*gain[0]
                sc_yaw = int(np.tanh(dist_center_h)*gain[1])
                
                print(f"Distance: ({dist_center_h:8.4f}, {dist_center_v:8.4f})   Signal Control: ({sc_pitch:8.4f}, {sc_yaw:8.4f})", end='')

                camera.move_camera(sc_pitch, 0)
                
                # Show the image in real time
                cv2.imshow('frame', camera.image_compressed)
                cv2.imshow('cropped_image', cropped_image)
                cv2.waitKey(1)
            except:
                # Show the image in real time
                cv2.imshow('frame', camera.image_compressed)
                cv2.waitKey(1)
        
        # pressione a tecla q para encerrar o programa
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


try:
    # Diretório para salvar as imagens
    DIR_NAME = os.path.dirname(__file__)
    file_path = os.path.join(DIR_NAME, 'images')
    # cap = cv2.VideoCapture(4)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    bebop = BebopROS()

    track = myYolo('yolov8n-pose.pt')
    camera = bebop.camera


    rospy.loginfo("Drone camera system initialized")
    my_imshow(camera)

except rospy.ROSInterruptException:
    rospy.logerr("ROS node interrupted.")
