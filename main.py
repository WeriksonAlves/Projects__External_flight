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
    while not rospy.is_shutdown() and time.time() - start_time < 60:  
        if camera.success_compressed_image:
            # Detect people in the frame
            results_people, results_identifies = track.detects_people_in_frame(camera.image_compressed)

            # Identify operator
            boxes, track_ids = track.identifies_operator(results_people)

            # Crop operator in frame
            cropped_image, _ = track.crop_operator_in_frame(boxes, track_ids, results_identifies, camera.image_compressed)
            
            # Centralize person in frame
            hd,dh, vd, dv =  track.centralize_person_in_frame(camera.image_compressed, boxes[0])
            print(f"Direction: ({hd}, {vd}) and Distance: ({dh}, {dv})")

            sc_pitch = 5*np.tanh(dh*(10/250))
            sc_yaw = 5*np.tanh(dv*(10/250))
            camera.move_camera(-sc_yaw,sc_pitch)
            
            
            # Show the image in real time
            cv2.imshow('frame', camera.image_compressed)
            cv2.imshow('cropped_image', cropped_image)
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
