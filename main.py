#!/usr/bin/env python
import cv2
import time
import os
from modules import BebopROS
# from modules import myYolo
from modules import DroneCamera
import rospy

def my_imshow(camera: DroneCamera):
    time.sleep(2)
    start_time = time.time() - 5
    while not rospy.is_shutdown() and time.time() - start_time < 100:  
        # ang = 30
        # if time.time() - start_time < 5:
        #     camera.move_camera(0,0)
        # elif time.time() - start_time < 10:
        #     camera.move_camera(ang, 0.)
        # elif time.time() - start_time < 15:
        #     camera.move_camera(0, ang)
        # elif time.time() - start_time < 20:
        #     camera.move_camera(-ang, 0)
        # elif time.time() - start_time < 25:
        #     camera.move_camera(0, -ang)
        # else:
        #     camera.move_camera(0., 0.)

        if camera.success_compressed_image:
            # yolo_model.track(camera.image_compressed, persist=True)
            # results_people = track.find_people(camera.image_compressed)
            # results_identifies = track.identify_operator(results_people)
            # cv2.imshow('/bebop/image_raw/compressed ', results_identifies)
            cv2.imshow('/bebop/image_raw/compressed ', camera.image_compressed)
            # cv2.imshow('RealSense ', cap.read()[1])
            cv2.waitKey(1)
        else:
            print('No image received')
        
        # pressione a tecla q para encerrar o programa
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


try:
    # Diretório para salvar as imagens
    DIR_NAME = os.path.dirname(__file__)
    file_path = os.path.join(DIR_NAME, 'images')
    # cap = cv2.VideoCapture(4)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    bebop = BebopROS()

    # track = myYolo('yolov8n.pt')
    
    
    camera = bebop.camera


    rospy.loginfo("Drone camera system initialized")
    my_imshow(camera)

except rospy.ROSInterruptException:
    rospy.logerr("ROS node interrupted.")
