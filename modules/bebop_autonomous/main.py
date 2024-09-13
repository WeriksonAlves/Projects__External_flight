#!/usr/bin/env python
import cv2
import time
import os
from BebopROS import BebopROS
from DroneCamera import DroneCamera
import rospy

def my_imshow(camera: DroneCamera):
    time.sleep(2)
    start_time = time.time() - 5
    while not rospy.is_shutdown() and time.time() - start_time < 100:        
        if time.time() - start_time < 10:
            camera.set_exposure(0)
        elif time.time() - start_time < 20:
            camera.set_exposure(0.9)
        elif time.time() - start_time < 30:
            camera.set_exposure(-0.9)
        else:
            camera.set_exposure(0)

        if camera.success_compressed_image:
            cv2.imshow('/bebop/image_raw/compressed ', camera.image_compressed)
            cv2.waitKey(1)
        else:
            print('No image received')
        
        # pressione a tecla q para encerrar o programa
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    try:
        # Diretório para salvar as imagens
        DIR_NAME = os.path.dirname(__file__)
        file_path = os.path.join(DIR_NAME, 'images')

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        bebop = BebopROS()
        camera = bebop.camera

        rospy.loginfo("Drone camera system initialized")
        my_imshow(camera)

    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")
