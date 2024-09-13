#!/usr/bin/env python
import cv2
import time
import os
from DroneCamera import DroneCamera
import rospy

def my_imshow(bit):
    time.sleep(2)
    start_time = time.time()
    while time.time() - start_time < 5:
        if bit.success_image:
            cv2.imshow('/bebop/image_raw ', bit.image)
            cv2.waitKey(1)
        if bit.success_compressed_image:
            cv2.imshow('/bebop/image_raw/compressed ', bit.image_compressed)
            cv2.waitKey(1)
        if not (bit.success_image or bit.success_compressed_image):
            print('No image received')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        # Diretório para salvar as imagens
        DIR_NAME = os.path.dirname(__file__)
        file_path = os.path.join(DIR_NAME, 'images')

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        camera = DroneCamera(file_path)
        my_imshow(camera)

        rospy.loginfo("Drone camera system initialized")
        camera.start_camera_stream()
    except rospy.ROSInterruptException:
        pass
