#!/usr/bin/env python
'''
rostopic lit

/bebop/autoflight/navigate_home
/bebop/autoflight/pause
/bebop/autoflight/start
/bebop/autoflight/stop
/bebop/bebop_driver/parameter_descriptions
/bebop/bebop_driver/parameter_updates
/bebop/camera_control
/bebop/camera_info
/bebop/cmd_vel
/bebop/fix
/bebop/flattrim
/bebop/flip

/bebop/joint_states
/bebop/land
/bebop/odom
/bebop/record
/bebop/reset
/bebop/set_exposure
/bebop/snapshot
/bebop/states/ardrone3/CameraState/Orientation
/bebop/states/ardrone3/GPSState/NumberOfSatelliteChanged
/bebop/states/ardrone3/MediaStreamingState/VideoEnableChanged
/bebop/states/ardrone3/PilotingState/AltitudeChanged
/bebop/states/ardrone3/PilotingState/AttitudeChanged
/bebop/states/ardrone3/PilotingState/FlatTrimChanged
/bebop/states/ardrone3/PilotingState/FlyingStateChanged
/bebop/states/ardrone3/PilotingState/NavigateHomeStateChanged
/bebop/states/ardrone3/PilotingState/PositionChanged
/bebop/states/ardrone3/PilotingState/SpeedChanged
/bebop/states/common/CommonState/BatteryStateChanged
/bebop/states/common/CommonState/WifiSignalChanged
/bebop/states/common/FlightPlanState/AvailabilityStateChanged
/bebop/states/common/FlightPlanState/ComponentStateListChanged
/bebop/states/common/MavlinkState/MavlinkFilePlayingStateChanged
/bebop/states/common/MavlinkState/MavlinkPlayErrorStateChanged
/bebop/states/common/OverHeatState/OverHeatChanged
/bebop/takeoff

'''

import cv2
import os
import rospy
import time
from Projects__External_flight.modules.bebop_autonomous.ImageBebop import ImageListener, ParameterListener
from BebopROS import BebopROS

# Diretório para salvar as imagens
DIR_NAME = os.path.dirname(__file__)
file_path = os.path.join(DIR_NAME, 'images')

if not os.path.exists(file_path):
    os.makedirs(file_path)

ba = BebopROS()

def my_imshow(bit: ImageListener):
    time.sleep(2)
    start_time = time.time()
    while time.time() - start_time < 15:
        if bit.success_image:
            cv2.imshow('/bebop/image_raw ', bit.image)
            cv2.waitKey(1)
        if bit.success_compressed_image:
            cv2.imshow('/bebop/image_raw/compressed ', bit.image_compressed)
            cv2.waitKey(1)
        if not (bit.success_image or bit.success_compressed_image):
            print('No image received')
    cv2.destroyAllWindows()

def main():
    rospy.init_node('bebop_image_processor', anonymous=True)

    # Subscrever aos tópicos de imagens
    ba.image_listener.subscribe_to_image_raw()
    ba.image_listener.subscribe_to_image_raw()
    ba.image_listener.subscribe_to_image_raw_compressed()
    
    # Subscrever aos tópicos de descrição e atualizações de parâmetros
    ba.parameter_listener.subscribe_to_parameter_descriptions()
    ba.parameter_listener.subscribe_to_parameter_updates()

    my_imshow(ba.image_listener)

    # rospy.spin()  # Mantém o nó ativo enquanto as mensagens são processadas

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

    
