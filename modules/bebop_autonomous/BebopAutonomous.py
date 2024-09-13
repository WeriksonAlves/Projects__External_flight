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
/bebop/image_raw
/bebop/image_raw/compressed
/bebop/image_raw/compressed/parameter_descriptions
/bebop/image_raw/compressed/parameter_updates
/bebop/image_raw/compressedDepth
/bebop/image_raw/compressedDepth/parameter_descriptions
/bebop/image_raw/compressedDepth/parameter_updates
/bebop/image_raw/theora
/bebop/image_raw/theora/parameter_descriptions
/bebop/image_raw/theora/parameter_updates
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
from datetime import datetime
from sensor_msgs.msg import CompressedImage, Image
from ImageRawTools import ImageListener, ParameterListener
from base import ImageTools

# Criação de uma instância da classe ImageTools
it = ImageTools()

# Diretório para salvar as imagens
output_dir = "/home/ubuntu/Documentos/Werikson/GitHub/env_master/Projects__External_flight/modules/bebop_autonomous/images/"  # Altere para o diretório desejado

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def my_imshow(bit):
    time.sleep(5)
    
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

    DIR_NAME = os.path.dirname(__file__)

    # Instanciar o listener de imagens
    image_listener = ImageListener(DIR_NAME)

    # Subscrever aos tópicos de imagens
    image_listener.subscribe_to_image_raw()
    image_listener.subscribe_to_image_raw_compressed()
    my_imshow(image_listener)

    # Instanciar o listener de parâmetros
    parameter_listener = ParameterListener(image_listener)
    
    # Subscrever aos tópicos de descrição e atualizações de parâmetros
    parameter_listener.subscribe_to_parameter_descriptions()
    parameter_listener.subscribe_to_parameter_updates()

    rospy.spin()  # Mantém o nó ativo enquanto as mensagens são processadas

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

    
