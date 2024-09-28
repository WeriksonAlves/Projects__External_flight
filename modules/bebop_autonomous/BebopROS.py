import os

from .DroneCamera import DroneCamera


class BebopROS:
    def __init__(self):
        # Diretório para salvar as imagens
        self.file_path = os.path.join(os.path.dirname(__file__), 'images')
        self.drone_type = 'bebop2'

        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)

    def VideoCapture(self):
        try:
            self.camera = DroneCamera(self.file_path)
            self.camera.initialize_publishers(['camera_control',
                                               'snapshot',
                                               'set_exposure']
                                              )
            self.camera.initialize_subscribers(['compressed'
                                                ])
            self.camera.success_flags["isOpened"] = True
        except Exception as e:
            print(f"Error: {e}")
            self.camera.success_flags["isOpened"] = False
        return self.camera.success_flags["isOpened"]
