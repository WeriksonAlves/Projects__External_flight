#!/usr/bin/env python

import os

from .DroneCamera import DroneCamera


class BebopROS:
    def __init__(self):
        # Diretório para salvar as imagens
        file_path = os.path.join(os.path.dirname(__file__), 'images')

        if not os.path.exists(file_path): os.makedirs(file_path)

        self.camera = DroneCamera(file_path)