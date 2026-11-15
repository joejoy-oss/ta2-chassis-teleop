class XYZ:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class Pose:
    def __init__(self):
        self.position = XYZ()
        self.orientation = XYZ()
        self.orientation.w = 1.0
