from Parameters import *
from Face_Detection import *
from Classify import *

param = Parameters()
face = FacialDetector(param)
face.train()

detections, scores, file_names = face.run()

classify(param, detections, scores, file_names)