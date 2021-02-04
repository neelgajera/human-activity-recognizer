# IMPORTS
import time
import torch
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import cv2
from Detection.Utils import ResizePadding
from Track.Tracker import Detection, Tracker
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
import json
def preproc(image):
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
def kpt2bbox(kpt, ex=20):
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
detect_model = TinyYOLOv3_onecls(384,device=device)
max_age = 30
resize_fn = ResizePadding(384,384)
tracker = Tracker(max_age=max_age, n_init=3)
inp_pose = (224,116)
pose_model = SPPE_FastPose("resnet50", inp_pose[0], inp_pose[1], device=device)
action_model = TSSTG()

l_pair = [(0, 13), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (13, 7), (13, 8),  # Body
             (7, 9), (8, 10), (9, 11), (10, 12)]






# added to put object in JSON
class Object(object):
    def __init__(self):
        self.name="webrtcHacks TensorFlow Object Detection REST API"

    def toJSON(self):
        return json.dumps(self.__dict__)

def get_objects(image):
  output = []
  image = np.array(image)  
  frame = preproc(image)
  detected = detect_model.detect(frame, need_resize=False, expand_bb=10)
  print(detected)
  tracker.predict()
  for track in tracker.tracks:
    det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
    detected = torch.cat([detected, det], dim=0) if detected is not None else det
  detections = []
  if detected is not None:
    poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
    detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),np.concatenate((ps['keypoints'].numpy(), ps['kp_score'].numpy()), axis=1),ps['kp_score'].mean().numpy()) for ps in poses]

  tracker.update(detections)
  for i, track in enumerate(tracker.tracks):
    if not track.is_confirmed():
      continue

    track_id = track.track_id
    bbox = track.to_tlbr().astype(int)
    center = track.get_center().astype(int)
    action = 'pending..'
  
                  # Use 30 frames time-steps to prediction.
    if len(track.keypoints_list) == 30:
      pts = np.array(track.keypoints_list, dtype=np.float32)
      out = action_model.predict(pts, frame.shape[:2])
      action_name = action_model.class_names[out[0].argmax()]
      action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)

                  # VISUALIZE.
    if track.time_since_update == 0:
      pts = track.keypoints_list[-1]
      part_line = {}
      pts = np.concatenate((pts, np.expand_dims((pts[1, :] + pts[2, :]) / 2, 0)), axis=0)
      for n in range(pts.shape[0]):
        if pts[n, 2] <= 0.05:
          continue

        cor_x, cor_y = int(pts[n, 0]), int(pts[n, 1])
        part_line[n] = (cor_x, cor_y)
        itemc = Object()
        itemc.name = 'Object'
        itemc.circle = 'circle'
        itemc.y =float(cor_y)
        itemc.x = float(cor_x)
        output.append(itemc)

      for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
          start_xy = part_line[start_p]
          end_xy = part_line[end_p]
          iteml = Object()
          iteml.name = 'object'
          iteml.line = 'line'
          iteml.startx = float(start_xy[0])
          iteml.starty = float(start_xy[1])
          iteml.endx = float(end_xy[0])
          iteml.endy = float(end_xy[1])
          output.append(iteml)


      item = Object()
      item.name = 'Object'
      item.class_name = action
      item.track_id = str(track_id)
      item.y = float(bbox[1])
      item.x = float(bbox[0])
      item.height = float(bbox[3])
      item.width = float(bbox[2])
      output.append(item)

  outputJson = json.dumps([ob.__dict__ for ob in output])
  return outputJson
