
import os
import cv2
import time
import torch
import argparse
import numpy as np
from Detection.Utils import ResizePadding
from Track.Tracker import Detection, Tracker
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
source = 'videoplayback.mp4'
def preproc(image):
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
def kpt2bbox(kpt, ex=20):
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))
if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                        help='Source of camera (0) or video file path.')
    args = par.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detect_model = TinyYOLOv3_onecls(384,device=device)
    max_age = 30
    resize_fn = ResizePadding(384,384)
    tracker = Tracker(max_age=max_age, n_init=3)
    inp_pose = (224,116)
    pose_model = SPPE_FastPose("resnet50", inp_pose[0], inp_pose[1], device=device)
    action_model = TSSTG()
    fps_time = 0
    cap = cv2.VideoCapture(source)
    f = 0
    while(True):
        f += 1
        ret, frame = cap.read()
         if ret:
            frame = preproc(frame)
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
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)


            tracker.update(detections)
            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)
                action = 'pending..'
                clr = (0, 255, 0)
                # Use 30 frames time-steps to prediction.
                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_model.predict(pts, frame.shape[:2])
                    action_name = action_model.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

                # VISUALIZE.
                if track.time_since_update == 0:
                    frame = draw_single(frame, track.keypoints_list[-1])
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                    frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0), 2)
                    frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.4, clr, 1)

            frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
            frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frame = frame[:, :, ::-1]
            fps_time = time.time()
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    breakpoint
    cam.stop()        
    cv2.destroyAllWindows()
