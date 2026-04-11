import torch
import torch.nn as nn
from .model_poseformer import PoseTransformerV2 as Model
from .poseformer_vis import PoseFormerLiveVisualizer
from ultralytics import YOLO
from collections import deque
import numpy as np
import cv2
import glob
import os

RECEPTIVE_FIELD = 27

class PoseLive:
    def __init__(self):
        self.yolo_pose = None
        self.vid_cap = None
        self.vis = None
        self.poseformer = None
        self.pose_queue = None

    def load(self, args):
        # PoseFormer normally works only with videos and for inference it uses
        # the previous and next frame to generate the estimation so for live 
        # inference we will delay the input video to address this
        self.pose_queue = deque(maxlen=RECEPTIVE_FIELD)
        dummy_pose = np.zeros((17, 2)) 
        for _ in range(RECEPTIVE_FIELD):
            self.pose_queue.append(dummy_pose)

        self.vis = PoseFormerLiveVisualizer()
        self.vid_cap = cv2.VideoCapture(0)

        # Loads required models
        self.yolo_pose = YOLO("model/checkpoint/yolo26m-pose.pt")

        self.poseformer = nn.DataParallel(Model(args=args)).cuda()

        model_dict = self.poseformer.state_dict()
        model_path = sorted(glob.glob(os.path.join(args.previous_dir, '1_3_27_48.7.bin')))[0]

        pre_dict = torch.load(model_path, weights_only=False)
        self.poseformer.load_state_dict(pre_dict['model_pos'], strict=True)

        self.poseformer.eval()

    def run_inference(self):
        while self.vid_cap.isOpened():
            success, frame = self.vid_cap.read()

            if not success:
                break

            results = self.yolo_pose.predict(source=frame, verbose=False)
            raw_pose = self.extract_best_person_pose(results)
            raw_pose = self.yolo_to_h36m(raw_pose)

            if raw_pose is not None:
                norm_pose = self.normalize_2d_pose(raw_pose, frame.shape[1], frame.shape[0])
                self.pose_queue.append(norm_pose)
            else:
                self.pose_queue.append(self.pose_queue[-1])

            pose_sequence = np.array(self.pose_queue) 
            input_tensor = torch.tensor(pose_sequence, dtype=torch.float32).unsqueeze(0).cuda()

            for r in results:
                annotated_frame = r.plot()

            with torch.no_grad():
                predicted_3d_pose = self.poseformer(input_tensor)
                pose_numpy = predicted_3d_pose.squeeze().cpu().numpy()

            self.vis.update(pose_numpy)
            cv2.imshow('Camera', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.vid_cap.release()
        cv2.destroyAllWindows()
    
    def extract_best_person_pose(self, yolo_points):
        if len(yolo_points) == 0 or yolo_points[0].keypoints is None:
            return None
        
        keypoints = yolo_points[0].keypoints.xy[0].cpu().numpy() 
        return keypoints
    
    def normalize_2d_pose(self, X, image_width, image_height):
        return X / image_width * 2 - np.array([1, image_height / image_width])
    
    def yolo_to_h36m(self, yolo_points):
        # Empty array for all 17 points of H36M joints
        h36m_pose = np.zeros((17, 2))
        
        pelvis = (yolo_points[11] + yolo_points[12]) / 2.0 
        thorax = (yolo_points[5] + yolo_points[6]) / 2.0
        spine = (pelvis + thorax) / 2.0
        head = (yolo_points[3] + yolo_points[4]) / 2.0
        neck = yolo_points[0]
        
        # Map everything to the exact H36M indices
        h36m_pose[0] = pelvis
        h36m_pose[1] = yolo_points[12] # Right Hip
        h36m_pose[2] = yolo_points[14] # Right Knee
        h36m_pose[3] = yolo_points[16] # Right Ankle
        
        h36m_pose[4] = yolo_points[11] # Left Hip
        h36m_pose[5] = yolo_points[13] # Left Knee
        h36m_pose[6] = yolo_points[15] # Left Ankle
        
        h36m_pose[7] = spine
        h36m_pose[8] = thorax
        h36m_pose[9] = neck
        h36m_pose[10] = head
        
        h36m_pose[11] = yolo_points[5] # Left Shoulder
        h36m_pose[12] = yolo_points[7] # Left Elbow
        h36m_pose[13] = yolo_points[9] # Left Wrist
        
        h36m_pose[14] = yolo_points[6] # Right Shoulder
        h36m_pose[15] = yolo_points[8] # Right Elbow
        h36m_pose[16] = yolo_points[10] # Right Wrist
        
        return h36m_pose