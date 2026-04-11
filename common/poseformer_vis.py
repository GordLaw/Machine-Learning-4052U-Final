import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

H36M_BONES = [
    (0, 1), (1, 2), (2, 3),       # Right Leg (Pelvis -> Hip -> Knee -> Ankle)
    (0, 4), (4, 5), (5, 6),       # Left Leg (Pelvis -> Hip -> Knee -> Ankle)
    (0, 7), (7, 8), (8, 9),       # Spine to Thorax to Neck
    (9, 10),                      # Neck to Head
    (8, 11), (11, 12), (12, 13),  # Left Arm (Thorax -> Shoulder -> Elbow -> Wrist)
    (8, 14), (14, 15), (15, 16)   # Right Arm (Thorax -> Shoulder -> Elbow -> Wrist)
]

class PoseFormerLiveVisualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # PoseFormerV2's official colours
        lcolor = '#3498db' # left
        rcolor = '#e74c3c' # right
        pcolor = '#9b59b6' # spine
        
        # Colours set to aligned H36M_Bones connection
        self.colors = [
            rcolor, rcolor, rcolor, 
            lcolor, lcolor, lcolor, 
            pcolor, pcolor, pcolor, 
            pcolor, 
            lcolor, lcolor, lcolor, 
            rcolor, rcolor, rcolor
        ]
        
        self.lines = []
        for color in self.colors:
            line, = self.ax.plot([0, 0], [0, 0], [0, 0], c=color, lw=2.5)
            self.lines.append(line)
            
        self.radius = 1.0 
        self.ax.set_xlim3d([-self.radius, self.radius])
        self.ax.set_ylim3d([-self.radius, self.radius])
        self.ax.set_zlim3d([-self.radius, self.radius])
        
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        
        # Match PoseFormer's default viewing angle
        self.ax.view_init(elev=15, azim=70)
        plt.show(block=False)

    def update(self, pose_3d):
        for i, (j1, j2) in enumerate(H36M_BONES):
            # Extract coordinates
            x = [pose_3d[j1, 0], pose_3d[j2, 0]]
            y = [pose_3d[j1, 1], pose_3d[j2, 1]]
            z = [pose_3d[j1, 2], pose_3d[j2, 2]]    
                    
            self.lines[i].set_data(x, z) 
            self.lines[i].set_3d_properties([-val for val in y]) 

        plt.pause(0.001)