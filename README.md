# 3D Human Pose Estimation

### Machine Learning 2 4052U Final Project

Description: An end-to-end application that estimates 3D human body pose from single camera video or webcam input. The system uses YOLO v26m-Pose for 2D keypoint detection and PoseFormerV2 for lifting 2D joint sequences into 3D space, all presented in a Gradio web interface.

---

## Problem Formulation
The goal of this project is to create a 3D human pose estimation for a live video stream. Reconstructing 3D joint positions of a person from 2D images.

### Neural Network Components
In the original implementation, PoseFormerV2 builds off its previous iteration and uses several different models in its pipeline to create its 3D estimations from video inputs. This is done by using a YOLO object detection model along side a specially trained HRNet to create 2D keypoints estimations. This would then be inputted into the poseformer model build on a spatial transformer architecture trained on the Human3.6M dataset to lift the 2D points into 3D space. 

For our project, this was used as the base of our implementation and augmented to include live video inferences. In this, we started by simplifying the original 2D pose estimation pipeline, replacing it entirely with a single YOLOv26-pose model for 2D keypoint pose estimation. Next, avoiding the need to retrain the original  poseformer model, the 2D keypoints estimated by YOLO are mapped from its COCO format to a H36M format and normalized before inference. Finally, since PoseFormer looks at a window of frames centered around a target frame for creating its estimations, it cannot normally run with a live video stream. To address this issue, we applied a small buffer to the queue of frames so it would always have access to a full temporal window of frames, enabling a live video stream.

### Video Demo
<video width="100%" controls>
  <source src="./demo/demo.mp4" type="video/mp4">
</video>

### Tech Stack
* Frontend: Gradio
* Python 3.9
* Pytorch

### Project Structure
```
3D Human Pose Estimation
│   .gitignore
│   app.py
│   main.py
│   README.md
│   requirements.txt
├───common
│       model_poseformer.py
│       poseformer_vis.py
│       poselive.py
└───model
    └───checkpoint
```

### Setup & Installation
1. Clone github repository 
```
git clone https://github.com/GordLaw/Machine-Learning-4052U-Final.git
```
2. Create virtual environment in working directory
```
python -m venv .venv
.venv\Scripts\activate
```
3. Install python requirements
```
pip install -r requirements.txt
```
4. Install PyTorch with CUDA
```
# For development we used CUDA 12.8
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
4. Demoing application
    * Serving gradio frontend
    ```
    python ./app.py
    ```
    * Serving standalone model 
    ```
    python ./main.py
    ```

### Contributors
* BeatrizPO2
* GordLaw

### Resources
* https://qitaozhao.github.io/PoseFormerV2
* https://github.com/zczcwh/PoseFormer
* https://github.com/Vegetebird/MHFormer
* https://docs.ultralytics.com/tasks/pose/
* https://github.com/paTRICK-swk/P-STMO?tab=readme-ov-file#mpi-inf-3dhp-1
