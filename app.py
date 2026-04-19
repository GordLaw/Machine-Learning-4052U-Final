import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import glob
import argparse
import tempfile
import gradio as gr
import math
import subprocess
import imageio_ffmpeg
import time
from collections import deque
from ultralytics import YOLO
from common.model_poseformer import PoseTransformerV2 as Model

# Make ffmpeg available to Gradio
import shutil
ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dst = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
if not os.path.exists(ffmpeg_dst):
    shutil.copy2(ffmpeg_src, ffmpeg_dst)
os.environ["PATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + os.environ["PATH"]

#  Constants 
RECEPTIVE_FIELD = 27

H36M_BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]

# BGR colors for OpenCV
RIGHT_COLOR = (60, 76, 231)    # Red-ish
LEFT_COLOR = (219, 152, 52)    # Blue-ish  
SPINE_COLOR = (180, 89, 155)   # Purple-ish

BONE_COLORS = [
    RIGHT_COLOR, RIGHT_COLOR, RIGHT_COLOR,
    LEFT_COLOR, LEFT_COLOR, LEFT_COLOR,
    SPINE_COLOR, SPINE_COLOR, SPINE_COLOR,
    SPINE_COLOR,
    LEFT_COLOR, LEFT_COLOR, LEFT_COLOR,
    RIGHT_COLOR, RIGHT_COLOR, RIGHT_COLOR
]

#  Helper Functions 
def yolo_to_h36m(yolo_points):
    h36m_pose = np.zeros((17, 2))
    pelvis = (yolo_points[11] + yolo_points[12]) / 2.0
    thorax = (yolo_points[5] + yolo_points[6]) / 2.0
    spine = (pelvis + thorax) / 2.0
    head = (yolo_points[3] + yolo_points[4]) / 2.0
    neck = yolo_points[0]

    h36m_pose[0] = pelvis
    h36m_pose[1] = yolo_points[12]
    h36m_pose[2] = yolo_points[14]
    h36m_pose[3] = yolo_points[16]
    h36m_pose[4] = yolo_points[11]
    h36m_pose[5] = yolo_points[13]
    h36m_pose[6] = yolo_points[15]
    h36m_pose[7] = spine
    h36m_pose[8] = thorax
    h36m_pose[9] = neck
    h36m_pose[10] = head
    h36m_pose[11] = yolo_points[5]
    h36m_pose[12] = yolo_points[7]
    h36m_pose[13] = yolo_points[9]
    h36m_pose[14] = yolo_points[6]
    h36m_pose[15] = yolo_points[8]
    h36m_pose[16] = yolo_points[10]
    return h36m_pose


def normalize_2d_pose(X, image_width, image_height):
    return X / image_width * 2 - np.array([1, image_height / image_width])


def extract_best_person_pose(yolo_results):
    if len(yolo_results) == 0 or yolo_results[0].keypoints is None:
        return None
    if len(yolo_results[0].keypoints.xy) == 0:
        return None
        # Pick the highest-confidence person instead of just the first one
    boxes = yolo_results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    
    best_idx = boxes.conf.argmax().item()
    
    # Filter out low-confidence detections
    if boxes.conf[best_idx] < 0.5:
        return None
    keypoints = yolo_results[0].keypoints.xy[best_idx].cpu().numpy()
    return keypoints

def project_3d_to_2d(pose_3d, img_size=500, elev=15, azim=70):
    """Fast 3D to 2D projection using rotation matrices instead of matplotlib."""
    elev_rad = math.radians(elev)
    azim_rad = math.radians(azim)

    cos_a, sin_a = math.cos(azim_rad), math.sin(azim_rad)
    cos_e, sin_e = math.cos(elev_rad), math.sin(elev_rad)

    x = pose_3d[:, 0]
    y = -pose_3d[:, 1]
    z = pose_3d[:, 2]

    x_rot = x * cos_a + z * sin_a
    z_rot = -x * sin_a + z * cos_a

    y_rot = y * cos_e - z_rot * sin_e
    z_final = y * sin_e + z_rot * cos_e

    margin = 80
    scale = (img_size - 2 * margin) / 2.0
    cx, cy = img_size // 2, img_size // 2

    pts_2d = np.zeros((len(x), 2), dtype=np.int32)
    pts_2d[:, 0] = (x_rot * scale + cx).astype(np.int32)
    pts_2d[:, 1] = (-y_rot * scale + cy).astype(np.int32)

    depth = z_final
    return pts_2d, depth


def render_3d_skeleton_cv2(pose_3d, img_size=500):
    """Render 3D skeleton using OpenCV — much faster than matplotlib."""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:] = (50, 62, 44)  # Dark background

    cv2.putText(img, "3D Pose Estimation", (img_size // 2 - 130, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (241, 240, 236), 2)

    for i in range(0, img_size, 50):
        cv2.line(img, (i, 0), (i, img_size), (70, 80, 60), 1)
        cv2.line(img, (0, i), (img_size, i), (70, 80, 60), 1)

    pts_2d, depth = project_3d_to_2d(pose_3d, img_size)

    bone_depths = []
    for i, (j1, j2) in enumerate(H36M_BONES):
        avg_depth = (depth[j1] + depth[j2]) / 2
        bone_depths.append((avg_depth, i))
    bone_depths.sort(key=lambda x: x[0])

    for _, i in bone_depths:
        j1, j2 = H36M_BONES[i]
        pt1 = tuple(pts_2d[j1])
        pt2 = tuple(pts_2d[j2])
        cv2.line(img, pt1, pt2, BONE_COLORS[i], 3, cv2.LINE_AA)

    for j in range(17):
        pt = tuple(pts_2d[j])
        cv2.circle(img, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, pt, 3, (200, 200, 200), -1, cv2.LINE_AA)

    return img


# Load Models 
def load_models():
    args, _ = argparse.ArgumentParser().parse_known_args([])
    args.embed_dim_ratio = 32
    args.depth = 4
    args.frames = 27
    args.number_of_kept_frames = 1
    args.number_of_kept_coeffs = 3
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'model/checkpoint'
    args.n_joints = 17
    args.out_joints = 17

    yolo = YOLO("model/checkpoint/yolo26m-pose.pt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        poseformer = nn.DataParallel(Model(args=args)).cuda()
    else:
        poseformer = Model(args=args)

    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '1_3_27_48.7.bin')))[0]
    pre_dict = torch.load(model_path, map_location=device, weights_only=False)

    if torch.cuda.is_available():
        poseformer.load_state_dict(pre_dict['model_pos'], strict=True)
    else:
        state_dict = {k.replace('module.', ''): v for k, v in pre_dict['model_pos'].items()}
        poseformer.load_state_dict(state_dict, strict=True)

    poseformer.eval()
    return yolo, poseformer, device


print("Loading models... This may take a moment.")
yolo_model, poseformer_model, device = load_models()
print("Models loaded successfully!")


#  Video Processing
def process_video(video_path, progress=gr.Progress()):
    if video_path is None:
        return None, None

    # Convert input video to compatible format
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    uid = str(int(time.time()))
    tmp_dir = tempfile.gettempdir()
    converted_input = os.path.join(tmp_dir, f'converted_input_{uid}.mp4')
    subprocess.run([ffmpeg_path, '-y', '-i', video_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', converted_input], capture_output=True)
    video_path = converted_input

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_queue = deque(maxlen=RECEPTIVE_FIELD)
    dummy_pose = np.zeros((17, 2))
    for _ in range(RECEPTIVE_FIELD):
        pose_queue.append(dummy_pose)

    tmp_2d = os.path.join(tmp_dir, f'output_2d_{uid}.mp4')
    tmp_3d = os.path.join(tmp_dir, f'output_3d_{uid}.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_2d = cv2.VideoWriter(tmp_2d, fourcc, fps, (width, height))
    out_3d = cv2.VideoWriter(tmp_3d, fourcc, fps, (500, 500))

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if total_frames > 0:
            progress((frame_idx + 1) / total_frames, desc=f"Processing frame {frame_idx + 1}/{total_frames}")

        results = yolo_model.predict(source=frame, verbose=False)
        raw_pose = extract_best_person_pose(results)

        if raw_pose is not None:
            h36m_pose = yolo_to_h36m(raw_pose)
            norm_pose = normalize_2d_pose(h36m_pose, width, height)
            pose_queue.append(norm_pose)
        else:
            pose_queue.append(pose_queue[-1])

        pose_sequence = np.array(pose_queue)
        input_tensor = torch.tensor(pose_sequence, dtype=torch.float32).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            predicted_3d = poseformer_model(input_tensor)
            pose_3d = predicted_3d.squeeze().cpu().numpy()

        annotated = results[0].plot()
        out_2d.write(annotated)

        skeleton_img = render_3d_skeleton_cv2(pose_3d)
        out_3d.write(skeleton_img)

        frame_idx += 1

    cap.release()
    out_2d.release()
    out_3d.release()

    # Re-encode for browser playback
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_2d_h264 = os.path.join(tmp_dir, f'output_2d_h264_{uid}.mp4')
    tmp_3d_h264 = os.path.join(tmp_dir, f'output_3d_h264_{uid}.mp4')
    subprocess.run([ffmpeg_path, '-y', '-i', tmp_2d, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', tmp_2d_h264], capture_output=True)
    subprocess.run([ffmpeg_path, '-y', '-i', tmp_3d, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', tmp_3d_h264], capture_output=True)

    return tmp_2d_h264, tmp_3d_h264


def process_webcam_frame(frame):
    if frame is None:
        return None, None

    if not hasattr(process_webcam_frame, 'pose_queue'):
        process_webcam_frame.pose_queue = deque(maxlen=RECEPTIVE_FIELD)
        dummy = np.zeros((17, 2))
        for _ in range(RECEPTIVE_FIELD):
            process_webcam_frame.pose_queue.append(dummy)

    height, width = frame.shape[:2]

    results = yolo_model.predict(source=frame, verbose=False)
    raw_pose = extract_best_person_pose(results)

    if raw_pose is not None:
        h36m_pose = yolo_to_h36m(raw_pose)
        norm_pose = normalize_2d_pose(h36m_pose, width, height)
        process_webcam_frame.pose_queue.append(norm_pose)
    else:
        process_webcam_frame.pose_queue.append(process_webcam_frame.pose_queue[-1])

    pose_sequence = np.array(process_webcam_frame.pose_queue)
    input_tensor = torch.tensor(pose_sequence, dtype=torch.float32).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        predicted_3d = poseformer_model(input_tensor)
        pose_3d = predicted_3d.squeeze().cpu().numpy()

    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    skeleton_img = render_3d_skeleton_cv2(pose_3d)
    skeleton_rgb = cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2RGB)

    return annotated_rgb, skeleton_rgb


# Gradio Interface 
with gr.Blocks(
    title="3D Human Pose Estimation",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple")
) as demo:

    gr.Markdown(
        """
        # 3D Human Pose Estimation
        ### YOLO Pose (2D Detection) → PoseFormerV2 (3D Lifting)
        Upload a video or use your webcam to see real-time 2D and 3D pose estimation.
        """
    )

    with gr.Tabs():
        with gr.Tab("Upload Video"):
            with gr.Row():
                video_input = gr.Video(sources=["upload"], label="Upload a Video")

            process_btn = gr.Button("Run Pose Estimation", variant="primary", size="lg")

            with gr.Row():
                video_2d = gr.Video(label="2D Pose Detection (YOLO)")
                video_3d = gr.Video(label="3D Pose Estimation (PoseFormerV2)")

            process_btn.click(
                fn=process_video,
                inputs=[video_input],
                outputs=[video_2d, video_3d]
            )
        with gr.Tab("Webcam (Capture)"):
                with gr.Tabs():
                    with gr.Tab("Snapshot"):
                        gr.Markdown("Click the **camera icon** to take a snapshot, then click **Run Pose Estimation on Frame** to see results.")

                        with gr.Row():
                            webcam_input = gr.Image(sources=["webcam"], label="Webcam Input", type="numpy")

                        capture_btn = gr.Button("Run Pose Estimation on Frame", variant="primary", size="lg")

                        with gr.Row():
                            webcam_2d = gr.Image(label="2D Pose Detection")
                            webcam_3d = gr.Image(label="3D Skeleton")

                        capture_btn.click(
                            fn=process_webcam_frame,
                            inputs=[webcam_input],
                            outputs=[webcam_2d, webcam_3d]
                        )

                    with gr.Tab("Video Recording"):
                        gr.Markdown("Record a video from your webcam, then click **Run Pose Estimation** to process it.")

                        with gr.Row():
                            webcam_video = gr.Video(sources=["webcam"], label="Record from Webcam")

                        record_btn = gr.Button("Run Pose Estimation", variant="primary", size="lg")

                        with gr.Row():
                            record_2d = gr.Video(label="2D Pose Detection (YOLO)")
                            record_3d = gr.Video(label="3D Pose Estimation (PoseFormerV2)")

                        record_btn.click(
                            fn=process_video,
                            inputs=[webcam_video],
                            outputs=[record_2d, record_3d]
                        )
                        
        with gr.Tab("Live Stream"):
            gr.Markdown("Live webcam streaming — frames are processed in real-time as they come in.")

            with gr.Row():
                stream_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam Input", type="numpy")

            with gr.Row():
                stream_2d = gr.Image(label="2D Pose Detection")
                stream_3d = gr.Image(label="3D Skeleton")

            stream_input.stream(
                fn=process_webcam_frame,
                inputs=[stream_input],
                outputs=[stream_2d, stream_3d]
            )
        
    gr.Markdown(
        """
        ---s
        **Pipeline:** YOLO v26m-Pose detects 2D keypoints → mapped to Human3.6M format →
        PoseFormerV2 lifts 2D sequence to 3D pose using frequency-domain transformer.

        Built for CSCI 4052U — Machine Learning II Final Project
        """
    )


if __name__ == "__main__":
    demo.launch(share=True)
