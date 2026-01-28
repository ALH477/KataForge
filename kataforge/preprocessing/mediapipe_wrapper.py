#!/usr/bin/env python3
"""
KataForge - Adaptive Martial Arts Analysis System
Copyright © 2026 DeMoD LLC. All rights reserved.

This file is part of KataForge, released under the KataForge License
(based on Elastic License v2). See LICENSE in the project root for full terms.

SPDX-License-Identifier: Elastic-2.0

Description:
    [Brief module description – please edit]

Usage notes:
    - Private self-hosting, dojo use, and modifications are permitted.
    - Offering as a hosted/managed service to third parties is prohibited
      without explicit written permission from DeMoD LLC.
"""

"""
MediaPipe Pose Estimation Wrapper
Extracts 3D skeletal data from videos
"""

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    mp = None
    np = None

from pathlib import Path
from typing import Dict, List, Optional
import json

from ..core.error_handling import ProcessingError


class MediaPipePoseExtractor:
    """Wrapper around MediaPipe Pose for consistent extraction"""
    
    def __init__(self,
                model_complexity: int = 2,
                min_detection_confidence: float = 0.7,
                min_tracking_confidence: float = 0.7,
                enable_3d: bool = True):
        if not OPENCV_AVAILABLE:
            raise ProcessingError("OpenCV and MediaPipe are required for pose extraction")
            
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.enable_3d = enable_3d
    
    def extract_from_video(self, 
                          video_path: str,
                          pixel_to_meter: Optional[float] = None) -> Dict:
        """
        Extract pose landmarks from entire video
        
        Returns:
            Dict with poses array [num_frames, 33, 4] (x, y, z, visibility)
            and metadata
        """
        if not OPENCV_AVAILABLE:
            raise ProcessingError("OpenCV is not available")
            
        video_path = Path(video_path)
        if not video_path.exists():
            raise ProcessingError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ProcessingError(f"Could not open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        poses = []
        frame_indices = []
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract landmarks
                landmarks = self._extract_landmarks(
                    results.pose_landmarks,
                    results.pose_world_landmarks if self.enable_3d else None,
                    frame.shape,
                    pixel_to_meter
                )
                poses.append(landmarks)
                frame_indices.append(frame_idx)
            else:
                # No pose detected - use previous or zeros
                if poses:
                    poses.append(poses[-1])  # Repeat last valid pose
                else:
                    poses.append(np.zeros((33, 4)))
                frame_indices.append(frame_idx)
            
            frame_idx += 1
        
        cap.release()
        
        poses_array = np.array(poses)  # [num_frames, 33, 4]
        
        return {
            'poses': poses_array,
            'frame_indices': frame_indices,
            'fps': fps,
            'total_frames': len(poses),
            'video_path': str(video_path),
            'landmark_names': self._get_landmark_names(),
        }
    
    def _extract_landmarks(self,
                          pose_landmarks,
                          pose_world_landmarks,
                          frame_shape,
                          pixel_to_meter):
        """Extract landmark coordinates"""
        landmarks = np.zeros((33, 4))
        
        for idx, landmark in enumerate(pose_landmarks.landmark):
            if pose_world_landmarks and self.enable_3d:
                # Use world coordinates (in meters)
                world = pose_world_landmarks.landmark[idx]
                landmarks[idx] = [world.x, world.y, world.z, landmark.visibility]
            else:
                # Use image coordinates
                x = landmark.x * frame_shape[1]  # pixel x
                y = landmark.y * frame_shape[0]  # pixel y
                z = landmark.z * frame_shape[1]  # relative depth
                
                # Convert to meters if calibrated
                if pixel_to_meter:
                    x *= pixel_to_meter
                    y *= pixel_to_meter
                    z *= pixel_to_meter
                
                landmarks[idx] = [x, y, z, landmark.visibility]
        
        return landmarks
    
    def _get_landmark_names(self) -> List[str]:
        """Get landmark names in order"""
        return [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky', 'right_pinky',
            'left_index', 'right_index',
            'left_thumb', 'right_thumb',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]
    
    def visualize_pose(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw pose landmarks on frame"""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    def save_poses(self, pose_data: Dict, output_path: str):
        """Save extracted poses to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy array to list for JSON serialization
        save_data = pose_data.copy()
        save_data['poses'] = pose_data['poses'].tolist()
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f)
    
    def load_poses(self, filepath: str) -> Dict:
        """Load poses from file"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise ProcessingError(f"Pose file not found: {filepath}")
            
        with open(filepath) as f:
            data = json.load(f)
        
        data['poses'] = np.array(data['poses'])
        return data