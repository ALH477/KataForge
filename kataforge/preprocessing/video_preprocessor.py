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
Video Preprocessing Pipeline
Handles camera calibration, pixel-to-meter conversion, multi-angle sync
"""

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    np = None

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from ..core.error_handling import ProcessingError


class CameraCalibrator:
    """Calibrate camera for metric measurements"""
    
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.focal_length = None
        self.pixel_to_meter_ratio = None
        
    def calibrate_from_checkerboard(self, 
                                   images: List[np.ndarray],
                                   checkerboard_size: Tuple[int, int] = (9, 6),
                                   square_size_m: float = 0.025) -> Dict:
        """
        Calibrate camera using checkerboard pattern
        
        Args:
            images: List of calibration images
            checkerboard_size: (width, height) in internal corners
            square_size_m: Physical size of checkerboard squares in meters
        """
        if not OPENCV_AVAILABLE:
            raise ProcessingError("OpenCV is required for camera calibration")
            
        # Prepare object points (0,0,0), (1,0,0), (2,0,0), ...
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                               0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size_m
        
        obj_points = []  # 3D points in real world
        img_points = []  # 2D points in image plane
        
        for img in images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            if ret:
                obj_points.append(objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners_refined)
        
        if not obj_points:
            raise ProcessingError("No valid checkerboard patterns found in images")
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.focal_length = camera_matrix[0, 0]
        
        return {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.tolist(),
            'focal_length': float(self.focal_length),
            'reprojection_error': ret,
        }
    
    def calibrate_from_known_height(self, 
                                   frame: np.ndarray,
                                   pixel_height: int,
                                   real_height_m: float) -> float:
        """
        Simple calibration using known object height
        
        Args:
            frame: Video frame
            pixel_height: Height in pixels of known object
            real_height_m: Real height in meters
            
        Returns:
            Pixel to meter conversion ratio
        """
        if pixel_height <= 0:
            raise ProcessingError("Pixel height must be positive")
        
        if real_height_m <= 0:
            raise ProcessingError("Real height must be positive")
            
        self.pixel_to_meter_ratio = real_height_m / pixel_height
        return self.pixel_to_meter_ratio
    
    def calibrate_from_reference_object(self,
                                       frame: np.ndarray,
                                       reference_points: List[Tuple[int, int]],
                                       real_distance_m: float) -> float:
        """
        Calibrate using two reference points with known distance
        Common: Use 2-meter marking on floor
        """
        if len(reference_points) != 2:
            raise ProcessingError("Reference points must be a list of exactly 2 points")
            
        if real_distance_m <= 0:
            raise ProcessingError("Real distance must be positive")
        
        p1, p2 = reference_points
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        if pixel_distance <= 0:
            raise ProcessingError("Reference points must be different")
            
        self.pixel_to_meter_ratio = real_distance_m / pixel_distance
        return self.pixel_to_meter_ratio
    
    def save_calibration(self, filepath: str):
        """Save calibration data"""
        data = {
            'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            'dist_coeffs': self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            'focal_length': float(self.focal_length) if self.focal_length else None,
            'pixel_to_meter_ratio': float(self.pixel_to_meter_ratio) if self.pixel_to_meter_ratio else None,
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_calibration(self, filepath: str):
        """Load calibration data"""
        if not Path(filepath).exists():
            raise ProcessingError(f"Calibration file not found: {filepath}")
            
        with open(filepath) as f:
            data = json.load(f)
        
        if data['camera_matrix']:
            self.camera_matrix = np.array(data['camera_matrix'])
        if data['dist_coeffs']:
            self.dist_coeffs = np.array(data['dist_coeffs'])
        self.focal_length = data['focal_length']
        self.pixel_to_meter_ratio = data['pixel_to_meter_ratio']
    
    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Remove lens distortion from frame"""
        if not OPENCV_AVAILABLE:
            return frame
            
        if self.camera_matrix is None or self.dist_coeffs is None:
            return frame
        
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
    
    def pixel_to_meters(self, pixel_coords: np.ndarray, depth_m: float = 2.0) -> np.ndarray:
        """
        Convert pixel coordinates to metric coordinates
        
        Args:
            pixel_coords: [N, 2] array of (x, y) pixel coordinates
            depth_m: Estimated depth from camera (default: 2m)
        """
        if self.pixel_to_meter_ratio is not None:
            # Simple scaling method
            return pixel_coords * self.pixel_to_meter_ratio
        elif self.camera_matrix is not None:
            # Pinhole camera model
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            
            metric_coords = np.zeros((len(pixel_coords), 3))
            metric_coords[:, 0] = (pixel_coords[:, 0] - cx) * depth_m / fx
            metric_coords[:, 1] = (pixel_coords[:, 1] - cy) * depth_m / fy
            metric_coords[:, 2] = depth_m
            
            return metric_coords
        else:
            raise ProcessingError("Camera not calibrated")


class VideoPreprocessor:
    """Complete video preprocessing pipeline"""
    
    def __init__(self, calibrator: Optional[CameraCalibrator] = None):
        self.calibrator = calibrator or CameraCalibrator()
        
    def preprocess_video(self,
                        video_path: str,
                        output_path: str,
                        target_fps: int = 60,
                        target_resolution: Tuple[int, int] = (1920, 1080),
                        apply_undistortion: bool = True) -> Dict:
        """
        Preprocess video for training
        
        Returns:
            Metadata dict with video info
        """
        if not OPENCV_AVAILABLE:
            raise ProcessingError("OpenCV is required for video preprocessing")
            
        video_path_obj = Path(video_path)
        output_path_obj = Path(output_path)
        
        if not video_path_obj.exists():
            raise ProcessingError(f"Video file not found: {video_path}")
            
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path_obj))
        
        if not cap.isOpened():
            raise ProcessingError(f"Could not open video file: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path_obj), fourcc, target_fps, target_resolution)
        
        if not out.isOpened():
            raise ProcessingError(f"Could not create output video file: {output_path}")
        
        frame_idx = 0
        processed_frames = 0
        
        # Calculate frame sampling ratio
        sample_ratio = target_fps / original_fps
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames to match target FPS
            if frame_idx * sample_ratio >= processed_frames:
                # Undistort if calibrated
                if apply_undistortion and self.calibrator.camera_matrix is not None:
                    frame = self.calibrator.undistort_frame(frame)
                
                # Resize to target resolution
                if (original_width, original_height) != target_resolution:
                    frame = cv2.resize(frame, target_resolution)
                
                # Apply quality enhancements
                frame = self._enhance_frame(frame)
                
                out.write(frame)
                processed_frames += 1
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        metadata = {
            'original_path': str(video_path_obj),
            'processed_path': str(output_path_obj),
            'original_fps': original_fps,
            'target_fps': target_fps,
            'original_resolution': (original_width, original_height),
            'target_resolution': target_resolution,
            'total_frames': processed_frames,
            'pixel_to_meter_ratio': self.calibrator.pixel_to_meter_ratio,
            'calibrated': self.calibrator.camera_matrix is not None,
        }
        
        return metadata
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply quality enhancements to frame"""
        if not OPENCV_AVAILABLE:
            return frame
            
        # Denoise
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        # Normalize brightness
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
        
        return frame
    
    def batch_preprocess(self,
                        video_dir: str,
                        output_dir: str,
                        **kwargs) -> List[Dict]:
        """Batch process multiple videos"""
        video_dir_obj = Path(video_dir)
        output_dir_obj = Path(output_dir)
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        
        if not video_dir_obj.exists():
            raise ProcessingError(f"Video directory not found: {video_dir}")
        
        metadata_list = []
        
        for video_path in video_dir_obj.glob("*.mp4"):
            output_path = output_dir_obj / f"processed_{video_path.name}"
            print(f"Processing {video_path.name}...")
            
            metadata = self.preprocess_video(
                str(video_path),
                str(output_path),
                **kwargs
            )
            metadata_list.append(metadata)
        
        # Save batch metadata
        with open(output_dir_obj / "preprocessing_metadata.json", 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        return metadata_list