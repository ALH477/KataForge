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

"""Biomechanics calculations for martial arts technique analysis."""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from ..core.error_handling import BiomechanicsError


class BiomechanicsCalculator:
    """Calculate biomechanical metrics from pose data."""
    
    def __init__(self):
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
        
        # Standard human body segment lengths (in meters)
        self.segment_lengths = {
            'upper_arm': 0.17,  # 17% of height
            'forearm': 0.15,    # 15% of height
            'thigh': 0.25,      # 25% of height
            'shin': 0.25,       # 25% of height
            'torso': 0.30,      # 30% of height
        }
    
    def calculate_velocity(self, positions: np.ndarray, fps: float) -> np.ndarray:
        """
        Calculate velocity from position data.
        
        Args:
            positions: Array of positions [frames, joints, 3] (x, y, z)
            fps: Frames per second
            
        Returns:
            Velocity array [frames, joints, 3]
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
            
        dt = 1.0 / fps
        velocity = np.zeros_like(positions)
        
        # Central difference for interior points
        velocity[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
        
        # Forward difference for first point
        velocity[0] = (positions[1] - positions[0]) / dt
        
        # Backward difference for last point
        velocity[-1] = (positions[-1] - positions[-2]) / dt
        
        return velocity
    
    def calculate_acceleration(self, positions: np.ndarray, fps: float) -> np.ndarray:
        """
        Calculate acceleration from position data.
        
        Args:
            positions: Array of positions [frames, joints, 3] (x, y, z)
            fps: Frames per second
            
        Returns:
            Acceleration array [frames, joints, 3]
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
        
        if positions.shape[0] < 3:
            raise BiomechanicsError("Need at least 3 frames for acceleration calculation")
            
        dt = 1.0 / fps
        acceleration = np.zeros_like(positions)
        
        # Central difference for interior points
        acceleration[1:-1] = (positions[2:] - 2*positions[1:-1] + positions[:-2]) / (dt**2)
        
        # Forward difference for first point
        acceleration[0] = (positions[2] - 2*positions[1] + positions[0]) / (dt**2)
        
        # Backward difference for last point
        acceleration[-1] = (positions[-1] - 2*positions[-2] + positions[-3]) / (dt**2)
        
        return acceleration
    
    def calculate_joint_angles(self, positions: np.ndarray, connections: List[Tuple[int, int]]) -> np.ndarray:
        """
        Calculate joint angles from landmark positions.
        
        Args:
            positions: Array of positions [frames, joints, 3] (x, y, z)
            connections: List of (joint1, joint2) tuples defining angle calculations
            
        Returns:
            Angles array [frames, angles]
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
            
        angles = np.zeros((positions.shape[0], len(connections)))
        
        for i, (j1, j2) in enumerate(connections):
            # Vector from joint1 to joint2
            vector = positions[:, j2, :2] - positions[:, j1, :2]  # Use x,y only
            # Angle from horizontal (in radians)
            angles[:, i] = np.arctan2(vector[:, 1], vector[:, 0])
        
        return angles
    
    def calculate_speed(self, positions: np.ndarray, fps: float) -> np.ndarray:
        """
        Calculate speed (magnitude of velocity) for each joint.
        
        Args:
            positions: Array of positions [frames, joints, 3] (x, y, z)
            fps: Frames per second
            
        Returns:
            Speed array [frames, joints]
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
            
        velocity = self.calculate_velocity(positions, fps)
        # Calculate magnitude of velocity vector
        speed = np.linalg.norm(velocity, axis=2)
        return speed
    
    def calculate_force(self, mass: float, acceleration: np.ndarray) -> np.ndarray:
        """
        Calculate force using F = ma.
        
        Args:
            mass: Mass in kg
            acceleration: Acceleration array [frames, joints, 3]
            
        Returns:
            Force array [frames, joints, 3]
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
            
        return mass * acceleration
    
    def calculate_power(self, force: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Calculate power using P = FÂ·v.
        
        Args:
            force: Force array [frames, joints, 3]
            velocity: Velocity array [frames, joints, 3]
            
        Returns:
            Power array [frames, joints]
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
            
        # Dot product of force and velocity vectors
        power = np.sum(force * velocity, axis=2)
        return power
    
    def calculate_kinetic_energy(self, mass: float, velocity: np.ndarray) -> np.ndarray:
        """
        Calculate kinetic energy using KE = 0.5 * m * v^2.
        
        Args:
            mass: Mass in kg
            velocity: Velocity array [frames, joints, 3]
            
        Returns:
            Kinetic energy array [frames, joints]
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
            
        speed_squared = np.sum(velocity**2, axis=2)
        kinetic_energy = 0.5 * mass * speed_squared
        return kinetic_energy
    
    def calculate_momentum(self, mass: float, velocity: np.ndarray) -> np.ndarray:
        """
        Calculate momentum using p = m * v.
        
        Args:
            mass: Mass in kg
            velocity: Velocity array [frames, joints, 3]
            
        Returns:
            Momentum array [frames, joints, 3]
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
            
        return mass * velocity
    
    def calculate_center_of_mass(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        Calculate center of mass for body segments.
        
        Args:
            positions: Array of positions [frames, joints, 3] (x, y, z)
            masses: Array of masses for each joint [joints]
            
        Returns:
            Center of mass array [frames, 3]
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
        
        # Weighted sum of positions
        weighted_positions = positions * masses[np.newaxis, :, np.newaxis]
        total_mass = np.sum(masses)
        
        if total_mass == 0:
            raise BiomechanicsError("Total mass cannot be zero")
        
        com = np.sum(weighted_positions, axis=1) / total_mass
        return com
    
    def calculate_angular_velocity(self, angles: np.ndarray, fps: float) -> np.ndarray:
        """
        Calculate angular velocity from angle data.
        
        Args:
            angles: Array of angles [frames, joints]
            fps: Frames per second
            
        Returns:
            Angular velocity array [frames, joints]
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
            
        dt = 1.0 / fps
        angular_velocity = np.zeros_like(angles)
        
        # Central difference for interior points
        angular_velocity[1:-1] = (angles[2:] - angles[:-2]) / (2 * dt)
        
        # Forward difference for first point
        angular_velocity[0] = (angles[1] - angles[0]) / dt
        
        # Backward difference for last point
        angular_velocity[-1] = (angles[-1] - angles[-2]) / dt
        
        return angular_velocity
    
    def calculate_torque(self, moment_of_inertia: float, angular_acceleration: np.ndarray) -> np.ndarray:
        """
        Calculate torque using Ï„ = I * Î±.
        
        Args:
            moment_of_inertia: Moment of inertia in kgâ‹…mÂ²
            angular_acceleration: Angular acceleration array [frames]
            
        Returns:
            Torque array [frames]
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
            
        return moment_of_inertia * angular_acceleration
    
    def analyze_kinetic_chain(self, positions: np.ndarray, joint_chain: List[int]) -> Dict:
        """
        Analyze kinetic chain efficiency through a sequence of joints.
        
        Args:
            positions: Array of positions [frames, joints, 3] (x, y, z)
            joint_chain: List of joint indices in sequence
            
        Returns:
            Dictionary with kinetic chain metrics
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
            
        # Calculate velocities for each joint in chain
        velocities = []
        for joint in joint_chain:
            vel = np.linalg.norm(self.calculate_velocity(positions[:, joint:joint+1, :], 60), axis=2)
            velocities.append(vel.flatten())
        
        # Calculate timing delays between joints
        delays = []
        for i in range(len(velocities) - 1):
            # Find peak velocity times
            peak1 = np.argmax(velocities[i])
            peak2 = np.argmax(velocities[i+1])
            delay = abs(peak2 - peak1)
            delays.append(delay)
        
        # Efficiency score (lower delays = higher efficiency)
        avg_delay = np.mean(delays) if delays else 0
        efficiency = max(0, 100 - avg_delay)  # Scale 0-100%
        
        return {
            'chain_efficiency': efficiency,
            'delays': delays,
            'avg_delay': avg_delay,
            'joint_velocities': velocities
        }
    
    def calculate_all_metrics(self, pose_data: Dict, subject_height: float = 1.75, 
                             subject_mass: float = 70.0) -> Dict:
        """
        Calculate comprehensive biomechanics metrics from pose data.
        
        Args:
            pose_data: Dictionary containing pose data
            subject_height: Subject height in meters
            subject_mass: Subject mass in kg
            
        Returns:
            Dictionary with all biomechanics metrics
        """
        if not NUMPY_AVAILABLE:
            raise BiomechanicsError("NumPy is required for biomechanics calculations")
            
        poses = pose_data['poses']  # [frames, 33, 4] (x, y, z, visibility)
        positions = poses[:, :, :3]  # Just x, y, z
        fps = pose_data.get('fps', 60)
        
        # Calculate basic metrics
        velocity = self.calculate_velocity(positions, fps)
        acceleration = self.calculate_acceleration(positions, fps)
        speed = self.calculate_speed(positions, fps)
        
        # Calculate forces and energies
        force = self.calculate_force(subject_mass, acceleration)
        power = self.calculate_power(force, velocity)
        kinetic_energy = self.calculate_kinetic_energy(subject_mass, velocity)
        momentum = self.calculate_momentum(subject_mass, velocity)
        
        # Calculate aggregate metrics
        metrics = {
            'mean_speed': float(np.mean(speed)),
            'max_speed': float(np.max(speed)),
            'mean_force': float(np.mean(np.linalg.norm(force, axis=2))),
            'max_force': float(np.max(np.linalg.norm(force, axis=2))),
            'mean_power': float(np.mean(power)),
            'max_power': float(np.max(power)),
            'mean_kinetic_energy': float(np.mean(kinetic_energy)),
            'total_energy_expenditure': float(np.sum(kinetic_energy)),
            'avg_momentum': float(np.mean(np.linalg.norm(momentum, axis=2))),
        }
        
        # Add frame-by-frame data for detailed analysis
        detailed_metrics = {
            'speed_by_frame': speed.tolist(),
            'power_by_frame': power.tolist(),
            'kinetic_energy_by_frame': kinetic_energy.tolist(),
        }
        
        return {**metrics, **detailed_metrics}
    
    def save_metrics(self, metrics: Dict, output_path: str):
        """Save biomechanics metrics to file."""
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def load_metrics(self, filepath: str) -> Dict:
        """Load biomechanics metrics from file."""
        filepath_obj = Path(filepath)
        if not filepath_obj.exists():
            raise BiomechanicsError(f"Metrics file not found: {filepath}")
            
        with open(filepath_obj) as f:
            return json.load(f)