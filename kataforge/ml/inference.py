"""
Inference Engine for KataForge

Handles model loading and real-time technique analysis.
When no trained model is available, falls back to biomechanics-only analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from ..core.settings import get_settings
from ..core.error_handling import ModelLoadingError, ModelInferenceError


@dataclass
class AnalysisResult:
    """Result of technique analysis."""
    overall_score: float
    aspect_scores: Dict[str, float]
    corrections: List[str]
    recommendations: List[str]
    biomechanics: Dict[str, float]
    model_used: bool = False  # True if ML model was used, False if biomechanics-only
    confidence: float = 1.0


class BiomechanicsAnalyzer:
    """
    Biomechanics-only analyzer for when no ML model is available.
    
    Provides deterministic analysis based on physics calculations.
    """
    
    # Aspect names and their computation methods
    ASPECTS = ['speed', 'force', 'timing', 'balance', 'coordination']
    
    # MediaPipe landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    
    def __init__(self, fps: float = 30.0):
        """
        Initialize biomechanics analyzer.
        
        Args:
            fps: Frames per second of input video
        """
        self.fps = fps
        self.dt = 1.0 / fps
    
    def analyze(self, poses: np.ndarray, technique: str = "unknown") -> AnalysisResult:
        """
        Analyze pose sequence using biomechanics calculations.
        
        Args:
            poses: Pose data [frames, 33 landmarks, 4 coords (x,y,z,visibility)]
            technique: Technique name for context-specific analysis
            
        Returns:
            AnalysisResult with scores and recommendations
        """
        if len(poses) < 3:
            return self._insufficient_data_result()
        
        # Calculate biomechanics metrics
        biomechanics = self._calculate_biomechanics(poses)
        
        # Calculate aspect scores based on biomechanics
        aspect_scores = self._calculate_aspect_scores(poses, biomechanics)
        
        # Calculate overall score (weighted average)
        weights = {'speed': 0.2, 'force': 0.25, 'timing': 0.2, 'balance': 0.2, 'coordination': 0.15}
        overall_score = sum(aspect_scores[k] * weights[k] for k in self.ASPECTS)
        
        # Generate corrections based on low scores
        corrections = self._generate_corrections(aspect_scores, biomechanics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(technique, aspect_scores)
        
        return AnalysisResult(
            overall_score=round(overall_score, 2),
            aspect_scores={k: round(v, 2) for k, v in aspect_scores.items()},
            corrections=corrections,
            recommendations=recommendations,
            biomechanics={k: round(v, 3) for k, v in biomechanics.items()},
            model_used=False,
            confidence=0.7  # Biomechanics-only analysis has lower confidence
        )
    
    def _calculate_biomechanics(self, poses: np.ndarray) -> Dict[str, float]:
        """Calculate raw biomechanics metrics from pose data."""
        # Extract positions (x, y, z) without visibility
        positions = poses[:, :, :3]
        
        # Calculate velocities using central difference
        velocities = np.zeros_like(positions)
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * self.dt)
        velocities[0] = (positions[1] - positions[0]) / self.dt
        velocities[-1] = (positions[-1] - positions[-2]) / self.dt
        
        # Calculate accelerations
        accelerations = np.zeros_like(positions)
        if len(positions) >= 3:
            accelerations[1:-1] = (positions[2:] - 2*positions[1:-1] + positions[:-2]) / (self.dt**2)
        
        # Key metrics
        
        # 1. Max velocity (striking limbs - wrists and ankles)
        striking_joints = [self.LEFT_WRIST, self.RIGHT_WRIST, self.LEFT_ANKLE, self.RIGHT_ANKLE]
        max_velocity = 0.0
        for joint in striking_joints:
            joint_vel = np.linalg.norm(velocities[:, joint], axis=1)
            max_velocity = max(max_velocity, float(np.max(joint_vel)))
        
        # 2. Peak force estimate (F = ma, using acceleration magnitude)
        # Assuming average limb mass of ~5kg for striking limbs
        limb_mass = 5.0
        max_acceleration = 0.0
        for joint in striking_joints:
            joint_acc = np.linalg.norm(accelerations[:, joint], axis=1)
            max_acceleration = max(max_acceleration, float(np.max(joint_acc)))
        peak_force = limb_mass * max_acceleration
        
        # 3. Power output (P = F * v)
        power_output = peak_force * max_velocity
        
        # 4. Center of mass stability
        hip_center = (positions[:, self.LEFT_HIP] + positions[:, self.RIGHT_HIP]) / 2
        hip_variance = float(np.var(hip_center[:, :2]))  # XY variance only
        stability = 1.0 / (1.0 + hip_variance * 100)  # Normalize to 0-1
        
        # 5. Kinetic chain efficiency
        # Measure sequential activation from hips -> shoulders -> striking limb
        shoulder_center = (positions[:, self.LEFT_SHOULDER] + positions[:, self.RIGHT_SHOULDER]) / 2
        
        hip_vel = np.linalg.norm(np.diff(hip_center, axis=0), axis=1)
        shoulder_vel = np.linalg.norm(np.diff(shoulder_center, axis=0), axis=1)
        
        # Find peak times
        hip_peak_frame = int(np.argmax(hip_vel)) if len(hip_vel) > 0 else 0
        shoulder_peak_frame = int(np.argmax(shoulder_vel)) if len(shoulder_vel) > 0 else 0
        
        # Good kinetic chain: hip peaks before shoulder
        if shoulder_peak_frame > hip_peak_frame:
            kinetic_chain_efficiency = min(100.0, 70.0 + (shoulder_peak_frame - hip_peak_frame) * 5)
        else:
            kinetic_chain_efficiency = max(50.0, 70.0 - (hip_peak_frame - shoulder_peak_frame) * 5)
        
        # 6. Range of motion
        joint_ranges = []
        for joint in striking_joints:
            joint_range = float(np.max(positions[:, joint, :2]) - np.min(positions[:, joint, :2]))
            joint_ranges.append(joint_range)
        avg_range_of_motion = float(np.mean(joint_ranges))
        
        return {
            'max_velocity': max_velocity,
            'peak_force': peak_force,
            'power_output': power_output,
            'stability': stability,
            'kinetic_chain_efficiency': kinetic_chain_efficiency,
            'range_of_motion': avg_range_of_motion,
        }
    
    def _calculate_aspect_scores(self, poses: np.ndarray, biomechanics: Dict[str, float]) -> Dict[str, float]:
        """Convert biomechanics metrics to 0-10 aspect scores."""
        scores = {}
        
        # Speed score: based on max velocity (normalize to reasonable range)
        # Typical striking velocity: 5-15 m/s (normalized coords might be smaller)
        max_vel = biomechanics['max_velocity']
        scores['speed'] = min(10.0, max(0.0, max_vel * 2))  # Scale factor for normalized coords
        
        # Force score: based on peak force and power
        peak_force = biomechanics['peak_force']
        power = biomechanics['power_output']
        scores['force'] = min(10.0, max(0.0, (peak_force / 500 + power / 1000) * 5))
        
        # Timing score: based on kinetic chain efficiency
        kinetic_eff = biomechanics['kinetic_chain_efficiency']
        scores['timing'] = min(10.0, max(0.0, kinetic_eff / 10))
        
        # Balance score: based on stability
        stability = biomechanics['stability']
        scores['balance'] = min(10.0, max(0.0, stability * 10))
        
        # Coordination score: combination of ROM and kinetic chain
        rom = biomechanics['range_of_motion']
        scores['coordination'] = min(10.0, max(0.0, (rom * 5 + kinetic_eff / 20)))
        
        return scores
    
    def _generate_corrections(self, aspect_scores: Dict[str, float], 
                             biomechanics: Dict[str, float]) -> List[str]:
        """Generate corrections based on low scores."""
        corrections = []
        
        if aspect_scores['timing'] < 6.0:
            corrections.append("Improve hip rotation timing - initiate power from hips before shoulders")
        
        if aspect_scores['balance'] < 6.0:
            corrections.append("Maintain better center of gravity - keep hips stable during technique")
        
        if aspect_scores['coordination'] < 6.0:
            corrections.append("Work on arm and leg synchronization for smoother technique flow")
        
        if aspect_scores['speed'] < 5.0:
            corrections.append("Focus on explosive power generation - practice speed drills")
        
        if aspect_scores['force'] < 5.0:
            corrections.append("Improve force generation through proper body mechanics")
        
        if biomechanics['kinetic_chain_efficiency'] < 70:
            corrections.append("Sequential body segment activation needs work - hip -> core -> limbs")
        
        return corrections[:5]  # Limit to top 5 corrections
    
    def _generate_recommendations(self, technique: str, aspect_scores: Dict[str, float]) -> List[str]:
        """Generate training recommendations."""
        recommendations = []
        
        # General recommendations
        recommendations.append("Practice shadowboxing for 10-15 minutes daily")
        
        # Technique-specific
        technique_lower = technique.lower()
        if 'kick' in technique_lower:
            recommendations.append("Focus on hip flexibility and rotation drills")
            recommendations.append("Practice balance exercises on one leg")
        elif 'punch' in technique_lower or 'jab' in technique_lower:
            recommendations.append("Work on footwork and weight transfer")
            recommendations.append("Practice speed bag for timing")
        elif 'elbow' in technique_lower or 'knee' in technique_lower:
            recommendations.append("Focus on close-range power generation")
            recommendations.append("Practice clinch work and body positioning")
        
        # Score-based recommendations
        lowest_aspect = min(aspect_scores, key=aspect_scores.get)
        if lowest_aspect == 'speed':
            recommendations.append("Incorporate plyometric exercises for explosiveness")
        elif lowest_aspect == 'balance':
            recommendations.append("Add single-leg stability exercises to training")
        elif lowest_aspect == 'timing':
            recommendations.append("Practice with a timing coach or metronome")
        
        return recommendations[:5]
    
    def _insufficient_data_result(self) -> AnalysisResult:
        """Return result when insufficient data is provided."""
        return AnalysisResult(
            overall_score=0.0,
            aspect_scores={k: 0.0 for k in self.ASPECTS},
            corrections=["Insufficient data - need at least 3 frames for analysis"],
            recommendations=["Provide more frames for accurate analysis"],
            biomechanics={},
            model_used=False,
            confidence=0.0
        )


class InferenceEngine:
    """
    Main inference engine for technique analysis.
    
    Loads trained ML models when available, falls back to biomechanics-only
    analysis when no model is found.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model file. If None, uses settings.
        """
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        
        # Initialize components
        self.biomechanics_analyzer = BiomechanicsAnalyzer()
        self.model: Optional[nn.Module] = None if not TORCH_AVAILABLE else None
        self.model_loaded = False
        self.device = self._get_device()
        
        # Try to load model
        self._load_model(model_path)
    
    def _get_device(self) -> str:
        """Determine the best device for inference."""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        device_setting = self.settings.model_device
        
        if device_setting == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        
        return device_setting
    
    def _load_model(self, model_path: Optional[str] = None) -> None:
        """
        Attempt to load a trained model.
        
        Args:
            model_path: Explicit path to model file
        """
        if not TORCH_AVAILABLE:
            self.logger.info("PyTorch not available - using biomechanics-only analysis")
            return
        
        # Determine model path
        if model_path:
            path = Path(model_path)
        elif self.settings.model_path:
            path = Path(self.settings.model_path)
        else:
            # Look in default model directory
            model_dir = self.settings.resolved_model_dir
            path = model_dir / "form_assessor.pt"
        
        # Check if model exists
        if not path.exists():
            self.logger.info(
                f"No model found at {path} - using biomechanics-only analysis. "
                "Train a model to enable ML-powered assessment."
            )
            return
        
        try:
            self.logger.info(f"Loading model from {path}")
            
            # Load model checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # Import model class
            from .models import FormAssessor
            
            # Create model with saved config or defaults
            model_config = checkpoint.get('config', {})
            self.model = FormAssessor(
                pose_dim=model_config.get('pose_dim', 4),
                hidden_dim=model_config.get('hidden_dim', 256),
                num_landmarks=model_config.get('num_landmarks', 33),
                num_aspects=model_config.get('num_aspects', 5),
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load model: {e} - using biomechanics-only analysis")
            self.model = None
            self.model_loaded = False
    
    def analyze(self, poses: np.ndarray, technique: str = "unknown", 
                coach_id: str = "default") -> AnalysisResult:
        """
        Analyze a pose sequence.
        
        Args:
            poses: Pose data [frames, 33 landmarks, 4 coords]
            technique: Name of the technique being performed
            coach_id: Coach ID for style-specific analysis
            
        Returns:
            AnalysisResult with scores, corrections, and recommendations
        """
        # Ensure poses is numpy array with correct shape
        poses = np.array(poses, dtype=np.float32)
        
        if len(poses.shape) == 2:
            # Single frame - reshape to [1, landmarks, coords]
            poses = poses.reshape(1, -1, 4)
        
        if poses.shape[1] != 33 or poses.shape[2] != 4:
            raise ModelInferenceError(
                f"Invalid pose shape: {poses.shape}. Expected (frames, 33, 4)"
            )
        
        # Try ML model first if available
        if self.model_loaded and self.model is not None:
            try:
                return self._ml_analysis(poses, technique)
            except Exception as e:
                self.logger.warning(f"ML inference failed: {e} - falling back to biomechanics")
        
        # Fall back to biomechanics-only analysis
        return self.biomechanics_analyzer.analyze(poses, technique)
    
    def _ml_analysis(self, poses: np.ndarray, technique: str) -> AnalysisResult:
        """Perform ML-based analysis."""
        if not TORCH_AVAILABLE or self.model is None:
            raise ModelInferenceError("ML model not available")
        
        # Convert to tensor
        poses_tensor = torch.tensor(poses, dtype=torch.float32).unsqueeze(0)  # Add batch dim
        poses_tensor = poses_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(poses_tensor)
        
        # Extract scores
        aspect_scores_tensor = output['aspect_scores'].squeeze().cpu().numpy()
        overall_score = float(output['overall_score'].squeeze().cpu().numpy())
        
        aspect_names = ['speed', 'force', 'timing', 'balance', 'coordination']
        aspect_scores = {name: float(score) for name, score in zip(aspect_names, aspect_scores_tensor)}
        
        # Also calculate biomechanics for the report
        biomechanics_result = self.biomechanics_analyzer.analyze(poses, technique)
        
        # Generate corrections based on ML scores
        corrections = self._generate_ml_corrections(aspect_scores)
        
        # Generate recommendations
        recommendations = biomechanics_result.recommendations
        
        return AnalysisResult(
            overall_score=round(overall_score, 2),
            aspect_scores={k: round(v, 2) for k, v in aspect_scores.items()},
            corrections=corrections,
            recommendations=recommendations,
            biomechanics=biomechanics_result.biomechanics,
            model_used=True,
            confidence=0.9
        )
    
    def _generate_ml_corrections(self, aspect_scores: Dict[str, float]) -> List[str]:
        """Generate corrections based on ML model scores."""
        corrections = []
        
        score_corrections = {
            'speed': "Work on explosive power and fast-twitch muscle training",
            'force': "Focus on proper body mechanics for maximum force generation",
            'timing': "Practice with rhythm drills to improve technique timing",
            'balance': "Strengthen core and practice single-leg stability exercises",
            'coordination': "Drill combination sequences for better movement flow",
        }
        
        # Add corrections for low scores (below 7.0)
        for aspect, score in sorted(aspect_scores.items(), key=lambda x: x[1]):
            if score < 7.0:
                corrections.append(f"{aspect.capitalize()} ({score:.1f}/10): {score_corrections[aspect]}")
        
        return corrections[:5]
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if ML model is loaded."""
        return self.model_loaded
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status information."""
        return {
            "model_loaded": self.model_loaded,
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "analysis_mode": "ml" if self.model_loaded else "biomechanics",
        }


# Module-level singleton for convenience
_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get or create the inference engine singleton."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def reset_inference_engine() -> None:
    """Reset the inference engine (useful for testing)."""
    global _engine
    _engine = None
