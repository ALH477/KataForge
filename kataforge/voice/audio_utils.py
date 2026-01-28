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

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False
    np = None

from . import VoiceIntent, IntentParser

logger = logging.getLogger(__name__)


class AudioUtils:
    """Utility functions for audio processing."""
    
    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Normalized audio data
        """
        if not NP_AVAILABLE:
            raise RuntimeError("numpy is required for audio processing")
        
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    @staticmethod
    def resample_audio(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """Resample audio to target rate.
        
        Args:
            audio: Audio data as numpy array
            original_rate: Original sample rate
            target_rate: Target sample rate
            
        Returns:
            Resampled audio data
        """
        if not NP_AVAILABLE:
            raise RuntimeError("numpy is required for audio processing")
        
        if original_rate == target_rate:
            return audio
        
        # Simple resampling (in production, use librosa or similar)
        try:
            import scipy
            
            # Calculate new length
            new_length = int(len(audio) * target_rate / original_rate)
            
            # Resample
            resampled = scipy.signal.resample(audio, new_length)
            
            return resampled
            
        except ImportError:
            # Fallback: simple linear interpolation
            ratio = target_rate / original_rate
            new_length = int(len(audio) * ratio)
            
            # Create new array
            resampled = np.zeros(new_length)
            
            # Simple linear interpolation
            for i in range(new_length):
                # Map to original position
                orig_pos = i / ratio
                orig_pos_int = int(orig_pos)
                orig_pos_frac = orig_pos - orig_pos_int
                
                # Linear interpolation
                if orig_pos_int < len(audio) - 1:
                    resampled[i] = audio[orig_pos_int] * (1 - orig_pos_frac) + audio[orig_pos_int + 1] * orig_pos_frac
                else:
                    resampled[i] = audio[-1]
            
            return resampled
    
    @staticmethod
    def convert_to_mono(audio: np.ndarray) -> np.ndarray:
        """Convert stereo audio to mono.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Mono audio data
        """
        if not NP_AVAILABLE:
            raise RuntimeError("numpy is required for audio processing")
        
        # If stereo, convert to mono by averaging channels
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return audio
    
    @staticmethod
    def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
        """Get audio duration in seconds.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            Duration in seconds
        """
        if not NP_AVAILABLE:
            raise RuntimeError("numpy is required for audio processing")
        
        return len(audio) / sample_rate
    
    @staticmethod
    def create_silence(length: int, sample_rate: int) -> np.ndarray:
        """Create silence audio.
        
        Args:
            length: Duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Silence audio data
        """
        if not NP_AVAILABLE:
            raise RuntimeError("numpy is required for audio processing")
        
        return np.zeros(int(length * sample_rate))
    
    @staticmethod
    def add_silence(audio: np.ndarray, duration: float, sample_rate: int, position: str = "end") -> np.ndarray:
        """Add silence to audio.
        
        Args:
            audio: Audio data as numpy array
            duration: Silence duration in seconds
            sample_rate: Audio sample rate
            position: Where to add silence ("start", "end", "both")
            
        Returns:
            Audio with added silence
        """
        if not NP_AVAILABLE:
            raise RuntimeError("numpy is required for audio processing")
        
        # Create silence
        silence = AudioUtils.create_silence(duration, sample_rate)
        
        # Add silence
        if position == "start":
            return np.concatenate([silence, audio])
        elif position == "end":
            return np.concatenate([audio, silence])
        elif position == "both":
            return np.concatenate([silence, audio, silence])
        else:
            return audio
    
    @staticmethod
    def trim_silence(audio: np.ndarray, sample_rate: int, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from audio.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Audio sample rate
            threshold: Silence threshold (0-1)
            
        Returns:
            Trimmed audio data
        """
        if not NP_AVAILABLE:
            raise RuntimeError("numpy is required for audio processing")
        
        # Find non-silent parts
        abs_audio = np.abs(audio)
        non_silent = abs_audio > threshold
        
        # Find start and end of non-silent parts
        if np.any(non_silent):
            start = np.argmax(non_silent)
            end = len(audio) - np.argmax(non_silent[::-1]) - 1
            
            return audio[start:end+1]
        else:
            return audio
    
    @staticmethod
    def get_audio_level(audio: np.ndarray) -> float:
        """Get RMS audio level.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            RMS level (0-1)
        """
        if not NP_AVAILABLE:
            raise RuntimeError("numpy is required for audio processing")
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio**2))
        return rms
    
    @staticmethod
    def adjust_audio_level(audio: np.ndarray, target_level: float = 0.5) -> np.ndarray:
        """Adjust audio level to target level.
        
        Args:
            audio: Audio data as numpy array
            target_level: Target RMS level (0-1)
            
        Returns:
            Adjusted audio data
        """
        if not NP_AVAILABLE:
            raise RuntimeError("numpy is required for audio processing")
        
        # Get current level
        current_level = AudioUtils.get_audio_level(audio)
        
        # Adjust level
        if current_level > 0:
            audio = audio * (target_level / current_level)
        
        return audio
    
    @staticmethod
    def create_audio_buffer(audio: np.ndarray, sample_rate: int, buffer_duration: float = 0.5) -> np.ndarray:
        """Create audio buffer.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Audio sample rate
            buffer_duration: Buffer duration in seconds
            
        Returns:
            Audio with buffer
        """
        if not NP_AVAILABLE:
            raise RuntimeError("numpy is required for audio processing")
        
        # Create buffer
        buffer = AudioUtils.create_silence(buffer_duration, sample_rate)
        
        # Add buffer
        return np.concatenate([buffer, audio, buffer])
