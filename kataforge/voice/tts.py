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
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False
    np = None

from . import STTProvider, TTSProvider, VoiceIntent, IntentParser

logger = logging.getLogger(__name__)


class PiperTTS(TTSProvider):
    """Text-to-Speech using Piper model."""
    
    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        device: str = "auto",
    ):
        """Initialize Piper TTS.
        
        Args:
            voice: Voice model to use
            device: Device to use (auto, cpu, cuda, rocm)
        """
        self.voice = voice
        self.device = device
        self.model = None
        self._initialized = False
        
    async def is_available(self) -> bool:
        """Check if Piper TTS is available."""
        if not NP_AVAILABLE:
            return False
        
        try:
            import piper  # type: ignore
            return True
        except ImportError:
            return False
    
    async def get_voices(self) -> List[Dict[str, str]]:
        """Get available voices."""
        # List of common Piper voices
        return [
            {"name": "en_US-lessac-medium", "description": "American English (Lessac)"},
            {"name": "en_US-vctk-medium", "description": "American English (VCTK)"},
            {"name": "en_GB-vctk-medium", "description": "British English (VCTK)"},
            {"name": "es_ES-medium", "description": "Spanish (Spain)"},
            {"name": "fr_FR-medium", "description": "French (France)"},
            {"name": "de_DE-medium", "description": "German (Germany)"},
            {"name": "it_IT-medium", "description": "Italian (Italy)"},
            {"name": "ja_JP-medium", "description": "Japanese (Japan)"},
            {"name": "ko_KR-medium", "description": "Korean (Korea)"},
            {"name": "zh_CN-medium", "description": "Chinese (China)"},
        ]
    
    async def set_voice(self, voice_id: str) -> bool:
        """Set the voice to use.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            True if voice was set successfully
        """
        self.voice = voice_id
        return True
    
    async def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not NP_AVAILABLE:
            raise RuntimeError("numpy is required for Piper TTS")
        
        try:
            import piper  # type: ignore
            
            # Initialize model if not already done
            if not self._initialized:
                logger.info("Loading Piper voice: %s", self.voice)
                self.model = piper.PiperVoice.load(
                    f"{self.voice}.onnx",
                    config_path=f"{self.voice}.json",
                    use_cuda=self.device == "cuda",
                )
                self._initialized = True
            
            # Synthesize speech
            audio = self.model.synthesize(text)
            
            # Return audio data and sample rate
            return audio, 22050  # Piper typically uses 22.05kHz
            
        except Exception as e:
            logger.error("Piper TTS synthesis failed", exc_info=e)
            raise RuntimeError(f"Piper TTS synthesis failed: {e}")


class CoquiTTS(TTSProvider):
    """Text-to-Speech using Coqui TTS (XTTS)."""
    
    def __init__(
        self,
        voice: str = "v2_en",
        device: str = "auto",
    ):
        """Initialize Coqui TTS.
        
        Args:
            voice: Voice model to use
            device: Device to use (auto, cpu, cuda, rocm)
        """
        self.voice = voice
        self.device = device
        self.model = None
        self._initialized = False
        
    async def is_available(self) -> bool:
        """Check if Coqui TTS is available."""
        if not NP_AVAILABLE:
            return False
        
        try:
            import TTS  # type: ignore
            return True
        except ImportError:
            return False
    
    async def get_voices(self) -> List[Dict[str, str]]:
        """Get available voices."""
        # List of common Coqui voices
        return [
            {"name": "v2_en", "description": "English (default)"},
            {"name": "v2_es", "description": "Spanish"},
            {"name": "v2_fr", "description": "French"},
            {"name": "v2_de", "description": "German"},
            {"name": "v2_it", "description": "Italian"},
            {"name": "v2_ja", "description": "Japanese"},
            {"name": "v2_ko", "description": "Korean"},
            {"name": "v2_zh", "description": "Chinese"},
            {"name": "v2_ru", "description": "Russian"},
            {"name": "v2_ar", "description": "Arabic"},
        ]
    
    async def set_voice(self, voice_id: str) -> bool:
        """Set the voice to use.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            True if voice was set successfully
        """
        self.voice = voice_id
        return True
    
    async def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not NP_AVAILABLE:
            raise RuntimeError("numpy is required for Coqui TTS")
        
        try:
            from TTS.api import TTS as CoquiTTS  # type: ignore
            
            # Initialize model if not already done
            if not self._initialized:
                logger.info("Loading Coqui TTS voice: %s", self.voice)
                self.model = CoquiTTS(
                    model_name=f"tts_models/{self.voice}/vits",
                    progress_bar=False,
                    gpu=self.device == "cuda",
                )
                self._initialized = True
            
            # Synthesize speech
            audio = self.model.tts(text)
            
            # Return audio data and sample rate
            return np.array(audio), 22050  # Coqui typically uses 22.05kHz
            
        except Exception as e:
            logger.error("Coqui TTS synthesis failed", exc_info=e)
            raise RuntimeError(f"Coqui TTS synthesis failed: {e}")


class EdgeTTS(TTSProvider):
    """Text-to-Speech using Microsoft Edge TTS (cloud-based)."""
    
    def __init__(
        self,
        voice: str = "en-US-JennyNeural",
        language: str = "en-US",
    ):
        """Initialize Edge TTS.
        
        Args:
            voice: Voice model to use
            language: Language code
        """
        self.voice = voice
        self.language = language
        
    async def is_available(self) -> bool:
        """Check if Edge TTS is available."""
        try:
            import edge_tts  # type: ignore
            return True
        except ImportError:
            return False
    
    async def get_voices(self) -> List[Dict[str, str]]:
        """Get available voices."""
        # List of common Edge voices
        return [
            {"name": "en-US-JennyNeural", "description": "English (US) - Jenny"},
            {"name": "en-US-GuyNeural", "description": "English (US) - Guy"},
            {"name": "en-GB-SoniaNeural", "description": "English (UK) - Sonia"},
            {"name": "es-ES-ElviraNeural", "description": "Spanish (Spain) - Elvira"},
            {"name": "fr-FR-DeniseNeural", "description": "French (France) - Denise"},
            {"name": "de-DE-KatjaNeural", "description": "German (Germany) - Katja"},
            {"name": "it-IT-ElsaNeural", "description": "Italian (Italy) - Elsa"},
            {"name": "ja-JP-NanamiNeural", "description": "Japanese (Japan) - Nanami"},
            {"name": "ko-KR-SunHiNeural", "description": "Korean (Korea) - SunHi"},
            {"name": "zh-CN-XiaoxiaoNeural", "description": "Chinese (China) - Xiaoxiao"},
        ]
    
    async def set_voice(self, voice_id: str) -> bool:
        """Set the voice to use.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            True if voice was set successfully
        """
        self.voice = voice_id
        return True
    
    async def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Edge TTS")
        
        try:
            import edge_tts  # type: ignore
            
            # Create audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name
            
            try:
                # Generate TTS
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(tmp_path)
                
                # Load audio file
                import soundfile as sf  # type: ignore
                audio, sample_rate = sf.read(tmp_path)
                
                # Return audio data and sample rate
                return audio, sample_rate
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            logger.error("Edge TTS synthesis failed", exc_info=e)
            raise RuntimeError(f"Edge TTS synthesis failed: {e}")
