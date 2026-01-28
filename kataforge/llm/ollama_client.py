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
Ollama and llama.cpp Client for Multimodal AI Coaching

Supports:
- Ollama API for vision (LLaVA) and text (Mistral) models
- llama.cpp server API for Vulkan-based inference
- Async operations for non-blocking UI
"""

from __future__ import annotations

import asyncio
import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    import ollama as ollama_lib
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama_lib = None

from .prompts import (
    VISION_ANALYSIS_PROMPT,
    COACHING_FEEDBACK_PROMPT,
    QUICK_FEEDBACK_PROMPT,
    get_technique_context,
    format_biomechanics_summary,
)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def analyze_image(self, image: Union[str, bytes], prompt: str, **kwargs) -> str:
        """Analyze an image with a text prompt."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM service is available."""
        pass


class OllamaClient(BaseLLMClient):
    """Client for Ollama API (supports both vision and text models)."""
    
    def __init__(
        self,
        host: str = "http://localhost:11434",
        vision_model: str = "llava:7b",
        text_model: str = "mistral:7b",
        timeout: int = 120,
    ):
        """Initialize Ollama client.
        
        Args:
            host: Ollama API URL
            vision_model: Model for image analysis (e.g., llava:7b)
            text_model: Model for text generation (e.g., mistral:7b)
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip("/")
        self.vision_model = vision_model
        self.text_model = text_model
        self.timeout = timeout
        
        if HTTPX_AVAILABLE:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                timeout=httpx.Timeout(timeout),
            )
        else:
            self._client = None
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate text using the text model.
        
        Args:
            prompt: Input prompt
            model: Model to use (defaults to text_model)
            system: System prompt
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Ollama client")
        
        model = model or self.text_model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama text generation failed: {e}")
    
    async def analyze_image(
        self,
        image: Union[str, bytes, Path],
        prompt: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Analyze an image using the vision model.
        
        Args:
            image: Image as base64 string, bytes, or file path
            prompt: Analysis prompt
            model: Model to use (defaults to vision_model)
            
        Returns:
            Vision analysis response
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Ollama client")
        
        model = model or self.vision_model
        
        # Convert image to base64 if needed
        if isinstance(image, Path) or (isinstance(image, str) and Path(image).exists()):
            with open(image, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image, bytes):
            image_b64 = base64.b64encode(image).decode("utf-8")
        else:
            image_b64 = image  # Assume already base64
        
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        
        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama vision analysis failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if Ollama is available and models are loaded."""
        if not HTTPX_AVAILABLE:
            return False
        
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """List available models."""
        if not HTTPX_AVAILABLE:
            return []
        
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
    
    async def pull_model(self, model: str) -> bool:
        """Pull a model if not available."""
        if not HTTPX_AVAILABLE:
            return False
        
        try:
            response = await self._client.post(
                "/api/pull",
                json={"name": model, "stream": False},
                timeout=httpx.Timeout(600),  # 10 minutes for pull
            )
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


class LlamaCppClient(BaseLLMClient):
    """Client for llama.cpp server API (OpenAI-compatible).
    
    Used for Vulkan-based inference with llama.cpp.
    """
    
    def __init__(
        self,
        host: str = "http://localhost:8080",
        timeout: int = 120,
    ):
        """Initialize llama.cpp client.
        
        Args:
            host: llama.cpp server URL
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip("/")
        self.timeout = timeout
        
        if HTTPX_AVAILABLE:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                timeout=httpx.Timeout(timeout),
            )
        else:
            self._client = None
    
    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Generate text using llama.cpp server.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for llama.cpp client")
        
        # Use OpenAI-compatible completion endpoint
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
            "stream": False,
        }
        
        try:
            response = await self._client.post("/completion", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("content", "")
        except Exception as e:
            raise RuntimeError(f"llama.cpp text generation failed: {e}")
    
    async def analyze_image(
        self,
        image: Union[str, bytes, Path],
        prompt: str,
        **kwargs,
    ) -> str:
        """Analyze an image using llama.cpp multimodal model.
        
        Args:
            image: Image as base64 string, bytes, or file path
            prompt: Analysis prompt
            
        Returns:
            Vision analysis response
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for llama.cpp client")
        
        # Convert image to base64 if needed
        if isinstance(image, Path) or (isinstance(image, str) and Path(image).exists()):
            with open(image, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image, bytes):
            image_b64 = base64.b64encode(image).decode("utf-8")
        else:
            image_b64 = image
        
        # llama.cpp multimodal format
        payload = {
            "prompt": prompt,
            "image_data": [{"data": image_b64, "id": 0}],
            "stream": False,
        }
        
        try:
            response = await self._client.post("/completion", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("content", "")
        except Exception as e:
            raise RuntimeError(f"llama.cpp vision analysis failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if llama.cpp server is available."""
        if not HTTPX_AVAILABLE:
            return False
        
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


def get_llm_client(
    backend: str = "ollama",
    **kwargs,
) -> BaseLLMClient:
    """Factory function to get the appropriate LLM client.
    
    Args:
        backend: 'ollama' or 'llamacpp'
        **kwargs: Arguments passed to client constructor
        
    Returns:
        Configured LLM client instance
    """
    if backend == "llamacpp":
        return LlamaCppClient(**kwargs)
    else:
        return OllamaClient(**kwargs)


async def generate_coaching_feedback(
    client: BaseLLMClient,
    analysis_data: Dict[str, Any],
    frames: Optional[List[Union[str, bytes]]] = None,
    coach_style: str = "Muay Thai",
    technique: str = "roundhouse_kick",
    quick: bool = False,
) -> Dict[str, str]:
    """Generate comprehensive coaching feedback using multimodal AI.
    
    Args:
        client: LLM client (Ollama or llama.cpp)
        analysis_data: Analysis results from dojo-api
        frames: Optional list of key frame images for vision analysis
        coach_style: Martial art style for context
        technique: Technique being analyzed
        quick: If True, generate shorter feedback
        
    Returns:
        Dictionary with 'vision_analysis' and 'feedback' keys
    """
    result = {
        "vision_analysis": "",
        "feedback": "",
    }
    
    # Get technique-specific context
    technique_context = get_technique_context(technique)
    
    # Extract scores from analysis data
    scores = analysis_data.get("aspect_scores", {})
    overall_score = analysis_data.get("overall_score", 5.0)
    biomechanics = analysis_data.get("biomechanics", {})
    
    # Step 1: Vision analysis of frames (if provided)
    if frames:
        vision_analyses = []
        frame_positions = ["start", "peak", "follow-through"]
        
        for i, frame in enumerate(frames[:3]):  # Analyze up to 3 key frames
            position = frame_positions[i] if i < len(frame_positions) else f"frame_{i}"
            
            prompt = VISION_ANALYSIS_PROMPT.format(
                technique=technique.replace("_", " "),
                coach=coach_style,
            )
            
            try:
                analysis = await client.analyze_image(frame, prompt)
                vision_analyses.append(f"**{position.title()}:** {analysis}")
            except Exception as e:
                vision_analyses.append(f"**{position.title()}:** Analysis unavailable ({e})")
        
        result["vision_analysis"] = "\n\n".join(vision_analyses)
    
    # Step 2: Generate coaching feedback
    if quick:
        # Quick feedback for UI responsiveness
        prompt = QUICK_FEEDBACK_PROMPT.format(
            style=coach_style,
            technique=technique.replace("_", " "),
            overall_score=overall_score,
            main_issues=", ".join(technique_context.get("common_errors", [])[:2]),
        )
    else:
        # Comprehensive feedback
        prompt = COACHING_FEEDBACK_PROMPT.format(
            style=coach_style,
            technique=technique.replace("_", " "),
            overall_score=overall_score,
            speed_score=scores.get("speed", 5.0),
            force_score=scores.get("force", 5.0),
            timing_score=scores.get("timing", 5.0),
            balance_score=scores.get("balance", 5.0),
            coordination_score=scores.get("coordination", 5.0),
            biomechanics_summary=format_biomechanics_summary(biomechanics),
            vision_analysis=result["vision_analysis"] or "No frame analysis available",
        )
    
    try:
        result["feedback"] = await client.generate_text(prompt)
    except Exception as e:
        result["feedback"] = f"Unable to generate feedback: {e}"
    
    return result


# Convenience function for synchronous usage
def generate_coaching_feedback_sync(
    analysis_data: Dict[str, Any],
    frames: Optional[List[Union[str, bytes]]] = None,
    coach_style: str = "Muay Thai",
    technique: str = "roundhouse_kick",
    backend: str = "ollama",
    **client_kwargs,
) -> Dict[str, str]:
    """Synchronous wrapper for generate_coaching_feedback.
    
    Args:
        analysis_data: Analysis results from dojo-api
        frames: Optional list of key frame images
        coach_style: Martial art style
        technique: Technique name
        backend: 'ollama' or 'llamacpp'
        **client_kwargs: Arguments for client constructor
        
    Returns:
        Dictionary with feedback
    """
    async def _run():
        client = get_llm_client(backend, **client_kwargs)
        try:
            return await generate_coaching_feedback(
                client,
                analysis_data,
                frames,
                coach_style,
                technique,
            )
        finally:
            await client.close()
    
    return asyncio.run(_run())
