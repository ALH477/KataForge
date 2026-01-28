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
LLM Integration Module for Dojo Manager

Provides multimodal AI capabilities for:
- Vision analysis of technique frames (LLaVA)
- Natural language coaching feedback (Mistral)

Supports both Ollama and llama.cpp backends.
"""

from .ollama_client import (
    OllamaClient,
    LlamaCppClient,
    get_llm_client,
    generate_coaching_feedback,
)
from .prompts import (
    VISION_ANALYSIS_PROMPT,
    COACHING_FEEDBACK_PROMPT,
    TECHNIQUE_PROMPTS,
)

__all__ = [
    # Clients
    "OllamaClient",
    "LlamaCppClient",
    "get_llm_client",
    # Functions
    "generate_coaching_feedback",
    # Prompts
    "VISION_ANALYSIS_PROMPT",
    "COACHING_FEEDBACK_PROMPT",
    "TECHNIQUE_PROMPTS",
]
