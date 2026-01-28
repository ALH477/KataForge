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
Gradio Web UI Module for Dojo Manager

Provides a beautiful web interface for:
- Video upload and technique analysis
- Frame-by-frame analysis with vision AI
- AI-generated coaching feedback
- Coach profile management
"""

from .gradio_app import (
    DojoGradioApp,
    create_gradio_app,
    launch_ui,
)
from .theme import (
    AMBER_COLORS,
    create_amber_theme,
)

__all__ = [
    # App
    "DojoGradioApp",
    "create_gradio_app",
    "launch_ui",
    # Theme
    "AMBER_COLORS",
    "create_amber_theme",
]
