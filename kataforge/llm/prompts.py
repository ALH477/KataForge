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
Prompt templates for martial arts coaching feedback.

These prompts are designed for:
- LLaVA (vision model) - Frame/technique analysis
- Mistral (text model) - Coaching feedback generation
"""

# =============================================================================
# Vision Analysis Prompts
# =============================================================================

VISION_ANALYSIS_PROMPT = """Analyze this martial arts technique frame. You are an expert martial arts coach.

Technique: {technique}
Coach Style: {coach}

Focus your analysis on:
1. Body positioning and alignment
2. Balance and weight distribution
3. Limb angles and extension
4. Guard position and defensive awareness
5. Common errors visible in the frame

Provide a brief, technical assessment in 2-3 sentences. Be specific about what you observe."""


VISION_KEYFRAME_PROMPT = """Analyze this key frame from a {technique} sequence.

Frame Position: {frame_position} (start/peak/follow-through)

Describe:
1. What is correct in the technique at this moment
2. Any visible errors or areas for improvement
3. How this frame connects to the overall technique flow

Keep response concise (2-3 sentences)."""


VISION_COMPARISON_PROMPT = """Compare this practitioner's {technique} frame to ideal form.

Known issues from biomechanical analysis:
- Speed score: {speed_score}/10
- Force score: {force_score}/10
- Balance score: {balance_score}/10
- Timing score: {timing_score}/10

Based on what you see in this frame, explain what specific adjustments would improve the weakest scoring areas. Be precise and actionable."""


# =============================================================================
# Coaching Feedback Prompts
# =============================================================================

COACHING_FEEDBACK_PROMPT = """You are an expert {style} coach providing feedback on a student's {technique}.

## Analysis Results

**Overall Score:** {overall_score}/10

**Aspect Scores:**
- Speed: {speed_score}/10
- Force: {force_score}/10
- Timing: {timing_score}/10
- Balance: {balance_score}/10
- Coordination: {coordination_score}/10

**Biomechanics Data:**
{biomechanics_summary}

**Visual Analysis (from video frames):**
{vision_analysis}

## Your Task

Provide comprehensive coaching feedback following this structure:

### What You Did Well
List 2-3 specific things the practitioner executed correctly. Be encouraging but honest.

### Areas for Improvement
List 2-3 specific corrections needed. Be precise about body mechanics.

### Drills to Practice
Recommend 2-3 specific drills that will address the identified weaknesses.

### Next Steps
One sentence about what to focus on in the next training session.

Keep your tone supportive and motivating while being technically precise. Use martial arts terminology appropriate for {style}."""


QUICK_FEEDBACK_PROMPT = """As a {style} coach, give brief feedback on this {technique}.

Score: {overall_score}/10
Main issues: {main_issues}

Provide:
1. One thing done well
2. One key correction
3. One drill recommendation

Keep response under 100 words."""


# =============================================================================
# Technique-Specific Prompts
# =============================================================================

TECHNIQUE_PROMPTS = {
    "roundhouse_kick": {
        "focus_points": [
            "Hip rotation and turnover",
            "Supporting foot pivot",
            "Shin angle at impact",
            "Return to guard position",
        ],
        "common_errors": [
            "Insufficient hip rotation",
            "Dropping hands during kick",
            "Not pivoting on support foot",
            "Leaning back instead of rotating",
        ],
    },
    "teep": {
        "focus_points": [
            "Chamber position",
            "Hip thrust forward",
            "Foot placement (ball of foot)",
            "Guard hand position",
        ],
        "common_errors": [
            "Pushing instead of snapping",
            "Telegraphing the kick",
            "Poor balance on support leg",
            "Not retracting quickly",
        ],
    },
    "jab": {
        "focus_points": [
            "Shoulder rotation",
            "Fist alignment at impact",
            "Chin protection with rear hand",
            "Return speed",
        ],
        "common_errors": [
            "Dropping the jab hand before punching",
            "Flaring elbow",
            "Leaning forward excessively",
            "Not rotating shoulder",
        ],
    },
    "cross": {
        "focus_points": [
            "Hip and shoulder rotation",
            "Weight transfer to lead leg",
            "Full extension without overreaching",
            "Chin tucked behind shoulder",
        ],
        "common_errors": [
            "Not rotating hips fully",
            "Dropping rear hand before punch",
            "Overextending and losing balance",
            "Head staying on centerline",
        ],
    },
    "hook": {
        "focus_points": [
            "Elbow angle (90 degrees)",
            "Hip rotation driving the punch",
            "Horizontal fist alignment",
            "Weight shift and pivot",
        ],
        "common_errors": [
            "Arm too straight (becoming a slap)",
            "Arm too tight (losing power)",
            "Not pivoting on lead foot",
            "Dropping opposite hand",
        ],
    },
    "uppercut": {
        "focus_points": [
            "Dropping slightly before rising",
            "Driving up through the legs",
            "Palm facing self at impact",
            "Short, compact motion",
        ],
        "common_errors": [
            "Too much wind-up",
            "Not bending knees for power",
            "Leaning back",
            "Telegraphing by dropping shoulder",
        ],
    },
    "elbow_strike": {
        "focus_points": [
            "Hip rotation",
            "Elbow staying tight to body",
            "Stepping into the strike",
            "Follow-through path",
        ],
        "common_errors": [
            "Using arm strength instead of body rotation",
            "Elbow flaring out",
            "Not closing distance properly",
            "Poor guard during execution",
        ],
    },
    "knee_strike": {
        "focus_points": [
            "Hip thrust forward",
            "Pulling opponent into knee",
            "Rising on support foot",
            "Guard position during strike",
        ],
        "common_errors": [
            "Not driving hips forward",
            "Kneeing straight up instead of through target",
            "Poor clinch control",
            "Losing balance on support leg",
        ],
    },
}


def get_technique_context(technique: str) -> dict:
    """Get technique-specific context for prompts.
    
    Args:
        technique: Technique name (e.g., 'roundhouse_kick')
        
    Returns:
        Dictionary with focus_points and common_errors
    """
    return TECHNIQUE_PROMPTS.get(technique, {
        "focus_points": ["Proper form", "Balance", "Power generation", "Recovery"],
        "common_errors": ["Poor form", "Balance issues", "Lack of power", "Slow recovery"],
    })


def format_biomechanics_summary(biomechanics: dict) -> str:
    """Format biomechanics data for inclusion in prompts.
    
    Args:
        biomechanics: Dictionary of biomechanics measurements
        
    Returns:
        Formatted string for prompt inclusion
    """
    lines = []
    
    key_metrics = [
        ("max_speed", "Max Speed", "m/s"),
        ("peak_force", "Peak Force", "N"),
        ("power_output", "Power Output", "W"),
        ("kinetic_chain_efficiency", "Kinetic Chain Efficiency", "%"),
    ]
    
    for key, label, unit in key_metrics:
        if key in biomechanics:
            value = biomechanics[key]
            if isinstance(value, float):
                lines.append(f"- {label}: {value:.2f} {unit}")
            else:
                lines.append(f"- {label}: {value} {unit}")
    
    return "\n".join(lines) if lines else "- No detailed biomechanics data available"
