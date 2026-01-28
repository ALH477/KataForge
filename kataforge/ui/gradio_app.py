"""
KataForge Web Interface

A warm, intuitive UI for martial arts technique analysis.
Built with Gradio and styled with an amber theme inspired by traditional dojos.
"""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

from .theme import create_amber_theme, CUSTOM_CSS
import time


# =============================================================================
# Techniques and Styles
# =============================================================================

# Format: (id, display_name) - IDs are used internally, display names shown to users
TECHNIQUES = [
    ("roundhouse_kick", "Roundhouse Kick"),
    ("teep", "Teep (Push Kick)"),
    ("jab", "Jab"),
    ("cross", "Cross"),
    ("hook", "Hook"),
    ("uppercut", "Uppercut"),
    ("elbow_strike", "Elbow Strike"),
    ("knee_strike", "Knee Strike"),
    ("front_kick", "Front Kick"),
    ("side_kick", "Side Kick"),
    ("back_fist", "Back Fist"),
]

STYLES = [
    "Muay Thai",
    "Boxing",
    "Kickboxing",
    "MMA",
    "Karate",
    "Taekwondo",
    "Wing Chun",
    "Jeet Kune Do",
]

# Human-readable aspect names for display
ASPECT_LABELS = {
    "speed": "Speed",
    "force": "Power",
    "timing": "Timing",
    "balance": "Balance",
    "coordination": "Flow",
}


# =============================================================================
# API Client
# =============================================================================

class KataForgeAPIClient:
    """Client for communicating with the KataForge API."""
    
    def __init__(self, api_url: str = "http://localhost:8000", timeout: int = 120):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self._client = None
    
    def _get_client(self):
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for API communication")
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.api_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    def health_check(self) -> bool:
        """Check if API is available."""
        try:
            response = self._get_client().get("/health/live")
            return response.status_code == 200
        except Exception:
            return False
    
    def analyze_video(
        self,
        video_path: str,
        coach_id: str,
        technique: str,
    ) -> Dict[str, Any]:
        """Send video for analysis."""
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")
            
            response = self._get_client().post(
                "/api/v1/analyze",
                json={
                    "video_base64": video_b64,
                    "coach_id": coach_id,
                    "technique": technique,
                },
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Server returned {e.response.status_code}. The video may be too large or in an unsupported format."}
        except FileNotFoundError:
            return {"error": "Video file not found. Please try uploading again."}
        except Exception as e:
            return {"error": f"Connection issue: {str(e)[:100]}"}
    
    def get_coaches(self) -> List[Dict[str, Any]]:
        """Get list of available coaches."""
        try:
            response = self._get_client().get("/api/v1/coaches")
            response.raise_for_status()
            return response.json().get("coaches", [])
        except Exception:
            return []
    
    def close(self):
        if self._client:
            self._client.close()
            self._client = None


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """Client for Ollama/llama.cpp LLM services."""
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        backend: str = "ollama",
        timeout: int = 120,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.backend = backend
        self.timeout = timeout
        self._client = None
    
    def _get_client(self):
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for LLM communication")
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.ollama_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    def health_check(self) -> bool:
        """Check if LLM service is available."""
        try:
            if self.backend == "llamacpp":
                response = self._get_client().get("/health")
            else:
                response = self._get_client().get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def generate_feedback(
        self,
        analysis_data: Dict[str, Any],
        style: str = "Muay Thai",
        technique: str = "roundhouse_kick",
    ) -> str:
        """Generate coaching feedback from analysis."""
        prompt = self._build_prompt(analysis_data, style, technique)
        
        try:
            if self.backend == "llamacpp":
                response = self._get_client().post(
                    "/completion",
                    json={
                        "prompt": prompt,
                        "temperature": 0.7,
                        "n_predict": 1024,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                return response.json().get("content", "")
            else:
                response = self._get_client().post(
                    "/api/generate",
                    json={
                        "model": "mistral:7b",
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                return response.json().get("response", "")
        except Exception:
            return "AI coaching is warming up. Your biomechanics analysis is complete above."
    
    def analyze_frame(self, image_bytes: bytes, technique: str) -> str:
        """Analyze a frame using vision model."""
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        technique_display = technique.replace('_', ' ').title()
        prompt = f"Analyze this martial arts technique frame. The practitioner is performing a {technique_display}. Describe the body positioning, any visible errors, and what is being done well. Keep response to 2-3 sentences."
        
        try:
            if self.backend == "llamacpp":
                response = self._get_client().post(
                    "/completion",
                    json={
                        "prompt": prompt,
                        "image_data": [{"data": image_b64, "id": 0}],
                        "stream": False,
                    },
                )
            else:
                response = self._get_client().post(
                    "/api/generate",
                    json={
                        "model": "llava:7b",
                        "prompt": prompt,
                        "images": [image_b64],
                        "stream": False,
                    },
                )
            response.raise_for_status()
            result = response.json()
            return result.get("response", result.get("content", ""))
        except Exception:
            return "Vision analysis is currently unavailable. Try again in a moment."
    
    def _build_prompt(
        self,
        analysis_data: Dict[str, Any],
        style: str,
        technique: str,
    ) -> str:
        """Build coaching feedback prompt."""
        scores = analysis_data.get("aspect_scores", {})
        overall = analysis_data.get("overall_score", 5.0)
        technique_display = technique.replace('_', ' ').title()
        
        return f"""You are an experienced {style} coach giving feedback to a student on their {technique_display}.

Analysis Results:
- Overall Score: {overall}/10
- Speed: {scores.get('speed', 5.0)}/10
- Power: {scores.get('force', 5.0)}/10
- Timing: {scores.get('timing', 5.0)}/10
- Balance: {scores.get('balance', 5.0)}/10
- Flow: {scores.get('coordination', 5.0)}/10

Give friendly, specific coaching feedback:
1. Start with what they did well (be genuine)
2. Give 2-3 specific corrections (be precise about body mechanics)
3. Suggest 1-2 drills they can practice
4. End with encouragement

Be warm but technically precise. Use {style} terminology where appropriate. Keep it conversational."""
    
    def close(self):
        if self._client:
            self._client.close()
            self._client = None


# =============================================================================
# Main Application Class
# =============================================================================

class KataForgeApp:
    """Main Gradio application for KataForge."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        ollama_url: str = "http://localhost:11434",
        llm_backend: str = "ollama",
    ):
        self.api_client = KataForgeAPIClient(api_url)
        self.llm_client = LLMClient(ollama_url, llm_backend)
    
    def check_services(self) -> Tuple[bool, bool]:
        """Check API and LLM service status."""
        api_ok = self.api_client.health_check()
        llm_ok = self.llm_client.health_check()
        return api_ok, llm_ok
    
    def get_status_html(self) -> str:
        """Get HTML status indicator with text labels."""
        api_ok, llm_ok = self.check_services()
        
        api_indicator = '<span class="status-online">Connected</span>' if api_ok else '<span class="status-offline">Offline</span>'
        llm_indicator = '<span class="status-online">Ready</span>' if llm_ok else '<span class="status-offline">Unavailable</span>'
        
        return f"""
        <div class="status-bar">
            <span class="status-item">
                <span class="status-label">Analysis Engine:</span> {api_indicator}
            </span>
            <span class="status-item">
                <span class="status-label">AI Coach:</span> {llm_indicator}
            </span>
        </div>
        """
    
    def analyze_video(
        self,
        video,
        coach_style: str,
        technique: str,
        include_ai_feedback: bool,
        progress=gr.Progress(),
    ) -> Tuple[str, str, str, str]:
        """Analyze uploaded video and return formatted results."""
        
        # Empty state
        if video is None:
            return (
                '<div class="score-display score-waiting">—</div>',
                '<div class="empty-state">Upload a video to see your breakdown</div>',
                "",
                ""
            )
        
        progress(0.1, desc="Reading your video...")
        
        # Get video path
        if isinstance(video, str):
            video_path = video
        else:
            video_path = video.name if hasattr(video, 'name') else str(video)
        
        progress(0.3, desc="Analyzing your technique...")
        
        # Call API for analysis
        result = self.api_client.analyze_video(
            video_path,
            coach_style.lower().replace(" ", "_"),
            technique,
        )
        
        if "error" in result:
            return (
                '<div class="score-display score-error">!</div>',
                f'<div class="error-message">{result["error"]}</div>',
                "",
                ""
            )
        
        progress(0.6, desc="Crunching the numbers...")
        
        # Format score with animation class
        overall_score = result.get("overall_score", 5.0)
        if overall_score >= 8:
            score_class = "score-good"
            score_message = "Excellent work!"
        elif overall_score >= 6:
            score_class = "score-medium"
            score_message = "Good foundation"
        else:
            score_class = "score-needs-work"
            score_message = "Keep practicing"
        
        score_html = f'''
        <div class="score-container">
            <div class="score-display {score_class}">{overall_score:.1f}</div>
            <div class="score-subtitle">{score_message}</div>
        </div>
        '''
        
        # Format aspect scores with animation
        aspects = result.get("aspect_scores", {})
        aspects_html = self._format_aspects_html(aspects)
        
        # Format corrections
        corrections = result.get("corrections", [])
        if corrections:
            corrections_text = "\n".join(f"• {c}" for c in corrections)
        else:
            corrections_text = "Looking solid! Keep refining the details."
        
        # Generate AI feedback
        ai_feedback = ""
        if include_ai_feedback:
            progress(0.8, desc="Getting coaching feedback...")
            ai_feedback = self.llm_client.generate_feedback(
                result,
                coach_style,
                technique,
            )
        
        progress(1.0, desc="Done!")
        
        return score_html, aspects_html, corrections_text, ai_feedback
    
    def analyze_frame(
        self,
        image,
        technique: str,
    ) -> Tuple[str, str]:
        """Analyze a single frame."""
        if image is None:
            return "Ready", "Upload a frame and I'll break down your form."
        
        # Convert to bytes
        if hasattr(image, 'read'):
            image_bytes = image.read()
        elif isinstance(image, str):
            try:
                with open(image, 'rb') as f:
                    image_bytes = f.read()
            except Exception:
                return "Error", "Could not read that image file."
        else:
            # Assume numpy array from gradio
            import io
            try:
                from PIL import Image as PILImage
                img = PILImage.fromarray(image)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=85)
                image_bytes = buf.getvalue()
            except Exception:
                return "Error", "Could not process that image format. Try a JPG or PNG."
        
        # Get vision analysis
        analysis = self.llm_client.analyze_frame(image_bytes, technique)
        
        return "Analysis Complete", analysis
    
    def _format_aspects_html(self, aspects: Dict[str, float]) -> str:
        """Format aspect scores as animated HTML bars."""
        if not aspects:
            return '<div class="empty-state">No detailed breakdown available</div>'
        
        html_parts = []
        for i, (aspect_id, score) in enumerate(aspects.items()):
            percentage = min(score * 10, 100)  # Cap at 100%
            
            # Determine color based on score
            if score >= 8:
                bar_class = "bar-good"
            elif score >= 6:
                bar_class = "bar-medium"
            else:
                bar_class = "bar-needs-work"
            
            # Get human-readable label
            label = ASPECT_LABELS.get(aspect_id, aspect_id.title())
            
            html_parts.append(f'''
            <div class="aspect-row" style="animation-delay: {i * 0.1}s">
                <div class="aspect-header">
                    <span class="aspect-label">{label}</span>
                    <span class="aspect-score">{score:.1f}</span>
                </div>
                <div class="aspect-bar">
                    <div class="aspect-fill {bar_class}" style="width: {percentage}%"></div>
                </div>
            </div>
            ''')
        
        return '<div class="aspects-container">' + "".join(html_parts) + '</div>'
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        if not GRADIO_AVAILABLE:
            raise RuntimeError("Gradio is required for the UI")
        
        theme = create_amber_theme()
        
        with gr.Blocks(
            title="KataForge",
            theme=theme,
            css=CUSTOM_CSS,
        ) as app:
            # Header
            gr.HTML("""
            <div class="app-header">
                <h1>🥋 KataForge</h1>
                <p class="tagline">Your personal training partner</p>
            </div>
            """)
            
            # Status bar
            status_html = gr.HTML(self.get_status_html())
            with gr.Row():
                refresh_btn = gr.Button("↻ Refresh", size="sm", elem_classes="refresh-btn")
            refresh_btn.click(fn=self.get_status_html, outputs=status_html)
            
            with gr.Tabs() as tabs:
                # =============================================================
                # Tab 1: Video Analysis
                # =============================================================
                with gr.TabItem("🎬 Analyze", id="analyze"):
                    gr.HTML('<p class="tab-description">Upload a video of your technique and get instant feedback</p>')
                    
                    with gr.Row():
                        # Left column: Input
                        with gr.Column(scale=1):
                            video_input = gr.Video(
                                label="Your Video",
                                sources=["upload"],
                                elem_classes="video-upload",
                            )
                            
                            gr.HTML('<p class="input-hint">MP4, MOV, or AVI • Keep it under 30 seconds for best results</p>')
                            
                            with gr.Row():
                                coach_style = gr.Dropdown(
                                    choices=STYLES,
                                    value="Muay Thai",
                                    label="Style",
                                    elem_classes="style-dropdown",
                                )
                                
                                technique = gr.Dropdown(
                                    choices=[(t[1], t[0]) for t in TECHNIQUES],  # (label, value)
                                    value="roundhouse_kick",
                                    label="Technique",
                                    elem_classes="technique-dropdown",
                                )
                            
                            include_ai = gr.Checkbox(
                                label="Include AI coaching tips",
                                value=True,
                                elem_classes="ai-checkbox",
                            )
                            
                            analyze_btn = gr.Button(
                                "Analyze My Form",
                                variant="primary",
                                size="lg",
                                elem_classes="analyze-btn",
                            )
                        
                        # Right column: Results
                        with gr.Column(scale=2):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.HTML('<h3 class="section-title">Score</h3>')
                                    score_output = gr.HTML(
                                        value='<div class="score-display score-waiting">—</div><div class="score-subtitle">Upload a video to begin</div>',
                                        elem_classes="score-container",
                                    )
                                
                                with gr.Column(scale=2):
                                    gr.HTML('<h3 class="section-title">Breakdown</h3>')
                                    aspects_output = gr.HTML(
                                        value='<div class="empty-state">Your detailed scores will appear here</div>',
                                        elem_classes="aspects-container",
                                    )
                            
                            gr.HTML('<h3 class="section-title">Corrections</h3>')
                            corrections_output = gr.Textbox(
                                label="",
                                lines=3,
                                interactive=False,
                                placeholder="Specific adjustments will appear here after analysis",
                                elem_classes="corrections-box",
                            )
                            
                            gr.HTML('<h3 class="section-title">Coach\'s Notes</h3>')
                            feedback_output = gr.Textbox(
                                label="",
                                lines=8,
                                interactive=False,
                                placeholder="AI coaching feedback will appear here",
                                elem_classes="feedback-box",
                            )
                    
                    # Wire up analyze button
                    analyze_btn.click(
                        fn=self.analyze_video,
                        inputs=[video_input, coach_style, technique, include_ai],
                        outputs=[score_output, aspects_output, corrections_output, feedback_output],
                    )
                
                # =============================================================
                # Tab 2: Frame Analysis
                # =============================================================
                with gr.TabItem("📸 Snapshot", id="snapshot"):
                    gr.HTML('<p class="tab-description">Get quick feedback on a single frame or photo</p>')
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(
                                label="Your Frame",
                                type="numpy",
                                elem_classes="image-upload",
                            )
                            
                            gr.HTML('<p class="input-hint">Upload a photo or screenshot of your technique</p>')
                            
                            frame_technique = gr.Dropdown(
                                choices=[(t[1], t[0]) for t in TECHNIQUES],
                                value="roundhouse_kick",
                                label="Technique",
                            )
                            
                            frame_analyze_btn = gr.Button(
                                "Analyze Frame",
                                variant="primary",
                                elem_classes="analyze-btn",
                            )
                        
                        with gr.Column(scale=2):
                            frame_status = gr.Textbox(
                                label="Status",
                                value="Ready",
                                interactive=False,
                                elem_classes="status-box",
                            )
                            
                            frame_analysis = gr.Textbox(
                                label="Analysis",
                                lines=8,
                                interactive=False,
                                value="Upload a frame and I'll break down your form.",
                                elem_classes="analysis-box",
                            )
                    
                    frame_analyze_btn.click(
                        fn=self.analyze_frame,
                        inputs=[image_input, frame_technique],
                        outputs=[frame_status, frame_analysis],
                    )
                
                # =============================================================
                # Tab 3: About
                # =============================================================
                with gr.TabItem("ℹ️ About", id="about"):
                    gr.Markdown("""
                    ## About KataForge
                    
                    **Built by martial artists, for martial artists.**
                    
                    KataForge helps you train smarter between sessions. Upload your technique 
                    videos and get instant feedback on what you're doing well and what needs work.
                    
                    ### What We Analyze
                    
                    - **Speed** — Are you generating velocity efficiently?
                    - **Power** — Is your force transfer solid?
                    - **Timing** — Does your technique flow naturally?
                    - **Balance** — Are you grounded and controlled?
                    - **Flow** — How well do your movements connect?
                    
                    ### How It Works
                    
                    KataForge uses computer vision to track your body movement, then applies 
                    biomechanics calculations to score each aspect. The AI coach layer 
                    (powered by Mistral) translates those numbers into practical feedback.
                    
                    ### Supported Techniques
                    
                    Kicks, punches, elbows, knees — the fundamentals that matter. We're adding 
                    more techniques regularly.
                    
                    ---
                    
                    *"The master has failed more times than the beginner has tried."*
                    
                    ---
                    
                    <small>KataForge is part of the Dojo Manager project — preserving martial arts 
                    knowledge through technology.</small>
                    """)
        
        return app
    
    def close(self):
        """Clean up resources."""
        self.api_client.close()
        self.llm_client.close()


# =============================================================================
# Factory Functions
# =============================================================================

def create_gradio_app(
    api_url: str = "http://localhost:8000",
    ollama_url: str = "http://localhost:11434",
    llm_backend: str = "ollama",
) -> gr.Blocks:
    """Create configured Gradio app.
    
    Args:
        api_url: KataForge API URL
        ollama_url: Ollama/llama.cpp URL
        llm_backend: 'ollama' or 'llamacpp'
        
    Returns:
        Configured Gradio Blocks app
    """
    app = KataForgeApp(
        api_url=api_url,
        ollama_url=ollama_url,
        llm_backend=llm_backend,
    )
    return app.create_interface()


def launch_ui(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    api_url: str = "http://localhost:8000",
    ollama_url: str = "http://localhost:11434",
    llm_backend: str = "ollama",
    **kwargs,
):
    """Launch the KataForge UI server.
    
    Args:
        host: Server host
        port: Server port
        share: Create public Gradio link
        api_url: KataForge API URL
        ollama_url: Ollama/llama.cpp URL
        llm_backend: LLM backend type
        **kwargs: Additional Gradio launch arguments
    """
    if not GRADIO_AVAILABLE:
        raise RuntimeError("Gradio is required. Install with: pip install gradio")
    
    interface = create_gradio_app(
        api_url=api_url,
        ollama_url=ollama_url,
        llm_backend=llm_backend,
    )
    
    interface.launch(
        server_name=host,
        server_port=port,
        share=share,
        **kwargs,
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point for standalone UI."""
    import os
    
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    api_url = os.getenv("KATAFORGE_API_URL", "http://localhost:8000")
    ollama_url = os.getenv("KATAFORGE_LLM_URL", "http://localhost:11434")
    llm_backend = os.getenv("KATAFORGE_LLM_BACKEND", "ollama")
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    
    print(f"\n🥋 Starting KataForge on http://{host}:{port}\n")
    
    launch_ui(
        host=host,
        port=port,
        share=share,
        api_url=api_url,
        ollama_url=ollama_url,
        llm_backend=llm_backend,
    )


if __name__ == "__main__":
    main()
