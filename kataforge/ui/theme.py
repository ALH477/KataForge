"""
Amber Theme for KataForge Gradio UI

A dark theme with amber/gold accent colors inspired by traditional martial arts aesthetics.
Includes animations and accessibility-focused styling for a warm, polished experience.
"""

from __future__ import annotations

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None


# =============================================================================
# Color Palette
# =============================================================================

AMBER_COLORS = {
    "primary": "#FFB000",
    "secondary": "#CC8800",
    "accent": "#FFCC44",
    "background": "#0A0800",
    "text": "#FFB000",
    "waveform": "#FFB000",
}


# =============================================================================
# Theme Creation
# =============================================================================

def create_amber_theme():
    """Create custom amber theme for KataForge UI.
    
    Returns:
        Gradio theme object with amber color scheme
    """
    if not GRADIO_AVAILABLE:
        return None
    
    # Create color scales
    amber_hue = gr.themes.Color(
        c50="#FFF8E6",
        c100="#FFECB3",
        c200="#FFE082",
        c300="#FFD54F",
        c400="#FFCC44",   # accent
        c500="#FFB000",   # primary
        c600="#CC8800",   # secondary
        c700="#A66D00",
        c800="#805300",
        c900="#5A3A00",
        c950="#3D2700",
    )
    
    dark_hue = gr.themes.Color(
        c50="#1A1400",
        c100="#141000",
        c200="#0F0C00",
        c300="#0A0800",   # background
        c400="#080600",
        c500="#050400",
        c600="#030200",
        c700="#020100",
        c800="#010100",
        c900="#000000",
        c950="#000000",
    )
    
    # Create theme
    theme = gr.themes.Base(
        primary_hue=amber_hue,
        secondary_hue=amber_hue,
        neutral_hue=dark_hue,
        font=gr.themes.GoogleFont("JetBrains Mono"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    )
    
    # Apply custom styling
    theme = theme.set(
        # === Background Colors ===
        body_background_fill="#0A0800",
        body_background_fill_dark="#0A0800",
        
        # === Text Colors ===
        body_text_color="#FFB000",
        body_text_color_dark="#FFB000",
        body_text_color_subdued="#CC8800",
        body_text_color_subdued_dark="#CC8800",
        
        # === Block (Card) Styling ===
        block_background_fill="#0F0C00",
        block_background_fill_dark="#0F0C00",
        block_border_color="#CC8800",
        block_border_color_dark="#CC8800",
        block_border_width="1px",
        block_label_background_fill="#141000",
        block_label_background_fill_dark="#141000",
        block_label_text_color="#FFB000",
        block_label_text_color_dark="#FFB000",
        block_label_text_weight="600",
        block_title_text_color="#FFCC44",
        block_title_text_color_dark="#FFCC44",
        block_title_text_weight="700",
        
        # === Input Fields ===
        input_background_fill="#141000",
        input_background_fill_dark="#141000",
        input_border_color="#CC8800",
        input_border_color_dark="#CC8800",
        input_border_color_focus="#FFB000",
        input_border_color_focus_dark="#FFB000",
        input_placeholder_color="#805300",
        input_placeholder_color_dark="#805300",
        
        # === Primary Buttons ===
        button_primary_background_fill="#FFB000",
        button_primary_background_fill_dark="#FFB000",
        button_primary_background_fill_hover="#FFCC44",
        button_primary_background_fill_hover_dark="#FFCC44",
        button_primary_text_color="#0A0800",
        button_primary_text_color_dark="#0A0800",
        button_primary_border_color="#FFB000",
        button_primary_border_color_dark="#FFB000",
        
        # === Secondary Buttons ===
        button_secondary_background_fill="#0F0C00",
        button_secondary_background_fill_dark="#0F0C00",
        button_secondary_background_fill_hover="#141000",
        button_secondary_background_fill_hover_dark="#141000",
        button_secondary_text_color="#FFB000",
        button_secondary_text_color_dark="#FFB000",
        button_secondary_border_color="#CC8800",
        button_secondary_border_color_dark="#CC8800",
        
        # === Cancel Buttons ===
        button_cancel_background_fill="#1A0000",
        button_cancel_background_fill_dark="#1A0000",
        button_cancel_text_color="#FF4444",
        button_cancel_text_color_dark="#FF4444",
        
        # === Tabs ===
        tab_nav_background_fill="#0A0800",
        tab_nav_background_fill_dark="#0A0800",
        
        # === Borders and Shadows ===
        border_color_primary="#CC8800",
        border_color_primary_dark="#CC8800",
        border_color_accent="#FFB000",
        border_color_accent_dark="#FFB000",
        shadow_spread="0px",
        
        # === Sliders / Progress ===
        slider_color="#FFB000",
        slider_color_dark="#FFB000",
        
        # === Checkboxes ===
        checkbox_background_color="#141000",
        checkbox_background_color_dark="#141000",
        checkbox_background_color_selected="#FFB000",
        checkbox_background_color_selected_dark="#FFB000",
        checkbox_border_color="#CC8800",
        checkbox_border_color_dark="#CC8800",
        checkbox_label_text_color="#FFB000",
        checkbox_label_text_color_dark="#FFB000",
        
        # === Tables ===
        table_border_color="#CC8800",
        table_border_color_dark="#CC8800",
        table_even_background_fill="#0F0C00",
        table_even_background_fill_dark="#0F0C00",
        table_odd_background_fill="#0A0800",
        table_odd_background_fill_dark="#0A0800",
        table_row_focus="#1A1400",
        table_row_focus_dark="#1A1400",
        
        # === Panels ===
        panel_background_fill="#0F0C00",
        panel_background_fill_dark="#0F0C00",
        panel_border_color="#CC8800",
        panel_border_color_dark="#CC8800",
        
        # === Code ===
        code_background_fill="#0F0C00",
        code_background_fill_dark="#0F0C00",
        
        # === Misc ===
        loader_color="#FFB000",
        loader_color_dark="#FFB000",
        stat_background_fill="#141000",
        stat_background_fill_dark="#141000",
    )
    
    return theme


# =============================================================================
# CSS Overrides
# =============================================================================

CUSTOM_CSS = """
/* =============================================================================
   KataForge Theme - CSS Animations & Polish
   A warm, accessible experience for martial arts practitioners
   ============================================================================= */

/* -----------------------------------------------------------------------------
   Keyframe Animations
   ----------------------------------------------------------------------------- */

/* Score entrance - scale up with fade */
@keyframes score-entrance {
    0% {
        opacity: 0;
        transform: scale(0.8);
    }
    60% {
        transform: scale(1.05);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}

/* Gentle pulse for waiting states */
@keyframes gentle-pulse {
    0%, 100% {
        opacity: 0.6;
    }
    50% {
        opacity: 1;
    }
}

/* Slide in from left for aspect bars */
@keyframes slide-in-left {
    0% {
        opacity: 0;
        transform: translateX(-20px);
    }
    100% {
        opacity: 1;
        transform: translateX(0);
    }
}

/* Bar fill animation */
@keyframes bar-fill-grow {
    0% {
        width: 0%;
    }
}

/* Subtle glow pulse for highlights */
@keyframes glow-pulse {
    0%, 100% {
        box-shadow: 0 0 0 rgba(255, 176, 0, 0);
    }
    50% {
        box-shadow: 0 0 12px rgba(255, 176, 0, 0.3);
    }
}

/* Fade in for general content */
@keyframes fade-in {
    0% {
        opacity: 0;
    }
    100% {
        opacity: 1;
    }
}

/* -----------------------------------------------------------------------------
   Responsive Design
   ----------------------------------------------------------------------------- */

@media (max-width: 768px) {
    .gradio-container { 
        max-width: 100%; 
        padding: 16px; 
    }
    .input-column { 
        flex-direction: column; 
        gap: 16px; 
    }
    .tab-item { 
        font-size: 1.1rem; 
    }
    .score-display { 
        font-size: 2.5rem !important; 
    }
    .voice-toggle { 
        width: 100%; 
    }
    .feedback-card {
        padding: 12px !important;
    }
}

@media (prefers-color-scheme: dark) {
    body { 
        background-color: #0A0800; 
    }
}

/* Respect reduced motion preferences */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* -----------------------------------------------------------------------------
   Global Styles
   ----------------------------------------------------------------------------- */

* {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Smooth transitions for all interactive elements */
button, input, select, textarea, a, .gradio-button {
    transition: all 0.2s ease-out;
}

/* -----------------------------------------------------------------------------
   Typography
   ----------------------------------------------------------------------------- */

h1, h2, h3, h4, h5, h6 {
    color: #FFCC44 !important;
    font-weight: 700 !important;
}

a {
    color: #FFB000 !important;
    text-decoration: none;
    transition: color 0.2s ease;
}

a:hover {
    color: #FFCC44 !important;
    text-decoration: underline;
}

/* -----------------------------------------------------------------------------
   Focus States (Accessibility)
   ----------------------------------------------------------------------------- */

/* Universal focus ring */
*:focus-visible {
    outline: 2px solid #FFB000 !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 4px rgba(255, 176, 0, 0.2) !important;
}

/* Button focus states */
button:focus-visible,
.gradio-button:focus-visible {
    outline: 2px solid #FFCC44 !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 8px rgba(255, 204, 68, 0.4) !important;
}

/* Input focus states */
input:focus-visible,
textarea:focus-visible,
select:focus-visible {
    border-color: #FFB000 !important;
    box-shadow: 0 0 0 3px rgba(255, 176, 0, 0.15) !important;
}

/* Tab focus */
.tab-nav button:focus-visible {
    outline: 2px solid #FFB000 !important;
    outline-offset: -2px !important;
}

/* -----------------------------------------------------------------------------
   Score Display
   ----------------------------------------------------------------------------- */

.score-display {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 3em !important;
    font-weight: 700 !important;
    text-align: center !important;
    padding: 20px;
    border-radius: 12px;
    animation: score-entrance 0.5s ease-out forwards;
}

/* Score colors based on performance */
.score-excellent {
    color: #44FF88 !important;
    text-shadow: 0 0 20px rgba(68, 255, 136, 0.3);
}

.score-good {
    color: #44FF44 !important;
    text-shadow: 0 0 20px rgba(68, 255, 68, 0.3);
}

.score-medium {
    color: #FFCC44 !important;
    text-shadow: 0 0 20px rgba(255, 204, 68, 0.3);
}

.score-developing {
    color: #FFB000 !important;
    text-shadow: 0 0 20px rgba(255, 176, 0, 0.3);
}

.score-poor {
    color: #FF8844 !important;
    text-shadow: 0 0 20px rgba(255, 136, 68, 0.3);
}

/* Score label below number */
.score-label {
    font-size: 0.4em;
    font-weight: 400;
    opacity: 0.9;
    margin-top: 8px;
    display: block;
}

/* -----------------------------------------------------------------------------
   Waiting / Empty States
   ----------------------------------------------------------------------------- */

.waiting-state {
    text-align: center;
    padding: 32px 20px;
    animation: gentle-pulse 2s ease-in-out infinite;
}

.waiting-state-icon {
    font-size: 2.5em;
    margin-bottom: 12px;
    opacity: 0.7;
}

.waiting-state-text {
    color: #CC8800 !important;
    font-size: 1.1em;
    line-height: 1.6;
}

.waiting-state-hint {
    color: #805300 !important;
    font-size: 0.9em;
    margin-top: 8px;
}

/* -----------------------------------------------------------------------------
   Aspect Score Bars
   ----------------------------------------------------------------------------- */

.aspect-scores-container {
    padding: 16px 0;
}

.aspect-row {
    margin-bottom: 12px;
    animation: slide-in-left 0.4s ease-out forwards;
    opacity: 0;
}

/* Staggered animation delays */
.aspect-row:nth-child(1) { animation-delay: 0.1s; }
.aspect-row:nth-child(2) { animation-delay: 0.2s; }
.aspect-row:nth-child(3) { animation-delay: 0.3s; }
.aspect-row:nth-child(4) { animation-delay: 0.4s; }
.aspect-row:nth-child(5) { animation-delay: 0.5s; }
.aspect-row:nth-child(6) { animation-delay: 0.6s; }

.aspect-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
    color: #FFB000;
    font-size: 0.9em;
}

.aspect-name {
    font-weight: 500;
}

.aspect-value {
    font-weight: 600;
    color: #FFCC44;
}

.aspect-bar {
    height: 12px;
    background: #1A1400;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #332800;
}

.bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.6s ease-out;
    animation: bar-fill-grow 0.6s ease-out;
}

/* Bar fill colors */
.bar-fill-excellent {
    background: linear-gradient(90deg, #22AA44, #44FF88);
}

.bar-fill-good {
    background: linear-gradient(90deg, #228822, #44FF44);
}

.bar-fill-medium {
    background: linear-gradient(90deg, #CC8800, #FFCC44);
}

.bar-fill-developing {
    background: linear-gradient(90deg, #AA6600, #FFB000);
}

.bar-fill-needs-work {
    background: linear-gradient(90deg, #AA4400, #FF8844);
}

/* -----------------------------------------------------------------------------
   Feedback Cards
   ----------------------------------------------------------------------------- */

.feedback-section {
    animation: fade-in 0.4s ease-out forwards;
}

.feedback-card {
    background: #0F0C00;
    border: 1px solid #332800;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.feedback-card:hover {
    border-color: #CC8800;
    transform: translateY(-1px);
}

.feedback-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    font-weight: 600;
    font-size: 1.05em;
}

.feedback-card-icon {
    font-size: 1.2em;
}

/* Strengths card */
.feedback-strengths {
    border-left: 3px solid #44FF44;
}

.feedback-strengths .feedback-card-header {
    color: #44FF44;
}

/* Improvements card */
.feedback-improvements {
    border-left: 3px solid #FFB000;
}

.feedback-improvements .feedback-card-header {
    color: #FFB000;
}

/* Feedback list */
.feedback-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.feedback-list li {
    padding: 8px 0;
    padding-left: 20px;
    position: relative;
    color: #CC8800;
    line-height: 1.5;
    border-bottom: 1px solid #1A1400;
}

.feedback-list li:last-child {
    border-bottom: none;
}

.feedback-list li::before {
    content: '';
    position: absolute;
    left: 0;
    top: 14px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    opacity: 0.6;
}

/* Overall feedback text */
.feedback-text {
    line-height: 1.7 !important;
    color: #FFB000 !important;
}

/* -----------------------------------------------------------------------------
   Technique Tag
   ----------------------------------------------------------------------------- */

.technique-tag {
    display: inline-block;
    background: linear-gradient(135deg, #1A1400, #0F0C00);
    border: 1px solid #CC8800;
    color: #FFB000;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: 500;
    transition: all 0.2s ease;
}

.technique-tag:hover {
    border-color: #FFB000;
    background: linear-gradient(135deg, #241A00, #1A1400);
}

/* -----------------------------------------------------------------------------
   Status Indicators
   ----------------------------------------------------------------------------- */

.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85em;
    font-weight: 500;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: glow-pulse 2s ease-in-out infinite;
}

.status-online {
    color: #44FF44;
}

.status-online .status-dot {
    background: #44FF44;
    box-shadow: 0 0 8px rgba(68, 255, 68, 0.5);
}

.status-offline {
    color: #FF4444;
}

.status-offline .status-dot {
    background: #FF4444;
    box-shadow: 0 0 8px rgba(255, 68, 68, 0.5);
    animation: none;
}

.status-processing {
    color: #FFB000;
}

.status-processing .status-dot {
    background: #FFB000;
    animation: gentle-pulse 1s ease-in-out infinite;
}

/* -----------------------------------------------------------------------------
   Video & Image Display
   ----------------------------------------------------------------------------- */

video {
    border: 2px solid #CC8800 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease;
}

video:hover {
    border-color: #FFB000 !important;
}

.image-container img {
    border: 2px solid #CC8800 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease;
}

.image-container img:hover {
    border-color: #FFB000 !important;
}

/* -----------------------------------------------------------------------------
   Buttons
   ----------------------------------------------------------------------------- */

.gradio-button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.gradio-button:hover {
    transform: translateY(-1px);
}

.gradio-button:active {
    transform: translateY(0);
}

/* Primary button glow on hover */
.gradio-button.primary:hover {
    box-shadow: 0 4px 12px rgba(255, 176, 0, 0.3);
}

/* -----------------------------------------------------------------------------
   Tabs
   ----------------------------------------------------------------------------- */

.tab-nav button {
    transition: all 0.2s ease !important;
    border-radius: 8px 8px 0 0 !important;
}

.tab-nav button.selected {
    background: #FFB000 !important;
    color: #0A0800 !important;
    font-weight: 600 !important;
}

.tab-nav button:not(.selected):hover {
    background: #1A1400 !important;
}

/* -----------------------------------------------------------------------------
   Progress Bars
   ----------------------------------------------------------------------------- */

.progress-bar {
    background: #1A1400;
    border-radius: 6px;
    height: 8px;
    margin: 8px 0;
    overflow: hidden;
}

.progress-fill {
    background: linear-gradient(90deg, #CC8800, #FFB000);
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s ease-out;
}

/* -----------------------------------------------------------------------------
   Voice Toggle
   ----------------------------------------------------------------------------- */

.voice-toggle {
    background: #1A1400;
    border: 2px solid #CC8800;
    border-radius: 24px;
    padding: 8px 16px;
    transition: all 0.2s ease;
}

.voice-toggle:hover {
    border-color: #FFB000;
    background: #241A00;
}

.voice-toggle.active {
    background: #FFB000;
    color: #0A0800;
    border-color: #FFB000;
}

/* -----------------------------------------------------------------------------
   Scrollbars
   ----------------------------------------------------------------------------- */

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0A0800;
}

::-webkit-scrollbar-thumb {
    background: #CC8800;
    border-radius: 4px;
    transition: background 0.2s ease;
}

::-webkit-scrollbar-thumb:hover {
    background: #FFB000;
}

/* Firefox scrollbar */
* {
    scrollbar-width: thin;
    scrollbar-color: #CC8800 #0A0800;
}

/* -----------------------------------------------------------------------------
   Accordion
   ----------------------------------------------------------------------------- */

.accordion {
    border-color: #CC8800 !important;
    border-radius: 8px !important;
    overflow: hidden;
}

.accordion-header {
    transition: background 0.2s ease !important;
}

.accordion-header:hover {
    background: #1A1400 !important;
}

/* -----------------------------------------------------------------------------
   Gallery
   ----------------------------------------------------------------------------- */

.gallery-item {
    border: 1px solid #CC8800 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.gallery-item:hover {
    border-color: #FFB000 !important;
    transform: scale(1.02);
}

/* -----------------------------------------------------------------------------
   Tables
   ----------------------------------------------------------------------------- */

table {
    border-radius: 8px !important;
    overflow: hidden;
}

table th {
    background: #1A1400 !important;
    color: #FFCC44 !important;
    font-weight: 600 !important;
}

table tr {
    transition: background 0.15s ease;
}

table tr:hover {
    background: #1A1400 !important;
}

/* -----------------------------------------------------------------------------
   Markdown Content
   ----------------------------------------------------------------------------- */

.markdown-body {
    color: #FFB000 !important;
    background: transparent !important;
}

.markdown-body h1, 
.markdown-body h2, 
.markdown-body h3 {
    color: #FFCC44 !important;
    border-bottom-color: #CC8800 !important;
}

.markdown-body code {
    background: #141000 !important;
    color: #FFCC44 !important;
    padding: 2px 6px;
    border-radius: 4px;
}

.markdown-body pre {
    background: #0F0C00 !important;
    border: 1px solid #CC8800 !important;
    border-radius: 8px !important;
}

.markdown-body blockquote {
    border-left: 3px solid #CC8800 !important;
    color: #CC8800 !important;
    padding-left: 16px;
}

.markdown-body ul, 
.markdown-body ol {
    padding-left: 24px;
}

.markdown-body li {
    margin-bottom: 8px;
}

/* -----------------------------------------------------------------------------
   Logo Area
   ----------------------------------------------------------------------------- */

.logo-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
}

/* -----------------------------------------------------------------------------
   Tooltips
   ----------------------------------------------------------------------------- */

[title] {
    position: relative;
}

.tooltip {
    background: #1A1400 !important;
    border: 1px solid #CC8800 !important;
    color: #FFB000 !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    font-size: 0.85em !important;
}

/* -----------------------------------------------------------------------------
   Loading States
   ----------------------------------------------------------------------------- */

.loading-spinner {
    border: 3px solid #1A1400;
    border-top-color: #FFB000;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* -----------------------------------------------------------------------------
   Notifications / Alerts
   ----------------------------------------------------------------------------- */

.alert {
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
    animation: fade-in 0.3s ease-out;
}

.alert-success {
    background: rgba(68, 255, 68, 0.1);
    border: 1px solid #44FF44;
    color: #44FF44;
}

.alert-warning {
    background: rgba(255, 176, 0, 0.1);
    border: 1px solid #FFB000;
    color: #FFB000;
}

.alert-error {
    background: rgba(255, 68, 68, 0.1);
    border: 1px solid #FF4444;
    color: #FF4444;
}

.alert-info {
    background: rgba(68, 136, 255, 0.1);
    border: 1px solid #4488FF;
    color: #4488FF;
}
"""
