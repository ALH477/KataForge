# Voice System Implementation Plan for Dojo Manager

## Overview

This document outlines the voice system implementation for the Dojo Manager application, enabling hands-free interaction for martial artists during training.

## Core Components

### 1. Voice Module Structure
- `voice/` - Main voice module directory
- `voice/__init__.py` - Module exports and base classes
- `voice/stt.py` - Speech-to-Text implementations
- `voice/tts.py` - Text-to-Speech implementations  
- `voice/voice_manager.py` - Main voice system controller
- `voice/audio_utils.py` - Audio processing utilities

### 2. Base Classes

#### STTProvider (Speech-to-Text)
```python
class STTProvider:
    async def transcribe(self, audio, sample_rate) -> str:
        """Convert audio to text"""
        pass
    
    async def is_available(self) -> bool:
        """Check if STT service is available"""
        pass
    
    async def get_languages(self) -> List[str]:
        """Get supported languages"""
        pass
```

#### TTSProvider (Text-to-Speech)
```python
class TTSProvider:
    async def synthesize(self, text) -> Tuple[bytes, int]:
        """Convert text to audio"""
        pass
    
    async def is_available(self) -> bool:
        """Check if TTS service is available"""
        pass
    
    async def get_voices(self) -> List[Dict[str, str]]:
        """Get available voices"""
        pass
    
    async def set_voice(self, voice_id) -> bool:
        """Set the voice to use"""
        pass
```

### 3. Voice Commands

#### Intent Parsing System
The system recognizes these voice commands:
- "analyze my roundhouse kick" → trigger analysis
- "set technique to teep" → select technique
- "repeat feedback" → replay last feedback
- "what's my score" → read score
- "help" → show voice commands
- "voice on/off" → toggle voice mode

### 4. Implementation Strategy

#### STT Providers
- **Whisper** - Primary offline option (recommended for CPU/GPU)
- **Browser STT** - Fallback for web browsers
- **Whisper.cpp** - Vulkan/CPU optimized version

#### TTS Providers  
- **Piper TTS** - Fast, low-latency (CPU/Vulkan)
- **Coqui TTS** - High quality (GPU)
- **Edge TTS** - Cloud fallback (internet required)

### 5. Integration Points

#### Gradio UI Integration
1. Voice toggle button in UI
2. Microphone input component
3. Audio playback component
4. Voice status indicators
5. Voice command parsing and execution

#### Settings Configuration
```python
# Voice settings in settings.py
tts_enabled: bool = True
tts_provider: str = "piper"
tts_voice: str = "en_US-lessac-medium"
stt_enabled: bool = True
stt_provider: str = "whisper"
voice_activation_phrase: str = "hey coach"
```

### 6. Technical Considerations

#### Audio Processing
- Audio normalization and resampling
- Silence detection and trimming
- Volume adjustment
- Buffer management

#### Error Handling
- Graceful degradation when services unavailable
- Clear error messages for users
- Retry mechanisms for network failures

#### Performance
- Asynchronous processing to prevent UI blocking
- Caching of frequently used voices/models
- Efficient audio format conversion

## Future Enhancements

1. **Voice Profile Learning** - Adapt to individual user voice patterns
2. **Multi-language Support** - Expanded language coverage
3. **Custom Voice Cloning** - Personalized coach voices
4. **Gesture Integration** - Combine voice with gesture recognition
5. **Offline Mode** - Fully local processing without internet
6. **Smart Context Awareness** - Understand training context
7. **Progress Tracking** - Voice-based progress reporting

## Deployment Considerations

### Docker Images
- CPU-only: Minimal requirements
- GPU versions: CUDA/ROCm support  
- Vulkan versions: Portable GPU support
- All versions include voice dependencies

### Package Dependencies
- Whisper models (large download)
- Piper TTS voices (moderate download)
- Coqui TTS models (large download)
- Edge TTS (network required)

## Implementation Status

✅ Base architecture defined
✅ Voice command parsing system implemented  
✅ Settings integration completed
❌ Actual STT/TTS implementations pending
❌ UI integration pending