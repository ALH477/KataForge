"""KataForge Machine Learning Module.

This module provides ML models and training infrastructure for martial arts
technique analysis.

Models:
- GraphSAGEModel: Graph neural network for skeletal pose classification
- FormAssessor: LSTM+Attention model for technique quality assessment
- TechniqueClassifier: Combined model for classification and assessment
- CoachStyleEncoder: Embeddings for coach-specific style analysis

Training:
- Trainer: Main training orchestrator with LR scheduling, early stopping
- FormAssessorTrainer: Specialized trainer for FormAssessor
- TrainingConfig: Configuration dataclass for training parameters

Inference:
- InferenceEngine: Production inference with automatic fallback
- BiomechanicsAnalyzer: Physics-based analysis without ML

Data:
- MartialArtsDataset: PyTorch Dataset for pose sequences
- create_data_loaders: Factory function for train/val loaders
"""

# Lazy imports to avoid loading PyTorch when not needed
__all__ = [
    # Models
    "GraphSAGEModel",
    "FormAssessor",
    "TechniqueClassifier",
    "CoachStyleEncoder",
    "TemporalAttentionBlock",
    "Scale",
    # Training
    "Trainer",
    "FormAssessorTrainer",
    "TrainingConfig",
    "MultiTaskLoss",
    "FormAssessmentLoss",
    "EarlyStopping",
    "create_trainer",
    # Inference
    "InferenceEngine",
    "BiomechanicsAnalyzer",
    "AnalysisResult",
    "get_inference_engine",
    "reset_inference_engine",
    # Data
    "MartialArtsDataset",
    "create_data_loaders",
]


def __getattr__(name: str):
    """Lazy import attributes on first access."""
    # Models
    if name in ("GraphSAGEModel", "FormAssessor", "TechniqueClassifier", 
                "CoachStyleEncoder", "TemporalAttentionBlock", "Scale"):
        from .models import (
            GraphSAGEModel, FormAssessor, TechniqueClassifier,
            CoachStyleEncoder, TemporalAttentionBlock, Scale
        )
        return locals()[name]
    
    # Training
    if name in ("Trainer", "FormAssessorTrainer", "TrainingConfig", 
                "MultiTaskLoss", "FormAssessmentLoss", "EarlyStopping", 
                "create_trainer"):
        from .trainer import (
            Trainer, FormAssessorTrainer, TrainingConfig,
            MultiTaskLoss, FormAssessmentLoss, EarlyStopping,
            create_trainer
        )
        return locals()[name]
    
    # Inference
    if name in ("InferenceEngine", "BiomechanicsAnalyzer", "AnalysisResult",
                "get_inference_engine", "reset_inference_engine"):
        from .inference import (
            InferenceEngine, BiomechanicsAnalyzer, AnalysisResult,
            get_inference_engine, reset_inference_engine
        )
        return locals()[name]
    
    # Data
    if name in ("MartialArtsDataset", "create_data_loaders"):
        from .data_loader import MartialArtsDataset, create_data_loaders
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def check_dependencies() -> dict:
    """Check which ML dependencies are available.
    
    Returns:
        Dict with availability status of each dependency
    """
    status = {
        "torch": False,
        "torch_geometric": False,
        "tensorboard": False,
    }
    
    try:
        import torch
        status["torch"] = True
        status["torch_version"] = torch.__version__
        status["cuda_available"] = torch.cuda.is_available()
        if status["cuda_available"]:
            status["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    
    try:
        import torch_geometric
        status["torch_geometric"] = True
        status["torch_geometric_version"] = torch_geometric.__version__
    except ImportError:
        pass
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        status["tensorboard"] = True
    except ImportError:
        pass
    
    return status
