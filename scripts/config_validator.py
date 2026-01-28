"""
Configuration Validator
Validates all configuration files for correctness and professional standards

Copyright (c) 2026 DeMoD LLC. All rights reserved.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class ValidationResult:
    """Result of validation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]


class ConfigValidator:
    """Validates configuration files"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
    
    def validate_yaml(self, filepath: Path) -> ValidationResult:
        """Validate YAML configuration file"""
        self.errors = []
        self.warnings = []
        self.info = []
        
        # Check file exists
        if not filepath.exists():
            self.errors.append(f"File not found: {filepath}")
            return self._result()
        
        # Load YAML
        try:
            with open(filepath) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML: {e}")
            return self._result()
        
        # Validate structure
        if not isinstance(config, dict):
            self.errors.append("Configuration must be a dictionary")
            return self._result()
        
        # Check for Framework 16 config
        if 'framework16' in str(filepath).lower():
            self._validate_framework16_config(config)
        
        # Check training config
        if 'training' in config:
            self._validate_training_config(config['training'])
        
        # Check hardware config
        if 'hardware' in config:
            self._validate_hardware_config(config['hardware'])
        
        # Check power config
        if 'power' in config:
            self._validate_power_config(config['power'])
        
        # Check paths
        if 'paths' in config:
            self._validate_paths_config(config['paths'])
        
        return self._result()
    
    def _validate_framework16_config(self, config: Dict):
        """Validate Framework 16 specific configuration"""
        if 'power' not in config:
            self.warnings.append("Missing 'power' section for Framework 16")
            return
        
        power = config['power']
        
        # Check power limit
        if 'gpu_power_limit' in power:
            limit = power['gpu_power_limit']
            if limit < 100:
                self.warnings.append(f"GPU power limit {limit}W is low, consider 150-175W")
            elif limit > 180:
                self.errors.append(f"GPU power limit {limit}W exceeds Framework 16 capability (180W max)")
            elif 150 <= limit <= 180:
                self.info.append(f"GPU power limit {limit}W is optimal for Framework 16")
        
        # Check temp targets
        if 'gpu_temp_target' in power:
            temp = power['gpu_temp_target']
            if temp > 95:
                self.errors.append(f"GPU temp target {temp}°C is too high (95°C max safe)")
            elif temp < 70:
                self.warnings.append(f"GPU temp target {temp}°C is conservative, 80-85°C is optimal")
    
    def _validate_training_config(self, config: Dict):
        """Validate training configuration"""
        # Check batch size
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if batch_size < 1:
                self.errors.append("Batch size must be >= 1")
            elif batch_size > 32:
                self.warnings.append(f"Batch size {batch_size} may be too large for 8GB VRAM")
            elif batch_size <= 8:
                self.info.append(f"Batch size {batch_size} is appropriate for 8GB VRAM")
        
        # Check learning rate
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if lr <= 0:
                self.errors.append("Learning rate must be positive")
            elif lr > 0.01:
                self.warnings.append(f"Learning rate {lr} may be too high")
        
        # Check mixed precision
        if 'mixed_precision' in config:
            if not config['mixed_precision']:
                self.warnings.append("Mixed precision disabled - training will be slower")
        
        # Check gradient accumulation
        if 'gradient_accumulation_steps' in config:
            steps = config['gradient_accumulation_steps']
            if steps < 1:
                self.errors.append("Gradient accumulation steps must be >= 1")
            elif steps > 16:
                self.warnings.append(f"Gradient accumulation {steps} is high")
    
    def _validate_hardware_config(self, config: Dict):
        """Validate hardware configuration"""
        # Check GPU memory
        if 'gpu_memory_gb' in config:
            memory = config['gpu_memory_gb']
            if memory <= 0:
                self.errors.append("GPU memory must be positive")
            elif memory < 8:
                self.warnings.append(f"GPU memory {memory}GB is limited")
        
        # Check architecture
        if 'architecture' in config:
            arch = config['architecture']
            valid_archs = ['RDNA3', 'RDNA2', 'RDNA', 'Ampere', 'Turing', 'Pascal']
            if arch not in valid_archs:
                self.warnings.append(f"Unknown architecture: {arch}")
    
    def _validate_power_config(self, config: Dict):
        """Validate power configuration"""
        # Check power limit
        if 'gpu_power_limit' in config:
            limit = config['gpu_power_limit']
            if limit <= 0:
                self.errors.append("Power limit must be positive")
            elif limit < 50:
                self.warnings.append(f"Power limit {limit}W is very low")
        
        # Check temp limits
        if 'gpu_temp_target' in config and 'gpu_temp_max' in config:
            target = config['gpu_temp_target']
            max_temp = config['gpu_temp_max']
            if target >= max_temp:
                self.errors.append(f"Target temp {target}°C must be less than max {max_temp}°C")
    
    def _validate_paths_config(self, config: Dict):
        """Validate paths configuration"""
        required_paths = ['data_root', 'models_dir', 'logs_dir']
        
        for path_key in required_paths:
            if path_key not in config:
                self.warnings.append(f"Missing path configuration: {path_key}")
    
    def _result(self) -> ValidationResult:
        """Create validation result"""
        return ValidationResult(
            valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            info=self.info
        )


def validate_config_file(filepath: str) -> bool:
    """
    Validate a configuration file
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        True if valid, False otherwise
    """
    validator = ConfigValidator()
    result = validator.validate_yaml(Path(filepath))
    
    # Print results
    print(f"\nValidating: {filepath}")
    print("=" * 70)
    
    if result.info:
        print("\nInfo:")
        for info in result.info:
            print(f"  ℹ {info}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  ✗ {error}")
    
    if result.valid:
        print(f"\n✓ Configuration is valid")
    else:
        print(f"\n✗ Configuration has errors")
    
    print("=" * 70)
    
    return result.valid


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python config_validator.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    valid = validate_config_file(config_file)
    
    sys.exit(0 if valid else 1)
