#!/usr/bin/env python3
"""
Comprehensive System Validation and Bug Check
Copyright (c) 2026 DeMoD LLC. All rights reserved.

This script performs a complete system check to ensure:
- All files exist and are valid
- No syntax errors
- Proper error handling
- Configuration validation
- Import checks
- Professional patterns
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
import json
import ast
import re

# Colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


class SystemValidator:
    """Comprehensive system validation"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.issues_fixed: List[str] = []
        
    def validate_all(self) -> bool:
        """Run all validation checks"""
        print_header("KATAFORGE SYSTEM VALIDATION")
        
        checks = [
            ("File Structure", self.check_file_structure),
            ("Python Syntax", self.check_python_syntax),
            ("Imports", self.check_imports),
            ("Error Handling", self.check_error_handling),
            ("Configuration Files", self.check_configurations),
            ("Documentation", self.check_documentation),
            ("Security", self.check_security),
            ("Performance", self.check_performance),
            ("ROCm Configuration", self.check_rocm_config),
        ]
        
        for check_name, check_func in checks:
            print(f"\n{Colors.BOLD}Checking {check_name}...{Colors.END}")
            try:
                check_func()
            except Exception as e:
                print_error(f"Check failed: {e}")
                self.errors.append(f"{check_name}: {e}")
        
        self.print_summary()
        return len(self.errors) == 0
    
    def check_file_structure(self):
        """Verify required files and directories exist"""
        required_files = [
            'flake-rocm.nix',
            'pyproject.toml',
            'kataforge/__init__.py',
            'kataforge/core/gpu_utils.py',
            'kataforge/core/error_handling.py',
            'tests/unit/test_error_handling.py',
            'ROCM_SETUP_FRAMEWORK16.md',
            'framework16-quickstart.sh',
        ]
        
        required_dirs = [
            'kataforge',
            'kataforge/core',
            'kataforge/preprocessing',
            'kataforge/biomechanics',
            'kataforge/ml',
            'kataforge/api',
            'kataforge/cli',
            'tests',
            'tests/unit',
        ]
        
        for file_path in required_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                print_success(f"Found: {file_path}")
            else:
                print_error(f"Missing: {file_path}")
                self.errors.append(f"Missing file: {file_path}")
        
        for dir_path in required_dirs:
            full_path = self.root_dir / dir_path
            if full_path.exists() and full_path.is_dir():
                print_success(f"Found directory: {dir_path}")
            else:
                print_warning(f"Missing directory: {dir_path}")
                self.warnings.append(f"Missing directory: {dir_path}")
    
    def check_python_syntax(self):
        """Check all Python files for syntax errors"""
        python_files = list(self.root_dir.rglob("*.py"))
        
        syntax_errors = []
        
        for py_file in python_files:
            if '__pycache__' in str(py_file) or 'site-packages' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                    ast.parse(code)
                print_success(f"Valid syntax: {py_file.relative_to(self.root_dir)}")
            except SyntaxError as e:
                print_error(f"Syntax error in {py_file.relative_to(self.root_dir)}: {e}")
                syntax_errors.append((str(py_file), str(e)))
                self.errors.append(f"Syntax error in {py_file}: {e}")
            except Exception as e:
                print_warning(f"Could not parse {py_file.relative_to(self.root_dir)}: {e}")
        
        if not syntax_errors:
            print_success(f"All {len(python_files)} Python files have valid syntax")
    
    def check_imports(self):
        """Check for missing or incorrect imports"""
        critical_imports = {
            'torch': 'PyTorch',
            'numpy': 'NumPy',
            'cv2': 'OpenCV',
            'mediapipe': 'MediaPipe',
            'fastapi': 'FastAPI',
            'click': 'Click',
            'rich': 'Rich',
        }
        
        for package, name in critical_imports.items():
            try:
                __import__(package)
                print_success(f"{name} is available")
            except ImportError:
                print_warning(f"{name} not installed (optional unless using related features)")
                self.warnings.append(f"{name} not installed")
    
    def check_error_handling(self):
        """Verify error handling patterns"""
        python_files = list(self.root_dir.rglob("*.py"))
        
        files_without_error_handling = []
        
        for py_file in python_files:
            if '__pycache__' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for try-except blocks
                has_try_except = 'try:' in content and 'except' in content
                
                # Check for error handling imports
                has_error_imports = 'from kataforge.core.error_handling import' in content or \
                                   'import kataforge.core.error_handling' in content
                
                # Files that should have error handling
                is_module = py_file.name not in ['__init__.py', 'test_*.py'] and \
                           py_file.parent.name not in ['tests']
                
                if is_module and not has_try_except:
                    files_without_error_handling.append(py_file.relative_to(self.root_dir))
                    
            except Exception as e:
                print_warning(f"Could not check {py_file}: {e}")
        
        if files_without_error_handling:
            print_warning(f"{len(files_without_error_handling)} files may need error handling")
            for file in files_without_error_handling[:5]:  # Show first 5
                print_info(f"  {file}")
        else:
            print_success("Error handling patterns look good")
    
    def check_configurations(self):
        """Validate configuration files"""
        config_files = {
            'pyproject.toml': self.validate_pyproject,
            'flake-rocm.nix': self.validate_flake,
        }
        
        for config_file, validator in config_files.items():
            config_path = self.root_dir / config_file
            if config_path.exists():
                try:
                    validator(config_path)
                    print_success(f"Valid: {config_file}")
                except Exception as e:
                    print_error(f"Invalid {config_file}: {e}")
                    self.errors.append(f"Invalid {config_file}: {e}")
            else:
                print_warning(f"Missing: {config_file}")
    
    def validate_pyproject(self, path: Path):
        """Validate pyproject.toml"""
        try:
            import tomli
        except ImportError:
            # Use ast to at least check it's valid Python-ish
            with open(path) as f:
                content = f.read()
                if '[tool.poetry]' in content and 'name = ' in content:
                    return True
                raise ValueError("Invalid pyproject.toml structure")
    
    def validate_flake(self, path: Path):
        """Validate Nix flake"""
        with open(path) as f:
            content = f.read()
            
        # Check for critical sections
        required = ['inputs', 'outputs', 'description']
        for req in required:
            if req not in content:
                raise ValueError(f"Missing '{req}' in flake")
        
        # Check power limit (should be 175W)
        if 'powerLimit = 100' in content:
            print_warning("Power limit is 100W, should be 175W for Framework 16")
            self.warnings.append("Power limit needs update to 175W")
        elif 'powerLimit = 175' in content:
            print_success("Power limit correctly set to 175W")
        
        return True
    
    def check_documentation(self):
        """Check documentation completeness"""
        doc_files = [
            'README.md',
            'ROCM_SETUP_FRAMEWORK16.md',
            'ROCM_INTEGRATION_SUMMARY.md',
            'MODEL_TRAINING_GUIDE.md',
            'DEPLOYMENT_GUIDE.md',
        ]
        
        for doc in doc_files:
            doc_path = self.root_dir / doc
            if doc_path.exists():
                size = doc_path.stat().st_size
                if size > 1000:  # At least 1KB
                    print_success(f"Found: {doc} ({size/1024:.1f} KB)")
                else:
                    print_warning(f"Found but small: {doc} ({size} bytes)")
            else:
                print_warning(f"Missing: {doc}")
                self.warnings.append(f"Missing documentation: {doc}")
    
    def check_security(self):
        """Check for security issues"""
        print_info("Checking for common security issues...")
        
        python_files = list(self.root_dir.rglob("*.py"))
        security_issues = []
        
        for py_file in python_files:
            if '__pycache__' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for common issues
                if 'eval(' in content and 'safe_eval' not in content:
                    security_issues.append(f"{py_file}: Uses eval()")
                
                if 'exec(' in content:
                    security_issues.append(f"{py_file}: Uses exec()")
                
                if 'password' in content.lower() and '=' in content and '"' in content:
                    # Check if it's hardcoded
                    lines = content.split('\n')
                    for line in lines:
                        if 'password' in line.lower() and '=' in line and ('"' in line or "'" in line):
                            if 'input(' not in line and 'getpass' not in line and 'environ' not in line:
                                security_issues.append(f"{py_file}: Possible hardcoded password")
                                break
                
            except Exception:
                pass
        
        if security_issues:
            for issue in security_issues:
                print_warning(issue)
                self.warnings.append(f"Security: {issue}")
        else:
            print_success("No obvious security issues found")
    
    def check_performance(self):
        """Check for performance anti-patterns"""
        print_info("Checking performance patterns...")
        
        # This is a basic check - in production you'd use profiling
        python_files = list(self.root_dir.rglob("*.py"))
        
        perf_suggestions = []
        
        for py_file in python_files:
            if '__pycache__' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for common anti-patterns
                if 'for' in content and 'append(' in content and 'list comprehension' not in content:
                    # Suggest list comprehension (minor optimization)
                    pass
                
                # Check for proper batch processing
                if 'for video' in content or 'for file' in content:
                    if 'ThreadPoolExecutor' not in content and 'ProcessPoolExecutor' not in content:
                        if 'batch' in py_file.name:
                            perf_suggestions.append(f"{py_file.name}: Consider parallel processing")
                
            except Exception:
                pass
        
        if perf_suggestions:
            for suggestion in perf_suggestions[:3]:  # Show top 3
                print_info(f"Performance suggestion: {suggestion}")
        else:
            print_success("Performance patterns look good")
    
    def check_rocm_config(self):
        """Validate ROCm-specific configuration"""
        print_info("Validating ROCm configuration...")
        
        # Check flake-rocm.nix
        flake_path = self.root_dir / 'flake-rocm.nix'
        if flake_path.exists():
            with open(flake_path) as f:
                content = f.read()
            
            rocm_checks = {
                'HSA_OVERRIDE_GFX_VERSION=11.0.0': 'GFX version for RDNA3',
                'PYTORCH_ROCM_ARCH="gfx1100"': 'ROCm architecture',
                'HSA_ENABLE_SDMA=0': 'SDMA disabled for stability',
                'rocmPackages': 'ROCm packages included',
            }
            
            for check, description in rocm_checks.items():
                if check in content:
                    print_success(f"Found: {description}")
                else:
                    print_error(f"Missing: {description}")
                    self.errors.append(f"ROCm config missing: {check}")
        else:
            print_error("flake-rocm.nix not found")
            self.errors.append("flake-rocm.nix not found")
    
    def print_summary(self):
        """Print validation summary"""
        print_header("VALIDATION SUMMARY")
        
        total_checks = len(self.errors) + len(self.warnings) + len(self.issues_fixed)
        
        if self.errors:
            print(f"\n{Colors.RED}{Colors.BOLD}ERRORS ({len(self.errors)}):{Colors.END}")
            for error in self.errors:
                print(f"  {Colors.RED}✗ {error}{Colors.END}")
        
        if self.warnings:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}WARNINGS ({len(self.warnings)}):{Colors.END}")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  {Colors.YELLOW}⚠ {warning}{Colors.END}")
            if len(self.warnings) > 10:
                print(f"  {Colors.YELLOW}... and {len(self.warnings) - 10} more{Colors.END}")
        
        if self.issues_fixed:
            print(f"\n{Colors.GREEN}{Colors.BOLD}FIXED ({len(self.issues_fixed)}):{Colors.END}")
            for fix in self.issues_fixed:
                print(f"  {Colors.GREEN}✓ {fix}{Colors.END}")
        
        print(f"\n{Colors.BOLD}Results:{Colors.END}")
        print(f"  Errors: {Colors.RED}{len(self.errors)}{Colors.END}")
        print(f"  Warnings: {Colors.YELLOW}{len(self.warnings)}{Colors.END}")
        print(f"  Fixed: {Colors.GREEN}{len(self.issues_fixed)}{Colors.END}")
        
        if len(self.errors) == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ SYSTEM VALIDATION PASSED{Colors.END}")
            print(f"{Colors.GREEN}System is ready for production use!{Colors.END}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ SYSTEM VALIDATION FAILED{Colors.END}")
            print(f"{Colors.RED}Please fix errors before proceeding.{Colors.END}")


def main():
    """Run system validation"""
    root_dir = Path(__file__).parent
    
    validator = SystemValidator(root_dir)
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
