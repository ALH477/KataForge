#!/usr/bin/env python3
"""
System Repair and Setup Script
Automatically fixes common issues and ensures system readiness

Copyright (c) 2026 DeMoD LLC. All rights reserved.
"""

import os
import sys
from pathlib import Path
import shutil
import subprocess

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
END = '\033[0m'


def print_header(text):
    print(f"\n{BLUE}{BOLD}{'='*70}{END}")
    print(f"{BLUE}{BOLD}{text:^70}{END}")
    print(f"{BLUE}{BOLD}{'='*70}{END}\n")


def print_success(text):
    print(f"{GREEN}✓ {text}{END}")


def print_warning(text):
    print(f"{YELLOW}⚠ {text}{END}")


def print_error(text):
    print(f"{RED}✗ {text}{END}")


def print_info(text):
    print(f"{BLUE}ℹ {text}{END}")


class SystemRepairer:
    """Repairs and sets up the system"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.fixes_applied = []
        self.errors = []
    
    def repair_all(self):
        """Run all repair operations"""
        print_header("KATAFORGE SYSTEM REPAIR")
        
        repairs = [
            ("Directory Structure", self.create_missing_directories),
            ("__init__ Files", self.create_init_files),
            ("Configuration Files", self.create_config_files),
            ("Scripts", self.fix_script_permissions),
            ("Power Limits", self.update_power_limits),
            ("Validation", self.validate_system),
        ]
        
        for repair_name, repair_func in repairs:
            print(f"\n{BOLD}Repairing {repair_name}...{END}")
            try:
                repair_func()
            except Exception as e:
                print_error(f"Repair failed: {e}")
                self.errors.append(f"{repair_name}: {e}")
        
        self.print_summary()
    
    def create_missing_directories(self):
        """Create any missing directories"""
        required_dirs = [
            'kataforge',
            'kataforge/core',
            'kataforge/preprocessing',
            'kataforge/biomechanics',
            'kataforge/ml',
            'kataforge/api',
            'kataforge/cli',
            'kataforge/profiles',
            'kataforge/ui',
            'kataforge/voice',
            'kataforge/llm',
            'tests',
            'tests/unit',
            'tests/integration',
            'tests/e2e',
            'config',
            'data',
            'data/raw',
            'data/processed',
            'data/poses',
            'data/metrics',
            'data/splits',
            'models',
            'logs',
            'checkpoints',
        ]
        
        for dir_path in required_dirs:
            full_path = self.root_dir / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                print_success(f"Created: {dir_path}")
                self.fixes_applied.append(f"Created directory: {dir_path}")
            else:
                print_info(f"Exists: {dir_path}")
    
    def create_init_files(self):
        """Create __init__.py files where missing"""
        python_dirs = [
            'kataforge',
            'kataforge/core',
            'kataforge/preprocessing',
            'kataforge/biomechanics',
            'kataforge/ml',
            'kataforge/api',
            'kataforge/cli',
            'kataforge/profiles',
            'tests',
            'tests/unit',
            'tests/integration',
            'tests/e2e',
        ]
        
        init_template = '"""Package initialization"""\n'
        
        for dir_path in python_dirs:
            full_path = self.root_dir / dir_path
            init_file = full_path / '__init__.py'
            
            if full_path.exists() and not init_file.exists():
                init_file.write_text(init_template)
                print_success(f"Created: {dir_path}/__init__.py")
                self.fixes_applied.append(f"Created __init__.py in {dir_path}")
    
    def create_config_files(self):
        """Create missing configuration files"""
        config_dir = self.root_dir / 'config'
        config_dir.mkdir(exist_ok=True)
        
        # Create framework16.yaml if missing
        framework16_config = config_dir / 'framework16.yaml'
        if not framework16_config.exists():
            # Copy from framework16_production.yaml if it exists
            production_config = config_dir / 'framework16_production.yaml'
            if production_config.exists():
                shutil.copy(production_config, framework16_config)
                print_success("Created: config/framework16.yaml")
                self.fixes_applied.append("Created framework16.yaml")
            else:
                print_warning("Production config not found, skipping")
    
    def fix_script_permissions(self):
        """Make scripts executable"""
        scripts = [
            'framework16-quickstart.sh',
            'system_validator.py',
            'config_validator.py',
        ]
        
        for script in scripts:
            script_path = self.root_dir / script
            if script_path.exists():
                os.chmod(script_path, 0o755)
                print_success(f"Made executable: {script}")
                self.fixes_applied.append(f"Made {script} executable")
    
    def update_power_limits(self):
        """Verify power limits are set to 175W"""
        flake_path = self.root_dir / 'flake-rocm.nix'
        if flake_path.exists():
            content = flake_path.read_text()
            if 'powerLimit = 175' in content:
                print_success("Power limit correctly set to 175W")
            else:
                print_warning("Power limit not set to 175W in flake-rocm.nix")
        
        config_dir = self.root_dir / 'config'
        for config_file in config_dir.glob('*.yaml'):
            content = config_file.read_text()
            if 'gpu_power_limit: 175' in content or 'power_limit: 175' in content:
                print_success(f"Power limit OK in {config_file.name}")
            elif 'power' in content.lower():
                print_info(f"Check power settings in {config_file.name}")
    
    def validate_system(self):
        """Run final validation"""
        print_info("Running system validation...")
        
        # Check critical files exist
        critical_files = [
            'flake-rocm.nix',
            'kataforge/core/gpu_utils.py',
            'kataforge/core/error_handling.py',
            'framework16-quickstart.sh',
        ]
        
        all_present = True
        for file_path in critical_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                print_success(f"Found: {file_path}")
            else:
                print_error(f"Missing: {file_path}")
                all_present = False
        
        if all_present:
            print_success("All critical files present")
        else:
            print_error("Some critical files missing")
            self.errors.append("Missing critical files")
    
    def print_summary(self):
        """Print repair summary"""
        print_header("REPAIR SUMMARY")
        
        if self.fixes_applied:
            print(f"\n{GREEN}{BOLD}FIXES APPLIED ({len(self.fixes_applied)}):{END}")
            for fix in self.fixes_applied:
                print(f"  {GREEN}✓ {fix}{END}")
        
        if self.errors:
            print(f"\n{RED}{BOLD}ERRORS ({len(self.errors)}):{END}")
            for error in self.errors:
                print(f"  {RED}✗ {error}{END}")
        
        if len(self.errors) == 0:
            print(f"\n{GREEN}{BOLD}✓ SYSTEM REPAIR COMPLETE{END}")
            print(f"{GREEN}System is ready for use!{END}")
        else:
            print(f"\n{YELLOW}{BOLD}⚠ SYSTEM REPAIR INCOMPLETE{END}")
            print(f"{YELLOW}Some issues remain. Please check manually.{END}")


def main():
    """Main repair function"""
    root_dir = Path(__file__).parent
    
    repairer = SystemRepairer(root_dir)
    repairer.repair_all()
    
    # Print next steps
    print_header("NEXT STEPS")
    print("\n1. Test GPU:")
    print("   ./framework16-quickstart.sh")
    print("\n2. Validate configuration:")
    print("   python config_validator.py config/framework16_production.yaml")
    print("\n3. Run system check:")
    print("   python system_validator.py")
    print("\n4. Start development:")
    print("   nix develop")
    print()
    
    sys.exit(0 if len(repairer.errors) == 0 else 1)


if __name__ == "__main__":
    main()
