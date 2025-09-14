#!/usr/bin/env python3
"""
Environment Check Script for EnvisionObjectAnnotator Integration
Checks if all required components are properly installed and configured.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path
import importlib.util

class EnvironmentChecker:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.integration_dir = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        
    def check_python_version(self):
        """Check Python version"""
        print("Checking Python version...")
        
        version = sys.version_info
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        
        if version < (3, 8):
            self.errors.append("Python 3.8+ required")
            return False
        elif version < (3, 9):
            self.warnings.append("Python 3.9+ recommended for better performance")
        
        print("   Python version OK")
        return True
    
    def check_required_packages(self):
        """Check if required packages are installed"""
        print("\nChecking required packages...")
        
        required_packages = [
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("cv2", "OpenCV"),
            ("numpy", "NumPy"),
            ("PIL", "Pillow"),
            ("matplotlib", "Matplotlib"),
            ("scipy", "SciPy"),
            ("sklearn", "Scikit-learn"),
            ("pandas", "Pandas"),
            ("tqdm", "TQDM"),
            ("wandb", "Weights & Biases"),
            ("ffmpeg", "FFmpeg Python"),
            ("psutil", "PSUtil")
        ]
        
        missing_packages = []
        
        for package, name in required_packages:
            try:
                if package == "cv2":
                    import cv2
                elif package == "PIL":
                    from PIL import Image
                elif package == "sklearn":
                    import sklearn
                elif package == "ffmpeg":
                    import ffmpeg
                else:
                    importlib.import_module(package)
                
                print(f"   Correct {name}")
                
            except ImportError:
                print(f"   False {name} - Missing")
                missing_packages.append(name)
        
        if missing_packages:
            self.errors.append(f"Missing packages: {', '.join(missing_packages)}")
            return False
        
        return True
    
    def check_cuda_availability(self):
        """Check CUDA availability"""
        print("\nChecking CUDA availability...")
        
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                print(f"   CUDA available")
                print(f"   Devices: {device_count}")
                print(f"   Current device: {device_name}")
                return True
            else:
                print("   CUDA not available - will use CPU")
                self.warnings.append("CUDA not available - performance may be limited")
                return False
                
        except ImportError:
            print("   PyTorch not installed")
            self.errors.append("PyTorch not installed")
            return False
    
    def check_sam2_installation(self):
        """Check SAM2 installation"""
        print("\nChecking SAM2 installation...")
        
        try:
            # Try to import SAM2
            import sam2
            print("   SAM2 package available")
            
            # Check if we can create a predictor
            try:
                from sam2.build_sam import build_sam2
                print("   SAM2 build functions available")
                return True
            except ImportError as e:
                print(f"   SAM2 build functions not available: {e}")
                self.errors.append("SAM2 build functions not available")
                return False
                
        except ImportError:
            print("   SAM2 package not found")
            self.errors.append("SAM2 package not installed")
            return False
    
    def check_checkpoints(self):
        """Check if SAM2 checkpoints are available"""
        print("\nChecking SAM2 checkpoints...")
        
        checkpoints_dir = self.integration_dir / "checkpoints"
        
        if not checkpoints_dir.exists():
            print("   Checkpoints directory not found")
            self.errors.append("Checkpoints directory not found")
            return False
        
        required_checkpoints = [
            "sam2_hiera_large.pt",
            "sam2_hiera_base_plus.pt",
            "sam2_hiera_small.pt"
        ]
        
        available_checkpoints = []
        missing_checkpoints = []
        
        for checkpoint in required_checkpoints:
            checkpoint_path = checkpoints_dir / checkpoint
            if checkpoint_path.exists():
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                print(f"   Correct {checkpoint} ({size_mb:.1f} MB)")
                available_checkpoints.append(checkpoint)
            else:
                print(f"   False {checkpoint} - Missing")
                missing_checkpoints.append(checkpoint)
        
        if missing_checkpoints:
            self.warnings.append(f"Missing checkpoints: {', '.join(missing_checkpoints)}")
        
        return len(available_checkpoints) > 0
    
    def check_ffmpeg(self):
        """Check FFmpeg installation"""
        print("\nChecking FFmpeg...")
        
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"   FFmpeg available: {version_line}")
                return True
            else:
                print("   FFmpeg not working properly")
                self.errors.append("FFmpeg not working properly")
                return False
                
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("   FFmpeg not found")
            self.errors.append("FFmpeg not installed")
            return False
    
    def check_hybrid_pipeline(self):
        """Check if hybrid pipeline is available"""
        print("\nChecking hybrid pipeline...")
        
        hybrid_pipeline_path = self.base_dir / "integrated_temporal_pipeline_hybrid.py"
        
        if hybrid_pipeline_path.exists():
            print("   Hybrid pipeline found")
            
            # Try to import it
            try:
                sys.path.append(str(self.base_dir))
                import integrated_temporal_pipeline_hybrid
                print("   Hybrid pipeline importable")
                return True
            except ImportError as e:
                print(f"  Hybrid pipeline import failed: {e}")
                self.errors.append("Hybrid pipeline import failed")
                return False
        else:
            print("   Hybrid pipeline not found")
            self.errors.append("Hybrid pipeline not found")
            return False
    
    def check_directories(self):
        """Check if required directories exist"""
        print("\nChecking directory structure...")
        
        required_dirs = [
            "setup/",
            "integration/",
            "test_scenarios/",
            "utils/",
            "configs/",
            "output/",
            "logs/"
        ]
        
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.integration_dir / dir_name
            if dir_path.exists():
                print(f"   Correct {dir_name}")
            else:
                print(f"   False {dir_name} - Missing")
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            self.errors.append(f"Missing directories: {', '.join(missing_dirs)}")
            return False
        
        return True
    
    def check_memory_requirements(self):
        """Check memory requirements"""
        print("\nChecking memory requirements...")
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            print(f"   Total memory: {memory_gb:.1f} GB")
            print(f"   Available memory: {available_gb:.1f} GB")
            
            if memory_gb < 8:
                self.warnings.append("Less than 8GB RAM - may cause performance issues")
            elif memory_gb < 16:
                self.warnings.append("Less than 16GB RAM - consider upgrading for better performance")
            else:
                print("   Memory requirements met")
            
            return True
            
        except ImportError:
            print("    psutil not available - cannot check memory")
            self.warnings.append("Cannot check memory requirements")
            return True
    
    def run_complete_check(self):
        """Run complete environment check"""
        print("EnvisionObjectAnnotator Environment Check")
        print("=" * 50)
        
        checks = [
            self.check_python_version,
            self.check_required_packages,
            self.check_cuda_availability,
            self.check_sam2_installation,
            self.check_checkpoints,
            self.check_ffmpeg,
            self.check_hybrid_pipeline,
            self.check_directories,
            self.check_memory_requirements
        ]
        
        all_passed = True
        
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                print(f"   Check failed with error: {e}")
                self.errors.append(f"Check failed: {e}")
                all_passed = False
        
        # Print summary
        print("\n" + "=" * 50)
        print("ENVIRONMENT CHECK SUMMARY")
        print("=" * 50)
        
        if self.errors:
            print("ERRORS:")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if not self.errors and not self.warnings:
            print("All checks passed! Environment is ready.")
        elif not self.errors:
            print("Environment is ready with warnings.")
        else:
            print("Environment has errors that need to be fixed.")
        
        return all_passed and not self.errors

def main():
    """Main check function"""
    checker = EnvironmentChecker()
    success = checker.run_complete_check()
    
    if success:
        print("\nEnvironment check completed successfully!")
        sys.exit(0)
    else:
        print("\nEnvironment check failed. Please fix the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
