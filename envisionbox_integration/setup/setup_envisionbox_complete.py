#!/usr/bin/env python3
"""
EnvisionObjectAnnotator Complete Setup Script
Following official guidelines from: https://github.com/DavAhm/EnvisionObjectAnnotator

This script sets up the complete environment for EnvisionObjectAnnotator integration
with your hybrid temporal pipeline.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import json

class EnvisionBoxSetup:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.integration_dir = Path(__file__).parent.parent
        self.setup_dir = Path(__file__).parent
        
    def check_system_requirements(self):
        """Check system requirements for EnvisionObjectAnnotator"""
        print("Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        print(f"   Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            print("   Python 3.8+ required")
            return False
        else:
            print("   Python version OK")
        
        # Check platform
        platform_name = platform.system()
        print(f"   Platform: {platform_name}")
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            print(f"   Available memory: {memory_gb:.1f} GB")
            
            if memory_gb < 8:
                print("   Warning: Less than 8GB RAM available")
            else:
                print("   Memory OK")
        except ImportError:
            print("    psutil not available, cannot check memory")
        
        return True
    
    def setup_conda_environment(self):
        """Set up conda environment for EnvisionObjectAnnotator"""
        print("\nSetting up conda environment...")
        
        env_name = "envisionbox_env"
        
        # Check if environment already exists
        try:
            result = subprocess.run(
                ["conda", "env", "list"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            if env_name in result.stdout:
                print(f"   Environment '{env_name}' already exists")
                return env_name
        except subprocess.CalledProcessError:
            print("   Error checking conda environments")
            return None
        
        # Create new environment
        try:
            print(f"   Creating conda environment: {env_name}")
            subprocess.run([
                "conda", "create", "-n", env_name, 
                "python=3.9", "-y"
            ], check=True)
            
            print(f"   Environment '{env_name}' created successfully")
            return env_name
            
        except subprocess.CalledProcessError as e:
            print(f"   Error creating conda environment: {e}")
            return None
    
    def install_requirements(self, env_name):
        """Install required packages"""
        print(f"\nInstalling requirements in {env_name}...")
        
        requirements = [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "opencv-python>=4.5.0",
            "numpy>=1.21.0",
            "Pillow>=8.3.0",
            "matplotlib>=3.4.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "tqdm>=4.62.0",
            "wandb>=0.12.0",
            "ffmpeg-python>=0.2.0",
            "tkinter",  # For GUI
            "psutil>=5.8.0"
        ]
        
        for req in requirements:
            try:
                print(f"   Installing {req}...")
                subprocess.run([
                    "conda", "run", "-n", env_name, 
                    "pip", "install", req
                ], check=True, capture_output=True)
                print(f"   Correct {req} installed")
            except subprocess.CalledProcessError as e:
                print(f"   False and Warning: Failed to install {req}: {e}")
    
    def setup_sam2_checkpoints(self):
        """Set up SAM2 model checkpoints"""
        print("\nSetting up SAM2 checkpoints...")
        
        checkpoints_dir = self.integration_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        # SAM2 checkpoint URLs (you'll need to download these)
        checkpoints = {
            "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
            "sam2_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
            "sam2_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
        }
        
        print("   Required checkpoints:")
        for name, url in checkpoints.items():
            checkpoint_path = checkpoints_dir / name
            if checkpoint_path.exists():
                print(f"   {name} already exists")
            else:
                print(f"   {name} - Download from: {url}")
                print(f"      Save to: {checkpoint_path}")
        
        return checkpoints_dir
    
    def setup_ffmpeg(self):
        """Set up FFmpeg for video processing"""
        print("\nSetting up FFmpeg...")
        
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                print("   FFmpeg is already installed")
                return True
            else:
                print("   FFmpeg not found")
                return False
                
        except FileNotFoundError:
            print("   FFmpeg not found")
            print("   Please install FFmpeg:")
            print("      Windows: Download from https://ffmpeg.org/download.html")
            print("      Or use: conda install ffmpeg")
            return False
    
    def create_config_files(self):
        """Create configuration files"""
        print("\nCreating configuration files...")
        
        config_dir = self.integration_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        
        # Main configuration
        main_config = {
            "sam2": {
                "model_type": "sam2_hiera_large",
                "checkpoint_path": "checkpoints/sam2_hiera_large.pt",
                "device": "cuda" if self.check_cuda() else "cpu"
            },
            "video_processing": {
                "max_frames": 1000,
                "frame_skip": 1,
                "output_fps": 30
            },
            "hybrid_integration": {
                "enable_text_prompts": True,
                "enable_selective_deidentification": True,
                "confidence_threshold": 0.5
            }
        }
        
        config_path = config_dir / "main_config.json"
        with open(config_path, 'w') as f:
            json.dump(main_config, f, indent=2)
        
        print(f"   Configuration saved to: {config_path}")
        return config_path
    
    def check_cuda(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def create_setup_summary(self):
        """Create setup summary"""
        print("\nCreating setup summary...")
        
        summary = {
            "setup_date": str(Path().cwd()),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.system(),
            "cuda_available": self.check_cuda(),
            "directories_created": [
                "setup/",
                "integration/",
                "test_scenarios/",
                "utils/",
                "configs/",
                "output/",
                "logs/"
            ]
        }
        
        summary_path = self.integration_dir / "setup_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   Setup summary saved to: {summary_path}")
        return summary_path
    
    def run_complete_setup(self):
        """Run the complete setup process"""
        print("Starting EnvisionObjectAnnotator Complete Setup")
        print("=" * 60)
        
        # Step 1: Check system requirements
        if not self.check_system_requirements():
            print("System requirements not met. Please fix issues and try again.")
            return False
        
        # Step 2: Set up conda environment
        env_name = self.setup_conda_environment()
        if not env_name:
            print("Failed to set up conda environment")
            return False
        
        # Step 3: Install requirements
        self.install_requirements(env_name)
        
        # Step 4: Set up SAM2 checkpoints
        self.setup_sam2_checkpoints()
        
        # Step 5: Set up FFmpeg
        self.setup_ffmpeg()
        
        # Step 6: Create configuration files
        self.create_config_files()
        
        # Step 7: Create setup summary
        self.create_setup_summary()
        
        print("\n" + "=" * 60)
        print("EnvisionObjectAnnotator setup completed!")
        print("\nNext steps:")
        print("   1. Download SAM2 checkpoints to checkpoints/ folder")
        print("   2. Install FFmpeg if not already installed")
        print("   3. Run: python integration/hybrid_envisionbox_integration.py")
        print("   4. Test with: python test_scenarios/test_behavioral_scenarios.py")
        
        return True

def main():
    """Main setup function"""
    setup = EnvisionBoxSetup()
    success = setup.run_complete_setup()
    
    if success:
        print("\nSetup completed successfully!")
        sys.exit(0)
    else:
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
