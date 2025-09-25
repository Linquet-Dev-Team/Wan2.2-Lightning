import modal
from pathlib import Path as PathlibPath
import os

# Create Modal app for volume creation
app = modal.App("wan2-volume-creator")

# Create the volume for Wan2.2 models
volume = modal.Volume.from_name("wan2-models", create_if_missing=True)
MODEL_DIR = PathlibPath("/models")

# Image with necessary packages for downloading models
volume_image = (
    modal.Image.debian_slim()
    .pip_install([
        "huggingface-hub[hf_transfer]>=0.29.1",
        "torch>=2.4.0",
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

@app.function(
    image=volume_image,
    volumes={MODEL_DIR: volume},
    timeout=3600,  # 1 hour timeout for large downloads
)
def download_wan2_models():
    """Download Wan2.2 base model and Lightning LoRA for 20x faster inference."""
    from huggingface_hub import snapshot_download
    
    print("🔄 Starting Wan2.2 model downloads...")
    print("📋 Downloading: Base model + Lightning LoRA (4-step generation)")
    
    # Create model directories
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Download Wan2.2-I2V-A14B base model (~14GB)
    print("📥 Downloading Wan-AI/Wan2.2-I2V-A14B (base model)...")
    base_model_path = MODEL_DIR / "Wan2.2-I2V-A14B"
    
    if not base_model_path.exists():
        snapshot_download(
            repo_id="Wan-AI/Wan2.2-I2V-A14B",
            local_dir=str(base_model_path),
            local_dir_use_symlinks=False,
        )
        print(f"✅ Base model downloaded to {base_model_path}")
    else:
        print(f"✅ Base model already exists at {base_model_path}")
    
    # Download Wan2.2-Lightning LoRA models
    print("⚡ Downloading lightx2v/Wan2.2-Lightning (LoRA for 4-step generation)...")
    lightning_path = MODEL_DIR / "Wan2.2-Lightning"
    
    if not lightning_path.exists():
        snapshot_download(
            repo_id="lightx2v/Wan2.2-Lightning",
            local_dir=str(lightning_path),
            local_dir_use_symlinks=False,
        )
        print(f"✅ Lightning LoRA downloaded to {lightning_path}")
    else:
        print(f"✅ Lightning LoRA already exists at {lightning_path}")
    
    print("🎉 Wan2.2-Lightning setup complete!")
    print("⚡ 20x faster inference with 4-step generation enabled")
    
    # Commit changes to volume
    volume.commit()
    print("💾 Volume changes committed!")

@app.local_entrypoint()
def main():
    """Download Wan2.2-Lightning models to the volume."""
    print("🚀 Wan2.2-Lightning Model Downloader")
    print("⚡ Setting up 20x faster video generation with 4-step inference")
    download_wan2_models.remote()
    print("✅ Lightning model download completed!")
    print("🔄 Next: Deploy with 'modal deploy wan2_modal.py'")

if __name__ == "__main__":
    main()
