import io
import random
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
import uuid
import os
import modal
from pydantic import BaseModel

DEFAULT_PROMPT = "Turn around and display the clothing"

app = modal.App("wan2-optimized-i2v")

flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)

# Production-ready image following Modal's official flash_attn example
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-dev", "libglib2.0-0")
    .pip_install(
        # Core ML dependencies - exact versions to match flash_attn wheel
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        "numpy>=1.23.5,<2",
        # Computer vision
        "opencv-python>=4.9.0.80",
        "diffusers>=0.31.0",
        "transformers>=4.49.0,<=4.51.3",
        "tokenizers>=0.20.3",
        "accelerate>=1.1.1",
        # Wan2.2 specific dependencies
        "einops",
        "decord",
        "librosa",
        "regex",
        "ftfy",
        "dashscope",
        "peft",
        "loguru",
        "hydra-core",
        "omegaconf",
        "onnxruntime",
        "matplotlib",
        "safetensors",
        "requests",
        "packaging",
        # Utils
        "tqdm",
        "imageio[ffmpeg]",
        "easydict",
        "imageio-ffmpeg",
        "pillow",
        # Web API
        "fastapi[standard]",
        "huggingface-hub[hf_transfer]",
        # S3 and webhook support
        "boto3",
        "pydantic",
        "requests",
    )
    .pip_install(flash_attn_release)
    .run_commands(
        "git clone https://github.com/ModelTC/Wan2.2-Lightning.git /app",
        "cd /app && pip install -e . --no-deps",  # Skip deps since we installed them above
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL_PATH = "/models"
OUTPUT_PATH = "/outputs"
model_volume = modal.Volume.from_name("wan2-models", create_if_missing=True)
output_volume = modal.Volume.from_name("wan2-outputs", create_if_missing=True)


# Pydantic models for API
class VideoGenerationRequest(BaseModel):
    image: str  # URL, base64, or local path
    prompt: str = DEFAULT_PROMPT
    num_frames: int = 37
    seed: Optional[int] = None
    webhook_url: Optional[str] = None
    webhook_token: Optional[str] = None


class VideoGenerationResponse(BaseModel):
    success: bool
    s3_key: Optional[str] = None
    s3_url: Optional[str] = None
    filename: Optional[str] = None
    performance: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None


class VideoGenerationAsyncResponse(BaseModel):
    success: bool
    job_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


with image.imports():
    import torch
    import numpy as np
    from PIL import Image
    import sys

    sys.path.insert(0, "/app")
    import wan
    from wan.configs.wan_i2v_A14B import i2v_A14B


@app.cls(
    image=image,
    gpu="H100",
    timeout=20 * 60,
    volumes={MODEL_PATH: model_volume, OUTPUT_PATH: output_volume},
    min_containers=1,  # Keep 1 container warm
    scaledown_window=60 * 15,  # 15 min idle timeout before scaling down
    enable_memory_snapshot=True,  # Memory snapshot for faster cold starts
    # buffer_containers=1,  # Extra warm container ready for bursts
)
class VideoGenerator:
    @modal.enter()
    def enter(self):
        """Load models when container starts - Modal's recommended pattern."""
        import torch
        from pathlib import Path
        import time

        start_time = time.time()
        print("🚀 Fast loading Wan2.2-Lightning models from volume...")

        # Optimize PyTorch for faster loading
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Model paths
        base_model_path = Path(MODEL_PATH) / "Wan2.2-I2V-A14B"
        lightning_path = Path(MODEL_PATH) / "Wan2.2-Lightning"
        lora_path = lightning_path / "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"

        print(f"Loading base model from: {base_model_path}")
        print(f"Loading Lightning LoRA from: {lora_path}")

        # Verify models exist in volume
        if not base_model_path.exists():
            print(f"❌ Base model not found at {base_model_path}")
            print("  Re-run: modal run create_wan2_volume.py")
            raise FileNotFoundError(f"Base model not found at {base_model_path}")

        if not lora_path.exists():
            print(f"❌ Lightning LoRA not found at {lora_path}")
            print("  Re-run: modal run create_wan2_volume.py")
            raise FileNotFoundError(f"Lightning LoRA not found at {lora_path}")

        print(f"✅ Found base model and Lightning LoRA")

        # Initialize pipeline with Lightning LoRA integration + Fast Loading Optimizations
        print("⚡ Applying fast loading optimizations...")
        print("📋 STARTUP TIMING DEBUG:")

        model_load_start = time.time()
        print(f"🔄 Loading WanI2V pipeline with concurrent optimizations...")

        # Apply concurrent loading optimizations from Modal docs
        from concurrent.futures import ThreadPoolExecutor
        import torch.multiprocessing as mp

        # Set multiprocessing start method for better performance
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set

        self.pipeline = wan.WanI2V(
            config=i2v_A14B,
            checkpoint_dir=str(base_model_path),
            lora_dir=str(lora_path),  # Enable Lightning LoRA for 4-step generation
            device_id=0,
            rank=0,
            t5_cpu=False,  # Keep T5 on GPU for faster loading (we have H100 80GB)
            init_on_cpu=False,  # Initialize directly on GPU
            convert_model_dtype=True,  # Convert to bfloat16 for faster loading
        )

        model_load_time = time.time() - model_load_start
        print(f"✅ WanI2V pipeline loaded in {model_load_time:.1f}s")

        # Warm up the pipeline with optimized kernel compilation
        print("🔥 Warming up pipeline and compiling CUDA kernels...")
        warmup_start = time.time()
        try:
            # Pre-compile CUDA kernels and optimize memory layout
            torch.cuda.empty_cache()

            # Trigger CUDA kernel compilation with a minimal dummy inference
            print("🧪 Running minimal dummy inference to compile kernels...")
            dummy_start = time.time()

            # Create a minimal dummy image for kernel compilation
            import numpy as np
            from PIL import Image

            dummy_img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            dummy_img = self._resize_image(dummy_img)

            # This will compile all the CUDA kernels
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = self.pipeline.generate(
                    input_prompt="test",
                    img=dummy_img,
                    max_area=320 * 576,
                    frame_num=5,  # Minimal frames for kernel compilation
                    shift=5.0,
                    sample_solver="euler",
                    sampling_steps=1,  # Just 1 step for compilation
                    guide_scale=1.0,
                    seed=42,
                    offload_model=False,
                )

            dummy_time = time.time() - dummy_start
            print(f"🧪 Kernel compilation completed in {dummy_time:.1f}s")

            # Clean up after dummy run
            torch.cuda.empty_cache()

            warmup_time = time.time() - warmup_start
            print(f"🔥 Pipeline warmed up in {warmup_time:.1f}s")

        except Exception as e:
            warmup_time = time.time() - warmup_start
            print(f"⚠️ Warmup failed in {warmup_time:.1f}s (non-critical): {e}")

        # Trigger memory snapshot after everything is loaded and warmed up
        print("📸 Triggering memory snapshot for faster future cold starts...")
        snapshot_start = time.time()
        try:
            # Import the snapshot functionality
            from modal import current_app

            # This creates a memory snapshot that will speed up future cold starts
            # The snapshot captures the current memory state including loaded models
            current_app().memory_snapshot()

            snapshot_time = time.time() - snapshot_start
            print(f"📸 Memory snapshot created in {snapshot_time:.1f}s")

        except Exception as e:
            snapshot_time = time.time() - snapshot_start
            print(
                f"⚠️ Memory snapshot failed in {snapshot_time:.1f}s (non-critical): {e}"
            )

        total_time = time.time() - start_time
        print(f"🎯 STARTUP BREAKDOWN:")
        print(
            f"   - Model loading: {model_load_time:.1f}s ({model_load_time/total_time*100:.1f}%)"
        )
        print(f"   - Warmup: {warmup_time:.1f}s ({warmup_time/total_time*100:.1f}%)")
        print(
            f"   - Snapshot: {snapshot_time:.1f}s ({snapshot_time/total_time*100:.1f}%)"
        )
        print(
            f"   - Other: {total_time - model_load_time - warmup_time - snapshot_time:.1f}s"
        )
        print(f"🚀 TOTAL STARTUP TIME: {total_time:.1f}s")
        print("⚡ 4-step Lightning generation enabled (20x speedup)")
        print(
            "🔧 Cold start optimizations: memory snapshots, warm containers, kernel compilation"
        )
        print("📸 Future cold starts should be significantly faster!")

        # Initialize S3 client
        self._init_s3_client()

    def _resize_image(self, image):
        """Resize to ~320p for faster generation while maintaining aspect ratio"""
        import numpy as np

        w, h = image.size
        area = 320 * 576  # Smaller area for faster inference
        aspect = h / w
        new_h = int(np.sqrt(area * aspect))
        new_w = int(np.sqrt(area / aspect))
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        return image.resize((new_w, new_h), Image.LANCZOS)

    def _init_s3_client(self):
        """Initialize S3 client with environment variables"""
        import boto3

        self.s3_bucket = os.environ.get("S3_BUCKET_NAME")
        if not self.s3_bucket:
            print("⚠️ S3_BUCKET_NAME not set. S3 uploads will be disabled.")
            self.s3_client = None
            return

        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )
            print(f"✅ S3 client initialized for bucket: {self.s3_bucket}")
        except Exception as e:
            print(f"❌ Failed to initialize S3 client: {e}")
            self.s3_client = None

    def _upload_to_s3(
        self, filepath: Path, content_type: str = "video/mp4"
    ) -> Dict[str, str]:
        """Upload file to S3 and return S3 key and URL"""
        if not self.s3_client or not self.s3_bucket:
            return {"s3_key": None, "s3_url": None}

        try:
            # Generate unique S3 key
            timestamp = int(time.time())
            s3_key = f"wan2-videos/{timestamp}_{filepath.name}"

            print(f"📤 Uploading to S3: s3://{self.s3_bucket}/{s3_key}")
            upload_start = time.time()

            # Upload with metadata
            self.s3_client.upload_file(
                str(filepath),
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    "ContentType": content_type,
                    "Metadata": {
                        "source": "wan2-lightning",
                        "timestamp": str(timestamp),
                    },
                },
            )

            upload_time = time.time() - upload_start
            print(f"✅ S3 upload completed in {upload_time:.1f}s")

            # Generate public URL (assumes bucket allows public read)
            s3_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"

            return {"s3_key": s3_key, "s3_url": s3_url, "upload_time": upload_time}

        except Exception as e:
            print(f"❌ S3 upload failed: {e}")
            return {"s3_key": None, "s3_url": None, "upload_time": 0}

    def _send_webhook(
        self, webhook_url: str, payload: dict, token: Optional[str] = None
    ):
        """Send webhook with retry logic and exponential backoff"""
        import requests

        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"📡 Sending webhook (attempt {attempt + 1}/{max_retries})")
                webhook_start = time.time()

                response = requests.post(
                    webhook_url, json=payload, headers=headers, timeout=30
                )
                response.raise_for_status()

                webhook_time = time.time() - webhook_start
                print(f"✅ Webhook sent successfully in {webhook_time:.1f}s")
                return True

            except Exception as e:
                print(f"❌ Webhook attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    sleep_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(sleep_time)
                else:
                    print(f"🚨 Failed to send webhook after {max_retries} attempts")
                    return False

    def _run_video_generation(
        self,
        image_bytes: bytes,
        prompt: str = DEFAULT_PROMPT,
        num_frames: int = 37,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Internal method that does the actual video generation work - following CatVTON pattern"""
        import time
        import torch

        seed = seed or random.randint(0, 2**32 - 1)

        # Load image from bytes (Modal best practice)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = self._resize_image(image)

        # Generate video with Lightning 4-step parameters - OPTIMIZED FOR SPEED
        print(
            f"🎬 Generating {num_frames} frames with Lightning 4-step (target: <20s)..."
        )

        # Detailed timing for debugging
        step_start = time.time()

        print(f"📋 Step 1: Starting inference...")
        inference_start = time.time()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            video = self.pipeline.generate(
                input_prompt=prompt,
                img=image,
                max_area=320 * 576,  # Smaller resolution for faster generation
                frame_num=num_frames,
                shift=5.0,  # Lightning config default
                sample_solver="euler",  # Required for Lightning distillation
                sampling_steps=4,  # Lightning 4-step generation
                guide_scale=1.0,  # Lightning uses 1.0, no CFG needed
                seed=seed,
                offload_model=False,  # Keep models on GPU for faster inference
            )

        inference_time = time.time() - inference_start
        print(f"⚡ Step 1 completed: Inference took {inference_time:.1f}s")

        # Save video
        print(f"📋 Step 2: Saving video...")
        save_start = time.time()

        filename = f"{seed}_{hash(prompt) % 10000}.mp4"
        filepath = Path(OUTPUT_PATH) / filename

        from wan.utils.utils import save_video

        save_video(
            tensor=video[None],
            save_file=str(filepath),
            fps=24,
            normalize=True,
            value_range=(-1, 1),
        )

        save_time = time.time() - save_start
        print(f"💾 Step 2 completed: Video save took {save_time:.1f}s")

        print(f"📋 Step 3: Committing to volume...")
        commit_start = time.time()

        output_volume.commit()

        commit_time = time.time() - commit_start
        print(f"☁️ Step 3 completed: Volume commit took {commit_time:.1f}s")

        print(f"📋 Step 4: Cleaning up...")
        cleanup_start = time.time()

        torch.cuda.empty_cache()

        cleanup_time = time.time() - cleanup_start
        print(f"🧹 Step 4 completed: Cleanup took {cleanup_time:.1f}s")

        total_generation_time = time.time() - step_start
        print(f"🎯 TOTAL GENERATION TIME: {total_generation_time:.1f}s")
        print(
            f"   - Inference: {inference_time:.1f}s ({inference_time/total_generation_time*100:.1f}%)"
        )
        print(
            f"   - Video save: {save_time:.1f}s ({save_time/total_generation_time*100:.1f}%)"
        )
        print(
            f"   - Volume commit: {commit_time:.1f}s ({commit_time/total_generation_time*100:.1f}%)"
        )
        print(
            f"   - Cleanup: {cleanup_time:.1f}s ({cleanup_time/total_generation_time*100:.1f}%)"
        )

        # Step 5: Upload to S3
        print(f"📋 Step 5: Uploading to S3...")
        s3_result = self._upload_to_s3(filepath)
        s3_upload_time = s3_result.get("upload_time", 0)

        # Update total time including S3 upload
        total_time_with_s3 = time.time() - step_start
        print(f"☁️ Step 5 completed: S3 upload took {s3_upload_time:.1f}s")
        print(f"🎯 TOTAL TIME WITH S3: {total_time_with_s3:.1f}s")

        # Compile performance metrics
        performance = {
            "inference_time": round(inference_time, 2),
            "video_save_time": round(save_time, 2),
            "volume_commit_time": round(commit_time, 2),
            "cleanup_time": round(cleanup_time, 2),
            "s3_upload_time": round(s3_upload_time, 2),
            "total_generation_time": round(total_generation_time, 2),
            "total_time_with_s3": round(total_time_with_s3, 2),
            "inference_percentage": round(
                (inference_time / total_time_with_s3) * 100, 1
            ),
            "s3_upload_percentage": round(
                (s3_upload_time / total_time_with_s3) * 100, 1
            ),
            "frames_generated": num_frames,
            "frames_per_second_generation": round(num_frames / inference_time, 2),
            "seed_used": seed,
        }

        return {
            "filename": filename,
            "s3_key": s3_result.get("s3_key"),
            "s3_url": s3_result.get("s3_url"),
            "performance": performance,
        }

    @modal.method()
    def inference(
        self,
        image_bytes: bytes,
        prompt: str = DEFAULT_PROMPT,
        num_frames: int = 37,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Modal method wrapper for CLI access - following CatVTON pattern"""
        return self._run_video_generation(
            image_bytes=image_bytes,
            prompt=prompt,
            num_frames=num_frames,
            seed=seed,
        )

    @modal.method()
    def process_video_async(self, request: VideoGenerationRequest, job_id: str):
        """Background processing method that sends results via webhook - following CatVTON pattern"""
        import requests
        import base64
        
        try:
            print(f"🔄 Starting async video generation for job {job_id}")
            
            # Download image from URL (following CatVTON pattern)
            def download_file(url: str) -> bytes:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                return response.content
            
            # Handle image input - URL download like CatVTON
            if request.image.startswith(("http://", "https://")):
                image_bytes = download_file(request.image)
            else:
                # Handle base64 or local file (for backward compatibility)
                if request.image.startswith("data:image"):
                    header, data = request.image.split(",", 1)
                    image_bytes = base64.b64decode(data)
                elif "/" in request.image or "\\" in request.image:
                    image_bytes = Path(request.image).read_bytes()
                else:
                    image_bytes = base64.b64decode(request.image)

            # Run the internal generation method directly (like CatVTON _run_inference)
            result = self._run_video_generation(
                image_bytes=image_bytes,
                prompt=request.prompt,
                num_frames=request.num_frames,
                seed=request.seed,
            )

            # Prepare success webhook payload
            webhook_payload = {
                "job_id": job_id,
                "success": True,
                "filename": result["filename"],
                "s3_key": result["s3_key"],
                "s3_url": result["s3_url"],
                "performance": result["performance"],
                "message": "Video generation completed successfully",
                "completed_at": time.time(),
            }

            print(f"✅ Video generation completed for job {job_id}")

        except Exception as e:
            print(f"❌ Video generation failed for job {job_id}: {str(e)}")
            # Prepare error webhook payload
            webhook_payload = {
                "job_id": job_id,
                "success": False,
                "error": f"Video generation failed: {str(e)}",
                "completed_at": time.time(),
            }

        # Send webhook with the result
        if request.webhook_url:
            self._send_webhook(
                request.webhook_url, webhook_payload, request.webhook_token
            )
        else:
            print(f"⚠️ No webhook URL provided for job {job_id}")

    @modal.fastapi_endpoint(method="POST")
    def api(
        self, request: VideoGenerationRequest
    ) -> Union[VideoGenerationResponse, VideoGenerationAsyncResponse]:
        """
        Hybrid API endpoint for video generation.
        - If webhook_url is provided: Returns job_id immediately and sends results to webhook (async)
        - If webhook_url is None: Returns results directly after processing (sync)
        """
        if request.webhook_url:
            # Async mode - return job ID immediately
            job_id = str(uuid.uuid4())
            print(f"🚀 Starting async job {job_id}")
            self.process_video_async.spawn(request, job_id)

            return VideoGenerationAsyncResponse(
                success=True,
                job_id=job_id,
                message=f"Job {job_id} started. Results will be sent to webhook: {request.webhook_url}",
            )
        else:
            # Sync mode - process and return results immediately
            return self.api_sync(request)

    def api_sync(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        """Internal method for synchronous video generation processing - following CatVTON pattern"""
        import requests
        import base64
        
        try:
            # Download image from URL (following CatVTON pattern)
            def download_file(url: str) -> bytes:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                return response.content
            
            # Handle image input - URL download like CatVTON
            if request.image.startswith(("http://", "https://")):
                image_bytes = download_file(request.image)
            else:
                # Handle base64 or local file (for backward compatibility)
                if request.image.startswith("data:image"):
                    header, data = request.image.split(",", 1)
                    image_bytes = base64.b64decode(data)
                elif "/" in request.image or "\\" in request.image:
                    image_bytes = Path(request.image).read_bytes()
                else:
                    image_bytes = base64.b64decode(request.image)

            # Call the internal generation method directly (like CatVTON _run_inference)
            result = self._run_video_generation(
                image_bytes=image_bytes,
                prompt=request.prompt,
                num_frames=request.num_frames,
                seed=request.seed,
            )

            # Return result using Pydantic response model
            return VideoGenerationResponse(
                success=True,
                filename=result["filename"],
                s3_key=result["s3_key"],
                s3_url=result["s3_url"],
                performance=result["performance"],
                message="Video generation completed successfully",
            )

        except requests.RequestException as e:
            return VideoGenerationResponse(
                success=False, error=f"Failed to download image: {str(e)}"
            )
        except Exception as e:
            return VideoGenerationResponse(
                success=False, error=f"Video generation failed: {str(e)}"
            )


@app.local_entrypoint()
def main(image_path: str, prompt: str, seed: Optional[int] = None):
    """CLI: modal run wan2_modal.py --image-path cat.jpg --prompt "text" """

    # For CLI, load image as bytes (following CatVTON pattern)
    generator = VideoGenerator()

    print(f"🎥 Generating video...")
    start = time.time()

    # Load image as bytes for consistent handling
    if image_path.startswith(("http://", "https://")):
        import requests
        response = requests.get(image_path, timeout=30)
        response.raise_for_status()
        image_bytes = response.content
    else:
        image_bytes = Path(image_path).read_bytes()

    # Call the modal method wrapper for CLI access (like CatVTON)
    result = generator.inference.remote(
        image_bytes=image_bytes,
        prompt=prompt,
        num_frames=37,
        seed=seed
    )

    print(f"✅ Done in {time.time() - start:.1f}s")

    # Extract results
    filename = result["filename"]
    s3_key = result.get("s3_key")
    s3_url = result.get("s3_url")
    performance = result.get("performance", {})

    # Download locally
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    local_path = output_dir / filename
    local_path.write_bytes(b"".join(output_volume.read_file(filename)))

    print(f"💾 Saved locally: {local_path}")

    if s3_key:
        print(f"☁️ S3 Key: {s3_key}")
    if s3_url:
        print(f"🌐 S3 URL: {s3_url}")

    # Print performance summary
    if performance:
        print(f"⚡ Performance Summary:")
        print(f"   - Total time: {performance.get('total_time_with_s3', 0)}s")
        print(
            f"   - Inference: {performance.get('inference_time', 0)}s ({performance.get('inference_percentage', 0)}%)"
        )
        print(
            f"   - S3 upload: {performance.get('s3_upload_time', 0)}s ({performance.get('s3_upload_percentage', 0)}%)"
        )
        print(
            f"   - FPS generation: {performance.get('frames_per_second_generation', 0)} frames/sec"
        )
