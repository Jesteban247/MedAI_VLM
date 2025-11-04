"""
MedGemma AI Assistant - Unified Launcher
Handles both HF Spaces and Local deployment automatically
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

# Import logger
sys.path.append('src')
from logger import setup_logger
from config import IS_HF_SPACE

logger = setup_logger(__name__)

# ============================================================================
# HF SPACES MODE - Just run app.py directly
# ============================================================================
if IS_HF_SPACE:
    logger.info("="*70)
    logger.info("🌐 HF Spaces Mode Detected")
    logger.info("="*70)
    logger.info("Starting Gradio app with transformers inference...")
    
    # In HF Spaces, just run app.py directly
    # Model server for classification/detection/segmentation still needed
    try:
        # Start model server in background
        logger.info("Starting Model Server (Classification/Detection/Segmentation)...")
        model_server = subprocess.Popen(
            [sys.executable, "src/server_models.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(3)
        
        if model_server.poll() is None:
            logger.info("✓ Model Server started")
        else:
            logger.warning("⚠️  Model Server may have failed to start")
        
        # Run Gradio app (this will block)
        logger.info("Starting Gradio UI...")
        subprocess.run([sys.executable, "app.py"])
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if model_server and model_server.poll() is None:
            model_server.terminate()
            model_server.wait(timeout=5)
    sys.exit(0)

# ============================================================================
# LOCAL MODE - Start all services
# ============================================================================
logger.info("💻 Running in local GGUF mode")

# Global list to track all processes
processes = []

def check_file_exists(path, description):
    """Check if a required file exists"""
    if not Path(path).exists():
        logger.error(f"{description} not found: {path}")
        return False
    return True

def check_requirements():
    """Check if all required files and dependencies exist"""
    logger.info("Checking Requirements...")
    
    all_good = True
    
    # Check Python files
    files_to_check = [
        ("src/server_models.py", "Model server script"),
        ("app.py", "Gradio app script"),
    ]
    
    for file_path, description in files_to_check:
        if check_file_exists(file_path, description):
            logger.info(f"✓ {description} found")
        else:
            all_good = False
    
    # Check GGUF requirements
    all_good = _check_gguf_requirements() and all_good
    
    return all_good

def _check_gguf_requirements():
    """Check GGUF mode requirements"""
    logger.info("Checking local GGUF model requirements")
    
    all_good = True
    model_dirs = [
        ("Models/Medgemma_Base", "Base model directory"),
        ("Models/Medgemma_FT", "Fine-tuned model directory"),
    ]
    
    for dir_path, description in model_dirs:
        if Path(dir_path).exists():
            logger.info(f"✓ {description} found")
        else:
            logger.warning(f"{description} not found: {dir_path}")
            logger.warning("  → Download models from Hugging Face")
    
    # Check llama-server
    try:
        result = subprocess.run(["which", "llama-server"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✓ llama-server found: {result.stdout.strip()}")
        else:
            logger.warning("llama-server not found in PATH")
            logger.warning("  → Install from: https://github.com/ggerganov/llama.cpp")
            all_good = False
    except Exception as e:
        logger.error(f"Could not check for llama-server: {e}")
        all_good = False
    
    return all_good

def cleanup(signum=None, frame=None):
    """Cleanup function to terminate all processes"""
    logger.info("="*70)
    logger.info("🛑 Shutting Down Services")
    logger.info("="*70)
    
    for name, process in processes:
        if process and process.poll() is None:
            logger.info(f"Terminating {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
                logger.info(f"✓ {name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {name}...")
                process.kill()
    
    logger.info("All services stopped. Goodbye!")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def start_service(name, command, wait_time=2, show_output=False):
    """Start a service and add it to the process list"""
    logger.info(f"Starting {name}...")
    logger.debug(f"Command: {' '.join(command)}")
    try:
        if show_output:
            # Show output directly
            process = subprocess.Popen(
                command,
                text=True
            )
        else:
            # Capture output for silent services
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
        
        processes.append((name, process))
        time.sleep(wait_time)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info(f"✓ {name} started successfully")
            return True
        else:
            logger.error(f"{name} failed to start")
            return False
    except Exception as e:
        logger.error(f"{name} error: {e}")
        return False

def main():
    logger.info("="*70)
    logger.info("🏥 MedGemma AI Assistant Launcher")
    logger.info("="*70)
    logger.info("🔧 Running in Local GGUF mode")
    
    # Check requirements
    if not check_requirements():
        logger.error("Some requirements are missing")
        try:
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
        except EOFError:
            logger.error("Cannot prompt for input (non-interactive mode). Exiting.")
            sys.exit(1)
    
    logger.info("Starting Services...")
    
    # Service 1: Model Server (Classification, Detection, Segmentation)
    if not start_service(
        "Model Server",
        [sys.executable, "src/server_models.py"],
        wait_time=3
    ):
        logger.error("Failed to start model server. Exiting.")
        cleanup()
        return
    
    # Service 2: GGUF llama-servers
    logger.info("Starting llama-servers...")
    
    llama_servers = [
        ("Base LLM (8080)", "Models/Medgemma_Base/medgemma-4b-it-Q5_K_M.gguf", 
         "Models/Medgemma_Base/mmproj-F16.gguf", "8080"),
        ("FT LLM (8081)", "Models/Medgemma_FT/brats_medgemma-q5_k_m.gguf",
         "Models/Medgemma_FT/mmproj_model_f16.gguf", "8081")
    ]
    
    for name, model_path, mmproj_path, port in llama_servers:
        if Path(model_path).exists() and Path(mmproj_path).exists():
            start_service(
                name,
                ["llama-server", "--model", model_path, "--mmproj", mmproj_path,
                 "--port", port, "-np", "5", "-c", "16384"],
                wait_time=5
            )
        else:
            logger.warning(f"{name} skipped (model files not found)")
    
    # Service 3: Gradio App
    if not start_service(
        "Gradio UI (7860)",
        [sys.executable, "app.py"],
        wait_time=5,
        show_output=True
    ):
        logger.error("Failed to start Gradio app. Exiting.")
        cleanup()
        return
    
    # Wait for Gradio to print URLs
    time.sleep(5)
    
    logger.info("="*70)
    logger.info("✅ All Services Running")
    logger.info("="*70)
    logger.info("🔧 Model Server: http://localhost:8000")
    logger.info("🤖 Base LLM: http://localhost:8080")
    logger.info("🧠 FT LLM: http://localhost:8081")
    logger.info("🌐 Gradio UI: http://127.0.0.1:7860 (see URLs above)")
    logger.info("="*70)
    logger.info("⏸️  Press Ctrl+C to stop all services")
    logger.info("="*70)
    
    # Keep the script running and monitor processes
    try:
        while True:
            time.sleep(1)
            # Check if any process has died
            for name, process in processes:
                if process.poll() is not None:
                    logger.error(f"{name} died unexpectedly (exit code: {process.returncode})")
                    cleanup()
                    return
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
