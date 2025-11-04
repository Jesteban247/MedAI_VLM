"""MedGemma AI Assistant - Gradio UI for medical image analysis and chat"""

import os
import warnings

# Import spaces BEFORE any torch/CUDA imports in HF Spaces
IS_HF_SPACE = os.getenv("SPACE_ID") is not None
if IS_HF_SPACE:
    import spaces

# Now safe to import torch-dependent modules
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
warnings.filterwarnings('ignore', message="Can't initialize NVML")

import gradio as gr
import logging
import base64
import json
from PIL import Image

from src.logger import setup_logger
from src.models_client import classify_image, segment_case, detect_objects
from src.config import MAX_CONCURRENT_USERS
from src.styles import CUSTOM_CSS, MODAL_JS
from src.server import respond_stream

# Import server_hf early in HF Spaces to register @spaces.GPU decorator
if IS_HF_SPACE:
    from src import server_hf
from src.image_utils import (
    get_images_from_folder, 
    get_seg3d_folders, 
    base64_to_image,
    save_image_to_temp,
    process_cam_visualizations,
    save_images_to_temp
)
from src.ui_helpers import (
    hide_all_results,
    hide_viewer_components,
    show_viewer_with_image,
    create_empty_state,
    format_classification_result,
    format_detection_result,
    format_segmentation_result
)

# Setup logger
logger = setup_logger(__name__)

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.WARNING)

if IS_HF_SPACE:
    logger.info("🌐 Running in HF Spaces mode (transformers)")
else:
    logger.info("🔧 Running in local GGUF mode (llama-cpp)")

# Load prompts from info.json
with open("info.json", "r") as f:
    INFO_CONFIG = json.load(f)
    SYSTEM_PROMPT = INFO_CONFIG["system_prompt"]


# Helper functions are now imported from src.image_utils and src.ui_helpers


async def run_classification(image_path, model_name):
    """Run classification with multiple CAM visualizations"""
    try:
        result = await classify_image(image_path, model_name)
        
        if not result or not result.get('top_prediction'):
            return [], "❌ Classification failed", ""
        
        # Format result text
        info_text = format_classification_result(result)
        
        # Process CAM visualizations
        viz_images, viz_labels = process_cam_visualizations(result.get('cam_visualizations', {}))
        
        # Create carousel label
        carousel_label = f"CAM Visualizations ({len(viz_images)} methods)" if viz_images else "No visualizations"
        
        return viz_images, info_text, carousel_label
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return [], f"❌ Error: {str(e)}", ""


async def run_detection(image_path, model_name):
    """Run YOLO object detection"""
    try:
        result = await detect_objects(image_path, model_name)
        
        if not result or not result.get('annotated_image'):
            return None, "❌ Detection failed"
        
        viz_img = base64_to_image(result['annotated_image'])
        info_text = format_detection_result(result)
        
        return viz_img, info_text
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return None, f"❌ Error: {str(e)}"


async def run_segmentation(case_folder):
    """Run 3D brain tumor segmentation"""
    try:
        case_path = f"Files_Seg3D/{case_folder}"
        result = await segment_case(case_path, "brats")
        
        if result and result.get('visualization'):
            single_slice = base64_to_image(result['additional_visualizations']['single_slice'])
            three_slices = base64_to_image(result['additional_visualizations']['three_slices'])
            five_slices = base64_to_image(result['additional_visualizations']['five_slices'])
            
            seg_images = [single_slice, three_slices, five_slices]
            html_3d = result.get('3d_html_visualization', '')
            
            # Check if 3D HTML is available
            if not html_3d:
                logger.warning("⚠️ No 3D HTML in result")
                logger.debug(f"Result keys: {result.keys()}")
            
            dice_scores = result['dice_scores']
            avg_dice = result['average_dice']
            vol_analysis = result.get('volumetric_analysis', {})
            spatial = result.get('spatial_analysis', {})
            
            info_text = f"🎯 SEGMENTATION RESULT\n\n"
            info_text += f"📊 Dice Scores:\n  Average: {avg_dice:.4f}\n"
            for k, v in dice_scores.items():
                info_text += f"  • {k}: {v:.4f}\n"
            
            if vol_analysis:
                info_text += f"\n📐 Volumetric Analysis:\n"
                info_text += f"  Whole Tumor: {vol_analysis['whole_tumor_volume_cm3']:.2f} cm³\n"
                info_text += f"  Tumor Core: {vol_analysis['tumor_core_volume_cm3']:.2f} cm³\n"
                info_text += f"  Enhancing Tumor: {vol_analysis['enhancing_tumor_volume_cm3']:.2f} cm³\n"
            
            if spatial and spatial.get('center_of_mass'):
                com = spatial['center_of_mass']
                info_text += f"\n📍 Tumor Location:\n"
                info_text += f"  Sagittal: {com['sagittal']:.1f}, Coronal: {com['coronal']:.1f}, Axial: {com['axial']:.1f}\n"
            
            # Start with single slice view (index 0)
            return seg_images, 0, single_slice, "View 1 of 3 (1 Slice)", info_text, html_3d
        else:
            return [], 0, None, "No images", "❌ Segmentation failed", ""
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        return [], 0, None, "No images", f"❌ Error: {str(e)}", ""


async def respond_with_context_control(message, history, system_message, max_tokens, temperature, top_p, model_choice):
    """Wrapper for VLM streaming with session management"""
    from src.config import IS_HF_SPACE
    
    # Extract session ID
    try:
        request: gr.Request = gr.context.LocalContext.request.get()
        session_hash = request.session_hash if request and hasattr(request, 'session_hash') else None
        session_id = session_hash[:8] if session_hash and len(session_hash) > 8 else "default"
    except:
        session_id = "default"
    
    # Clear session on new conversation (only for local mode with session management)
    if not IS_HF_SPACE and (not history or len(history) == 0):
        logger.info(f"🔄 NEW CONVERSATION | Session: {session_id}")
        from src.session_manager import session_manager
        session_manager.clear_session(session_id)
    
    # Log model selection
    model_type = "FT" if model_choice == "Fine-Tuned (BraTS)" else "Base"
    logger.info(f"🤖 MODEL | Session: {session_id} | Type: {model_type}")
    
    if IS_HF_SPACE:
        # HF Spaces: Use transformers inference
        async for response in respond_stream(message, history, system_message, max_tokens, temperature, top_p, session_id, model_choice=model_choice):
            yield response
    else:
        # Local: Use GGUF llama-cpp servers
        from src.config import FT_SERVER_URL, BASE_SERVER_URL
        server_url = FT_SERVER_URL if model_choice == "Fine-Tuned (BraTS)" else BASE_SERVER_URL
        async for response in respond_stream(message, history, system_message, max_tokens, temperature, top_p, session_id, server_url, model_choice):
            yield response


# Helper functions
def get_explain_message(category, model_name, view_type=None):
    """Get the appropriate explanation prompt from info.json"""
    if category == "Classification":
        if model_name == "Brain_Tumor":
            return INFO_CONFIG["classification_brain_tumor_message"]
        elif model_name == "Chest_X-Ray":
            return INFO_CONFIG["classification_chest_xray_message"]
        elif model_name == "Lung_Cancer":
            return INFO_CONFIG["classification_lung_histopathology_message"]
    elif category == "Detection":
        if model_name == "Blood_Cell":
            return INFO_CONFIG["detection_blood_message"]
        elif model_name == "Breast_Cancer":
            return INFO_CONFIG["detection_breast_cancer_message"]
        elif model_name == "Fracture":
            return INFO_CONFIG["detection_fracture_message"]
    elif category == "Segmentation":
        if view_type == "single":
            return INFO_CONFIG["segmentation_brats_single_message"]
        elif view_type == "three":
            return INFO_CONFIG["segmentation_brats_three_message"]
        elif view_type == "five":
            return INFO_CONFIG["segmentation_brats_five_message"]
    return "Analyze this medical image and provide a detailed report."


# Create chat interface - clean and simple
chat_interface = gr.ChatInterface(
    respond_with_context_control,
    type="messages",
    multimodal=True,
    chatbot=gr.Chatbot(
        type="messages",
        scale=4,
        height=600,
        show_copy_button=True,
    ),
    textbox=gr.MultimodalTextbox(
        file_types=["image"],
        file_count="multiple",
        placeholder="💬 Type your medical question or upload images for analysis...",
        show_label=False,
    ),
    additional_inputs=[
        gr.Textbox(
            value=SYSTEM_PROMPT, 
            label="System Prompt",
            lines=6,
            max_lines=10,
            info="Customize the AI assistant's behavior and medical expertise"
        ),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max tokens"),
        gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.95, step=0.05, label="Top-p"),
        gr.Radio(
            choices=["Base (General)", "Fine-Tuned (BraTS)"],
            value="Base (General)",
            label="Model Selection",
        ),
    ],
    stop_btn=True,
    cache_examples=False,
)


# Build full interface with Tabs
with gr.Blocks(title="MedGemma AI", css=CUSTOM_CSS, head=MODAL_JS, fill_height=True) as demo:
    gr.Markdown("# 🏥 MedGemma AI Assistant")
    
    with gr.Tabs() as tabs:
        # Tab 1: Chat Interface
        with gr.Tab("💬 Chat", id=0):
            chat_interface.render()
            chat_textbox = chat_interface.textbox
            chat_chatbot = chat_interface.chatbot
        
        # Tab 2: Image Analysis (Left: Selection + Images, Right: Predictions)
        with gr.Tab("🔬 Image Analysis", id=1):
            with gr.Row():
                # Left Column: Model Selection + Image Viewer
                with gr.Column(scale=1):
                    gr.Markdown("### 🤖 Medical AI Models")
                    
                    # Classification section
                    with gr.Accordion("🧠 Classification", open=True):
                        brain_tumor_btn = gr.Button("🔬 Brain Tumor", size="sm")
                        chest_xray_btn = gr.Button("🫁 Chest X-Ray", size="sm")
                        lung_cancer_btn = gr.Button("💨 Lung Cancer", size="sm")
                    
                    # Detection section
                    with gr.Accordion("🔍 Detection", open=False):
                        blood_cell_btn = gr.Button("🩸 Blood Cell", size="sm")
                        breast_cancer_btn = gr.Button("🎗️ Breast Cancer", size="sm")
                        fracture_btn = gr.Button("🦴 Fracture", size="sm")
                    
                    # Segmentation section
                    with gr.Accordion("📊 Segmentation", open=False):
                        folders = get_seg3d_folders()
                        seg3d_dropdown = gr.Dropdown(
                            choices=folders,
                            value=None,  # Don't auto-select, let user choose
                            label="Select Case",
                            interactive=True,
                        )
                    
                    gr.Markdown("---")
                    
                    # Image Viewer
                    gr.Markdown("### 🖼️ Image Viewer")
                    dataset_info = gr.Markdown("", visible=False)  # Hidden, not used anymore
                    image_filename = gr.Markdown("", visible=False, elem_classes="image-filename")
                    viewer_image = gr.Image(label="", show_label=False, height=400, visible=False, elem_classes="image-preview")
                    
                    with gr.Row(visible=False, elem_classes="nav-buttons") as nav_controls:
                        prev_btn = gr.Button("◀", size="sm")
                        image_counter = gr.Markdown("", elem_classes="counter-text")
                        next_btn = gr.Button("▶", size="sm")
                    
                    action_btn = gr.Button("🔬 Analyze", size="lg", visible=False, elem_classes="action-predict-btn")
                
                # Right Column: Predictions/Results
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Analysis Results")
                    result_display = gr.Image(label="", show_label=False, height=400, visible=False, elem_classes="image-preview")
                    html_display = gr.HTML(visible=False, elem_classes="html-3d-preview")
                    result_info = gr.Textbox(label="", show_label=False, lines=12, visible=False)
                    
                    # CAM/Segmentation navigation (for multiple views)
                    with gr.Row(visible=False, elem_classes="nav-buttons") as seg_nav_controls:
                        seg_prev_btn = gr.Button("◀", size="sm")
                        seg_counter = gr.Markdown("", elem_classes="counter-text")
                        seg_next_btn = gr.Button("▶", size="sm")
                    
                    with gr.Column(visible=False) as action_buttons:
                        explain_btn = gr.Button("🤖 Explain with AI", size="lg", elem_classes="action-explain-btn")
                        view_3d_btn = gr.Button("🌐 View 3D", size="lg", visible=False, elem_classes="action-view3d-btn")
                    
                    # Hidden textbox to pass HTML to JavaScript
                    html_storage_bridge = gr.Textbox(visible=False, elem_id="html-storage-bridge")
    
    # State variables
    current_images = gr.State([])
    current_image_idx = gr.State(0)
    current_dataset = gr.State("")
    current_category = gr.State("")
    current_result_image = gr.State(None)
    current_result_text = gr.State("")
    seg3d_images = gr.State([])
    seg3d_image_idx = gr.State(0)
    seg3d_html_storage = gr.State("")
    
    # Helper functions
    def show_images(folder_name, category):
        """Load and display images automatically - resets to first image"""
        images = get_images_from_folder(f"Images/{category}/{folder_name}")
        action_text = "🔬 Classify" if category == "Classification" else "🔍 Detect"
        
        if images:
            # Get filename from path
            filename = os.path.basename(images[0])
            return (
                "",  # Remove dataset info text
                gr.update(value=f"📄 {filename}", visible=True),
                gr.update(value=images[0], visible=True),
                gr.update(visible=True),
                f"Image 1 of {len(images)}",
                gr.update(value=action_text, visible=True),
                gr.update(visible=False),  # Hide result display
                gr.update(visible=False),  # Hide HTML display
                gr.update(visible=False),  # Hide result info
                gr.update(visible=False),  # Hide seg nav controls
                gr.update(visible=False),  # Hide action buttons
                images, 0, folder_name, category, None, ""
            )
        return (
            f"**{folder_name}** - No images found",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            [], 0, folder_name, category, None, ""
        )
    
    def show_seg3d_case(case_name):
        """Load segmentation case info - auto shows segment button, hides image viewer"""
        if not case_name:
            return (
                "",
                gr.update(visible=False),  # Hide image filename
                gr.update(visible=False),  # Hide viewer image
                gr.update(visible=False),  # Hide nav controls
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                [], 0, case_name, "Segmentation", None, ""
            )
        
        folder_path = f"Files_Seg3D/{case_name}"
        if os.path.exists(folder_path):
            return (
                "",  # Remove dataset info text
                gr.update(visible=False),  # Hide image filename - not needed for segmentation
                gr.update(visible=False),  # Hide viewer image - not needed for segmentation
                gr.update(visible=False),  # Hide nav controls - not needed for segmentation
                "",
                gr.update(value="📊 Segment", visible=True),  # Show segment button
                gr.update(visible=False),  # Hide result display
                gr.update(visible=False),  # Hide HTML display
                gr.update(visible=False),  # Hide result info
                gr.update(visible=False),  # Hide seg nav controls
                gr.update(visible=False),  # Hide action buttons
                [], 0, case_name, "Segmentation", None, ""
            )
        return (
            f"**{case_name}** - Not found",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            [], 0, "", "", None, ""
        )
    
    def navigate_image(images, current_idx, direction):
        """Navigate and hide previous results"""
        if not images:
            return current_idx, gr.update(), gr.update(), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        new_idx = (current_idx + direction) % len(images)
        filename = os.path.basename(images[new_idx])
        return (new_idx, gr.update(value=f"📄 {filename}"), gr.update(value=images[new_idx]), 
                f"Image {new_idx + 1} of {len(images)}",
                gr.update(visible=False),  # Hide result display
                gr.update(visible=False),  # Hide HTML display
                gr.update(visible=False),  # Hide result info
                gr.update(visible=False),  # Hide action buttons
                gr.update(visible=False))  # Hide seg nav controls
    
    async def handle_action(images, seg_images_state, idx, seg_idx, dataset_name, category):
        """Handle analyze button"""
        if category in ["Classification", "Detection"] and images:
            if idx >= len(images):
                return (gr.update(), "No image", gr.update(visible=False), gr.update(visible=False), 
                        gr.update(visible=False), gr.update(visible=False), "", None, "", [], 0, "")
            
            image_path = images[idx]
            if category == "Classification":
                viz_images, text, carousel_label = await run_classification(image_path, dataset_name)
            else:
                img, text = await run_detection(image_path, dataset_name)
                viz_images = [img] if img else []
                carousel_label = ""
            
            result_path = None
            result_images = []
            
            if viz_images:
                # Save first image for "Explain with AI"
                result_path = save_image_to_temp(viz_images[0])
                
                # Save all images for carousel
                result_images = save_images_to_temp(viz_images)
            
            # Show navigation if multiple visualizations (Classification CAMs)
            show_nav = len(viz_images) > 1
            counter_text = f"View 1 of {len(viz_images)} ({carousel_label})" if show_nav else ""
            
            return (gr.update(value=viz_images[0] if viz_images else None, visible=bool(viz_images)), 
                    gr.update(visible=False),  # html_display - not used for classification
                    gr.update(value=text, visible=True),
                    gr.update(visible=show_nav),  # seg_nav_controls (reused for CAM navigation)
                    gr.update(value=counter_text),  # seg_counter (reused for CAM counter)
                    gr.update(visible=True),  # action_buttons
                    gr.update(visible=False),  # view_3d_btn - hide for classification
                    result_path, text, result_images, 0, "")  # result_path, text, seg images, idx, html
        
        elif category == "Segmentation" and dataset_name:
            seg_imgs, new_idx, img, counter, text, html = await run_segmentation(dataset_name)
            
            result_path = None
            if img:
                result_path = save_image_to_temp(img)
            
            counter_text = f"View 1 of 3 (1 Slice)"
            
            # Decode HTML for inline display
            html_preview = ""
            if html:
                try:
                    decoded_html = base64.b64decode(html).decode('utf-8')
                    # Create iframe wrapper for inline display
                    html_preview = f'''
                    <div style="width: 100%; height: 700px; background: #000;">
                        <iframe srcdoc="{decoded_html.replace('"', '&quot;')}" 
                                style="width: 100%; height: 100%; border: none; background: white;">
                        </iframe>
                    </div>
                    '''
                except Exception as e:
                    logger.error(f"❌ Failed to decode HTML: {e}")
            
            return (gr.update(value=img, visible=True),
                    gr.update(value=html_preview, visible=False),  # HTML hidden initially
                    gr.update(value=text, visible=True),
                    gr.update(visible=True),  # seg_nav_controls
                    gr.update(value=counter_text),  # seg_counter
                    gr.update(visible=True),  # action_buttons
                    gr.update(visible=True),  # view_3d_btn - show for segmentation
                    result_path, text, seg_imgs, new_idx, html)  # result_path, text, seg images, idx, html
        
        return (gr.update(), "Please select a dataset", gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False), "", None, "", [], 0, "")
    
    def navigate_seg_image(images, current_idx, direction, category):
        """Navigate through segmentation views or CAM visualizations"""
        if not images:
            return current_idx, gr.update(), ""
        new_idx = (current_idx + direction) % len(images)
        
        # Load image (handle both PIL Image and file paths)
        img_value = images[new_idx]
        if isinstance(img_value, str):
            img_value = Image.open(img_value)
        
        # Different labels for segmentation vs classification
        if category == "Segmentation":
            view_names = ["1 Slice", "3 Slices", "5 Slices"]
            heights = [500, 600, 700]
            height = heights[new_idx] if new_idx < len(heights) else 700
            counter_text = f"View {new_idx + 1} of 3 ({view_names[new_idx]})"
        else:
            # Classification CAM visualizations
            height = 400
            counter_text = f"View {new_idx + 1} of {len(images)}"
        
        return new_idx, gr.update(value=img_value, height=height), counter_text
    
    def show_3d_view(html_storage):
        """Show 3D HTML visualization in popup modal"""
        if not html_storage:
            gr.Warning("No 3D visualization available")
            return gr.update(), ""
        try:
            html_content = base64.b64decode(html_storage).decode('utf-8')
            # Escape for HTML attribute
            escaped_html = html_content.replace('"', '&quot;').replace("'", '&#39;')
            # Create an iframe with the HTML content
            iframe_html = f'''
            <div style="width: 100%; height: 700px; background: #000;">
                <iframe srcdoc="{escaped_html}" 
                        style="width: 100%; height: 100%; border: none; background: white;">
                </iframe>
            </div>
            '''
            gr.Info("3D visualization loaded! Click to expand.")
            return gr.update(), iframe_html
        except Exception as e:
            logger.error(f"Error decoding 3D HTML: {e}")
            gr.Warning("Failed to load 3D visualization")
            return gr.update(), ""
    
    def explain_with_ai(seg_images, seg_idx, result_text, model_name, category):
        """Send CURRENT view to chat - uses the actual image being displayed"""
        # Get the current image being viewed
        if not seg_images or seg_idx >= len(seg_images):
            gr.Warning("No result image available")
            return {"text": "", "files": []}, gr.update(selected=0)
        
        # Get the current image path (it's already saved as a file path)
        current_image_path = seg_images[seg_idx]
        
        # If it's a PIL Image, save it
        if not isinstance(current_image_path, str):
            current_image_path = save_image_to_temp(current_image_path)
        
        if not os.path.exists(current_image_path):
            gr.Warning("Image file not found")
            return {"text": "", "files": []}, gr.update(selected=0)
        
        message = get_explain_message(category, model_name)
        full_message = f"Here are the prediction results:\n\n{result_text}\n\n---\n\n{message}" if result_text else message
        gr.Info("📝 Switching to chat with analysis loaded!")
        return {"text": full_message, "files": [current_image_path]}, gr.update(selected=0)
    
    # Event handlers
    outputs_list = [dataset_info, image_filename, viewer_image, nav_controls, image_counter, action_btn,
                   result_display, html_display, result_info, seg_nav_controls, action_buttons,
                   current_images, current_image_idx, current_dataset, current_category,
                   current_result_image, current_result_text]
    
    brain_tumor_btn.click(lambda: show_images("Brain_Tumor", "Classification"), outputs=outputs_list)
    chest_xray_btn.click(lambda: show_images("Chest_X-Ray", "Classification"), outputs=outputs_list)
    lung_cancer_btn.click(lambda: show_images("Lung_Cancer", "Classification"), outputs=outputs_list)
    
    blood_cell_btn.click(lambda: show_images("Blood_Cell", "Detection"), outputs=outputs_list)
    breast_cancer_btn.click(lambda: show_images("Breast_Cancer", "Detection"), outputs=outputs_list)
    fracture_btn.click(lambda: show_images("Fracture", "Detection"), outputs=outputs_list)
    
    seg3d_dropdown.change(show_seg3d_case, inputs=[seg3d_dropdown], outputs=outputs_list)
    
    prev_btn.click(lambda imgs, idx: navigate_image(imgs, idx, -1),
                  inputs=[current_images, current_image_idx],
                  outputs=[current_image_idx, image_filename, viewer_image, image_counter, result_display, html_display, result_info, action_buttons, seg_nav_controls])
    
    next_btn.click(lambda imgs, idx: navigate_image(imgs, idx, 1),
                  inputs=[current_images, current_image_idx],
                  outputs=[current_image_idx, image_filename, viewer_image, image_counter, result_display, html_display, result_info, action_buttons, seg_nav_controls])
    
    action_btn.click(
        fn=lambda cat: gr.update(value=f"⏳ {'Classifying' if cat == 'Classification' else 'Detecting' if cat == 'Detection' else 'Segmenting'}...", interactive=False),
        inputs=[current_category],
        outputs=[action_btn]
    ).then(
        handle_action,
        inputs=[current_images, seg3d_images, current_image_idx, seg3d_image_idx, current_dataset, current_category],
        outputs=[result_display, html_display, result_info, seg_nav_controls, seg_counter, action_buttons, view_3d_btn,
                current_result_image, current_result_text, seg3d_images, seg3d_image_idx, seg3d_html_storage]
    ).then(
        fn=lambda cat: gr.update(value=f"🔬 {'Classify' if cat == 'Classification' else 'Detect' if cat == 'Detection' else 'Segment'}", interactive=True),
        inputs=[current_category],
        outputs=[action_btn]
    )
    
    # CAM/Segmentation navigation (reused for both)
    seg_prev_btn.click(
        lambda imgs, idx, cat: navigate_seg_image(imgs, idx, -1, cat),
        inputs=[seg3d_images, seg3d_image_idx, current_category],
        outputs=[seg3d_image_idx, result_display, seg_counter]
    )
    
    seg_next_btn.click(
        lambda imgs, idx, cat: navigate_seg_image(imgs, idx, 1, cat),
        inputs=[seg3d_images, seg3d_image_idx, current_category],
        outputs=[seg3d_image_idx, result_display, seg_counter]
    )
    
    # First update the bridge, then trigger the modal
    view_3d_btn.click(
        fn=lambda html: html,
        inputs=[seg3d_html_storage],
        outputs=[html_storage_bridge]
    ).then(
        fn=None,
        inputs=[html_storage_bridge],
        outputs=None,
        js="""
        (html_storage) => {
            console.log('View 3D button clicked, html_storage length:', html_storage ? html_storage.length : 0);
            if (html_storage && html_storage.length > 0) {
                try {
                    const decoded = atob(html_storage);
                    console.log('Decoded HTML length:', decoded.length);
                    if (typeof window.showHTMLModal === 'function') {
                        console.log('Calling showHTMLModal...');
                        window.showHTMLModal(decoded);
                    } else {
                        console.error('showHTMLModal function not found on window object');
                        alert('Modal function not available. Please refresh the page.');
                    }
                } catch (e) {
                    console.error('Failed to decode or show HTML:', e);
                    alert('Failed to show 3D visualization: ' + e.message);
                }
            } else {
                console.error('No HTML storage provided');
                alert('No 3D visualization available');
            }
        }
        """
    )
    
    explain_btn.click(explain_with_ai,
                     inputs=[seg3d_images, seg3d_image_idx, current_result_text, current_dataset, current_category],
                     outputs=[chat_textbox, tabs])


if __name__ == "__main__":
    logger.info("Starting MedGemma AI Assistant")
    if IS_HF_SPACE:
        logger.info("Mode: HF Spaces (transformers)")
        demo.queue(default_concurrency_limit=MAX_CONCURRENT_USERS)
        demo.launch()  # HF Spaces handles server config
    else:
        logger.info("Mode: Local GGUF (llama-cpp)")
        demo.queue(default_concurrency_limit=MAX_CONCURRENT_USERS)
        demo.launch(share=True, server_name="127.0.0.1", server_port=7860)