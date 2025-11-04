"""UI helper functions for Gradio components"""

import gradio as gr
from typing import Tuple, List, Any


def hide_all_results() -> Tuple:
    """Hide all result display components"""
    return (
        gr.update(visible=False),  # result_display
        gr.update(visible=False),  # html_display
        gr.update(visible=False),  # result_info
        gr.update(visible=False),  # seg_nav_controls
        gr.update(visible=False),  # action_buttons
    )


def hide_viewer_components() -> Tuple:
    """Hide image viewer components"""
    return (
        gr.update(visible=False),  # image_filename
        gr.update(visible=False),  # viewer_image
        gr.update(visible=False),  # nav_controls
    )


def show_viewer_with_image(filename: str, image_path: str, counter: str) -> Tuple:
    """Show viewer with image loaded"""
    return (
        gr.update(value=f"📄 {filename}", visible=True),  # image_filename
        gr.update(value=image_path, visible=True),        # viewer_image
        gr.update(visible=True),                          # nav_controls
        counter,                                          # image_counter
    )


def create_empty_state() -> Tuple:
    """Create empty state for all components"""
    return (
        "",  # dataset_info
        *hide_viewer_components(),
        "",  # image_counter
        gr.update(visible=False),  # action_btn
        *hide_all_results(),
        [],  # current_images
        0,   # current_image_idx
        "",  # current_dataset
        "",  # current_category
        None,  # current_result_image
        "",  # current_result_text
    )


def format_classification_result(result: dict) -> str:
    """Format classification result as text"""
    if not result or not result.get('top_prediction'):
        return "❌ Classification failed"
    
    top_pred = result['top_prediction']
    info_text = (
        f"🎯 CLASSIFICATION RESULT\n\n"
        f"Prediction: {top_pred['class_name']}\n"
        f"Confidence: {top_pred['confidence']*100:.2f}%\n\n"
        f"All Predictions:\n"
    )
    info_text += "\n".join(
        f"  • {pred['class_name']}: {pred['confidence']*100:.2f}%" 
        for pred in result['all_predictions']
    )
    
    if result.get('note'):
        info_text += f"\n\n⚠️ {result['note']}"
    
    return info_text


def format_detection_result(result: dict) -> str:
    """Format detection result as text"""
    if not result or not result.get('annotated_image'):
        return "❌ Detection failed"
    
    total = result['total_detections']
    info_text = f"🎯 DETECTION RESULT\n\nTotal Detections: {total}\n\n"
    
    if result['predictions']:
        info_text += "Detected Objects:\n"
        info_text += "\n".join(
            f"  {i}. {pred['class_name']}: {pred['confidence']*100:.1f}%" 
            for i, pred in enumerate(result['predictions'], 1)
        )
    
    return info_text


def format_segmentation_result(result: dict) -> str:
    """Format segmentation result as text"""
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
    
    return info_text
