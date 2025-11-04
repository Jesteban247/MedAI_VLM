"""CSS styles and JavaScript for MedGemma AI Assistant UI"""

CUSTOM_CSS = """
    /* Global dark theme background */
    .gradio-container {
        background: linear-gradient(135deg, #0f0a1a 0%, #1a1625 50%, #1a1a2e 100%) !important;
    }
    
    /* Main section buttons (Classification, Detection, Seg_3D) */
    .main-action-btn {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: 1px solid #a78bfa !important;
        color: #ffffff !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.2) !important;
    }
    
    .main-action-btn:hover {
        background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%) !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Action buttons - Predict/Segment (GREEN) */
    .action-predict-btn {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: 1px solid #34d399 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.2) !important;
        margin-top: 8px !important;
    }
    
    .action-predict-btn:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Loading state for predict button */
    .action-predict-btn:disabled {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%) !important;
        border: 1px solid #9ca3af !important;
        cursor: wait !important;
        animation: pulse 1.5s ease-in-out infinite !important;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Action buttons - Explain (BLUE) */
    .action-explain-btn {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        border: 1px solid #60a5fa !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 10px rgba(59, 130, 246, 0.2) !important;
        margin-top: 8px !important;
    }
    
    .action-explain-btn:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Loading state for explain button */
    .action-explain-btn:disabled {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%) !important;
        border: 1px solid #9ca3af !important;
        cursor: wait !important;
        animation: pulse 1.5s ease-in-out infinite !important;
    }
    
    /* Action buttons - View 3D (YELLOW/ORANGE) */
    .action-view3d-btn {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        border: 1px solid #fbbf24 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 10px rgba(245, 158, 11, 0.2) !important;
        margin-top: 8px !important;
    }
    
    .action-view3d-btn:hover {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%) !important;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Image filename display */
    .image-filename {
        text-align: center !important;
        font-size: 14px !important;
        color: #a0a0a0 !important;
        margin: 8px 0 !important;
        padding: 6px 12px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 6px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* Navigation controls (prev/next buttons and counter) */
    .nav-buttons {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 10px !important;
        margin: 8px 0 !important;
    }
    .nav-buttons button {
        min-width: 40px !important;
        max-width: 40px !important;
        padding: 8px !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }
    .counter-text {
        text-align: center !important;
        margin: 0 auto !important;
        min-width: 150px !important;
        font-weight: 600 !important;
        color: #e0e0e0 !important;
    }
    .counter-text p {
        text-align: center !important;
        margin: 0 !important;
    }
    
    /* Standard image preview (400px for Classification/Detection) */
    .image-preview {
        width: 100% !important;
        min-height: 400px !important;
        max-height: 400px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: #000000 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    /* Gradio image container sizing */
    .image-preview .image-container,
    .image-preview [data-testid="image"],
    .image-preview .gr-image {
        width: 100% !important;
        height: 400px !important;
        min-height: 400px !important;
        max-height: 400px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: #000000 !important;
    }
    
    .image-preview img {
        object-fit: contain !important;
        width: 100% !important;
        height: 400px !important;
        min-height: 400px !important;
        max-height: 400px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    
    .image-preview:hover img {
        opacity: 0.9 !important;
        transform: scale(1.02) !important;
    }
    
    /* Segmentation image preview (700px, larger than standard) */
    #seg3d_image,
    #seg3d_image > *,
    #seg3d_image .image-container,
    #seg3d_image [data-testid="image"],
    #seg3d_image .gr-image,
    #seg3d_image .image-frame {
        width: 100% !important;
        max-width: 100% !important;
        height: 700px !important;
        min-height: 700px !important;
        max-height: 700px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    #seg3d_image img {
        object-fit: contain !important;
        width: 100% !important;
        max-width: 100% !important;
        height: 700px !important;
        min-height: 700px !important;
        max-height: 700px !important;
    }
    
    /* 3D HTML mini preview */
    .html-3d-preview {
        width: 100% !important;
        height: 400px !important;
        min-height: 400px !important;
        max-height: 400px !important;
        display: block !important;
        background: #000000 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    
    .html-3d-preview:hover {
        opacity: 0.95 !important;
        box-shadow: 0 0 0 2px #f59e0b !important;
    }
    
    /* 3D HTML viewer */
    #html-3d-viewer {
        width: 100% !important;
        height: 700px !important;
        min-height: 700px !important;
        border: 1px solid #35384a !important;
        border-radius: 8px !important;
        background: #000000 !important;
        overflow: hidden !important;
    }
    
    #html-3d-viewer iframe {
        width: 100% !important;
        height: 700px !important;
        border: none !important;
    }
    
    /* Image modal overlay */
    .temp-modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0, 0, 0, 0.92);
        backdrop-filter: blur(12px);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.3s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .temp-modal-content {
        background: #16181d;
        border: 1px solid #35384a;
        border-radius: 16px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
        max-width: 90vw;
        max-height: 90vh;
        overflow: hidden;
        animation: slideIn 0.3s ease-out;
        display: flex;
        flex-direction: column;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: scale(0.9) translateY(-20px);
        }
        to {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }
    
    .temp-modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 20px;
        background: #1d1f26;
        border-bottom: 1px solid #2a2d38;
    }
    
    .temp-modal-title {
        font-size: 18px;
        font-weight: 600;
        color: #e8eaed;
        margin: 0;
    }
    
    .temp-modal-close {
        background: #23252e;
        border: 1px solid #35384a;
        font-size: 20px;
        cursor: pointer;
        padding: 8px;
        border-radius: 8px;
        color: #b8bcc8;
        transition: all 0.2s ease;
    }
    
    .temp-modal-close:hover {
        background: #6a6a7a;
        color: white;
        border-color: #6a6a7a;
        transform: scale(1.1);
    }
    
    .temp-modal-body {
        padding: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        width: auto;
        height: auto;
        max-width: 90vw;
        max-height: calc(90vh - 80px);
        overflow: auto;
        flex: 1;
        background: #0f1014;
    }
    
    .temp-modal-image {
        width: auto !important;
        height: auto !important;
        max-width: 100% !important;
        max-height: 100% !important;
        min-width: 800px !important;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        object-fit: contain !important;
    }
    
    /* Segmentation modal (larger, 90vw for multi-slice views) */
    .temp-modal-content-seg3d {
        background: #16181d;
        border: 1px solid #35384a;
        border-radius: 16px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
        max-width: 90vw;
        max-height: 90vh;
        overflow: hidden;
        animation: slideIn 0.3s ease-out;
        display: flex;
        flex-direction: column;
    }
    
    .temp-modal-body-seg3d {
        padding: 20px;
        display: flex;
        justify-content: center;
        align-items: flex-start;
        width: auto;
        height: auto;
        max-width: 90vw;
        max-height: calc(90vh - 80px);
        overflow: auto;
        flex: 1;
        background: #0f1014;
    }
    
    .temp-modal-body-seg3d img {
        width: auto !important;
        height: auto !important;
        max-width: 100% !important;
        max-height: 100% !important;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        object-fit: contain !important;
    }
    
    /* Enhanced Textbox - Make it 3 lines tall visually */
    .gr-multimodal-textbox textarea,
    [data-testid="multimodal-textbox"] textarea,
    .multimodal-textbox textarea {
        min-height: 80px !important;
        height: 80px !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
        padding: 12px !important;
    }
    
    .gr-multimodal-textbox textarea:focus,
    [data-testid="multimodal-textbox"] textarea:focus,
    .multimodal-textbox textarea:focus {
        border-color: #35384a !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Make System Prompt textbox bigger in Additional Inputs */
    .gr-textbox textarea {
        min-height: 150px !important;
        font-size: 14px !important;
        line-height: 1.4 !important;
        padding: 12px !important;
        background: #1a1d23 !important;
        border: 1px solid #35384a !important;
        border-radius: 8px !important;
        color: #e8eaed !important;
    }
    
    .gr-textbox textarea:focus {
        border-color: #35384a !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Chat styling moved to style.css for simplicity */
    
    /* Remove purple focus effects */
    .chatbot-input-container:focus-within,
    [data-testid="textbox"]:focus-within,
    .gr-textbox:focus-within {
        box-shadow: none !important;
        border-color: #35384a !important;
        transform: none !important;
        transition: none !important;
    }
    
    /* Make chat messages more readable */
    .message-wrap {
        font-size: 15px !important;
        line-height: 1.6 !important;
    }
    
    /* Center the chat input area */
    .chat-interface-container,
    [data-testid="chat-interface"],
    .gradio-chatinterface {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    /* Center the multimodal textbox and buttons */
    .multimodal-textbox,
    [data-testid="multimodal-textbox"],
    .gr-multimodal-textbox {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        margin: 0 auto !important;
    }
    
    /* Center the input row container */
    .input-row,
    [class*="input"],
    [class*="textbox-row"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    /* Ensure buttons are centered with the textbox */
    .submit-btn,
    .clear-btn,
    [data-testid="submit-button"],
    [data-testid="clear-button"] {
        margin: 0 4px !important;
    }
    
    /* Fix chat message images to display inline and centered */
    .message-wrap .message-content img,
    .message .message-content img,
    [data-testid="message"] img {
        display: inline-block !important;
        max-width: 300px !important;
        max-height: 300px !important;
        object-fit: contain !important;
        border-radius: 8px !important;
        margin: 8px auto !important;
        vertical-align: middle !important;
    }
    
    /* Center images in chat messages */
    .message-wrap .message-content,
    .message .message-content,
    [data-testid="message"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        text-align: center !important;
    }
    
    /* Keep text left-aligned but images centered */
    .message-wrap .message-content p,
    .message .message-content p {
        text-align: left !important;
        width: 100% !important;
    }
"""

# JavaScript for image modal click functionality
MODAL_JS = """
<script>
function initImageModal() {
    function addImageClickHandlers() {
        document.querySelectorAll('.image-preview img').forEach((img) => {
            img.style.cursor = 'pointer';
            img.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                const isSeg3d = img.closest('[id*="seg3d"]') !== null;
                showImageModal(this.src, isSeg3d);
            };
        });
    }
    
    function showImageModal(imageSrc, isSeg3d = false) {
        const existingModal = document.getElementById('temp-image-modal');
        if (existingModal) existingModal.remove();
        
        const modalClass = isSeg3d ? 'temp-modal-content-seg3d' : 'temp-modal-content';
        const bodyClass = isSeg3d ? 'temp-modal-body-seg3d' : 'temp-modal-body';
        
        const modalHTML = `
            <div id="temp-image-modal" class="temp-modal-overlay">
                <div class="${modalClass}">
                    <div class="temp-modal-header">
                        <span class="temp-modal-title">🖼️ Image Preview</span>
                        <button class="temp-modal-close" onclick="closeImageModal()">✕</button>
                    </div>
                    <div class="${bodyClass}">
                        <img src="${imageSrc}" alt="Preview" class="temp-modal-image" />
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        document.body.style.overflow = 'hidden';
        
        document.getElementById('temp-image-modal').onclick = function(e) {
            if (e.target === this) closeImageModal();
        };
    }
    
    addImageClickHandlers();
    
    // Watch for new images added to DOM
    new MutationObserver(function(mutations) {
        if (mutations.some(m => m.addedNodes.length > 0 && 
            Array.from(m.addedNodes).some(n => n.querySelector?.('.image-preview img')))) {
            setTimeout(addImageClickHandlers, 200);
        }
    }).observe(document.body, { childList: true, subtree: true });
}

window.closeImageModal = function() {
    const modal = document.getElementById('temp-image-modal');
    if (modal) {
        modal.remove();
        document.body.style.overflow = 'auto';
    }
};

window.showHTMLModal = function(htmlContent) {
    console.log('showHTMLModal called with content length:', htmlContent ? htmlContent.length : 0);
    const existingModal = document.getElementById('temp-html-modal');
    if (existingModal) {
        console.log('Removing existing modal');
        existingModal.remove();
    }
    
    const escapedContent = htmlContent.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    
    const modalHTML = `
        <div id="temp-html-modal" class="temp-modal-overlay">
            <div style="background: #16181d; border: 1px solid #35384a; border-radius: 16px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8); width: 85vw; height: 90vh; max-width: 1400px; display: flex; flex-direction: column; animation: slideIn 0.3s ease-out;">
                <div class="temp-modal-header">
                    <span class="temp-modal-title">🌐 3D Brain Tumor Visualization</span>
                    <button class="temp-modal-close" onclick="closeHTMLModal()">✕</button>
                </div>
                <div style="padding: 0; height: calc(100% - 60px); width: 100%; display: flex; align-items: center; justify-content: center; background: #0f1014; overflow: hidden;">
                    <iframe srcdoc="${escapedContent}" 
                            style="width: 100%; height: 100%; border: none; background: white;">
                    </iframe>
                </div>
            </div>
        </div>
    `;
    
    console.log('Inserting modal HTML into body');
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    document.body.style.overflow = 'hidden';
    
    const modal = document.getElementById('temp-html-modal');
    if (modal) {
        console.log('Modal created successfully');
        modal.onclick = function(e) {
            if (e.target === this) closeHTMLModal();
        };
    } else {
        console.error('Failed to create modal element');
    }
};

window.closeHTMLModal = function() {
    const modal = document.getElementById('temp-html-modal');
    if (modal) {
        modal.remove();
        document.body.style.overflow = 'auto';
    }
};

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeImageModal();
        closeHTMLModal();
    }
});

document.readyState === 'loading' 
    ? document.addEventListener('DOMContentLoaded', initImageModal)
    : initImageModal();
setTimeout(initImageModal, 1000);
</script>
"""
