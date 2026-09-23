import streamlit as st
import cv2
import numpy as np
import fitz  # PyMuPDF
import io
import os
from PIL import Image
from ultralytics import YOLO

# ⚙️ Web Workspace Layout Initializer Configuration
st.set_page_config(page_title="Bhavani Xerox Master App", page_icon="✂️", layout="centered")

st.title("✂️💎 AI MASTER SCANNER & ENHANCER")
st.subheader("Bhavani Xerox")
st.write("Phase 1 runs your custom AI to crop the table out. Phase 2 optimizes text and photo contrast for print-readiness.")

# 🧠 Securely initialize your private Version 5 YOLO-OBB model weights
MODEL_PATH = "best.pt"

@st.cache_resource
def load_custom_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Missing Weights: Place '{MODEL_PATH}' in your repository root folder.")
        return None
    return YOLO(MODEL_PATH)

custom_ai_model = load_custom_model()

# ==============================================================================
# PRO-CLASS PERFORMANCE PIPELINE SETTINGS
# ==============================================================================
TARGET_LONG_SIDE = 2400     # Target resolution for crisp print formatting
MAX_UPSCALE_FACTOR = 2.0    # Soft interpolation limit to prevent blow-up blur
PDF_ZOOM_FACTOR = 2.5       # Resolution scale for PDF extraction loops
OUTPUT_MODE = "grayscale"   

DENOISE_STRENGTH = 15       # Softer denoising to protect fine lines
LIGHT_MAP_SIZE = 110        # Larger = keeps photo tone natural while erasing shadows
SHARPEN_AMOUNT = 1.7        # Main text sharpening weight multiplier
SHARPEN_SIGMA = 0.8         # Focus window radius
CONTRAST_BOOST = 1.0        # Smoothstep S-curve contrast boost factor

def order_points_obb(pts):
    """Consistent 4-point grouping allocation (top-left, top-right, bottom-right, bottom-left)"""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    tl = left_most[np.argmin(left_most[:, 1]), :]
    bl = left_most[np.argmax(left_most[:, 1]), :]
    tr = right_most[np.argmin(right_most[:, 1]), :]
    br = right_most[np.argmax(right_most[:, 1]), :]
    rect[0], rect[1], rect[2], rect[3] = tl, tr, br, bl
    return rect

def crop_only_pipeline(cv_img):
    """Phase 1: Custom AI locates corners and straightens paper perspective layout."""
    orig_cv = cv_img.copy()
    h, w = cv_img.shape[:2]
    if custom_ai_model is None:
        return orig_cv
    results = custom_ai_model(cv_img, verbose=False)
    doc_points = None
    for result in results:
        if result.obb is not None and len(result.obb.xyxyxyxy) > 0:
            doc_points = result.obb.xyxyxyxy.cpu().numpy().reshape(4, 2)
            break
    if doc_points is None:
        h_p, w_p = int(h * 0.04), int(w * 0.04)
        return orig_cv[h_p:h-h_p, w_p:w-w_p]
    rect = order_points_obb(doc_points).reshape(4, 2)
    center = np.mean(rect, axis=0)
    shaved = np.zeros((4, 2), dtype="float32")
    for i in range(4):
        shaved[i] = center + (rect[i] - center) * 0.992
    
    tl, tr, br, bl = shaved[0], shaved[1], shaved[2], shaved[3]
    w_a = np.linalg.norm(br - bl)
    w_b = np.linalg.norm(tr - tl)
    h_a = np.linalg.norm(tr - br)
    h_b = np.linalg.norm(tl - bl)
    
    mw, mh = int(max(w_a, w_b)), int(max(h_a, h_b))
    dst = np.array([[0, 0], [mw-1, 0], [mw-1, mh-1], [0, mh-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(shaved.astype(np.float32), dst)
    return cv2.warpPerspective(orig_cv, M, (mw, mh), flags=cv2.INTER_CUBIC)

def apply_commercial_grade_enhancements(pil_img):
    """Phase 2: Corrects shadows, normalizes contrast, and sharpens ink lines."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=DENOISE_STRENGTH, sigmaSpace=DENOISE_STRENGTH)
    k = int(LIGHT_MAP_SIZE * (min(h, w) / 1200.0))
    k = k if k >= 3 and k % 2 == 1 else (k + 1 if k >= 3 else 3)
    bg = cv2.GaussianBlur(denoised, (k, k), 0)
    bg[bg == 0] = 1
    norm = cv2.divide(denoised, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
    contrasted = clahe.apply(norm)
    blur_sharp = cv2.GaussianBlur(contrasted, (0, 0), sigmaX=SHARPEN_SIGMA * max(min(h, w)/1200.0, 1.0))
    crisp = cv2.addWeighted(contrasted, 1 + SHARPEN_AMOUNT, blur_sharp, -SHARPEN_AMOUNT, 0)
    if CONTRAST_BOOST > 0:
        lut = ((1.0 - CONTRAST_BOOST) * (np.arange(256)/255.0) + CONTRAST_BOOST * ((np.arange(256)/255.0)**2 * (3.0 - 2.0 * (np.arange(256)/255.0)))) * 255.0
        crisp = cv2.LUT(crisp, np.clip(lut, 0, 255).astype(np.uint8))
    
    final_output = cv2.cvtColor(crisp, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))

def smart_upscale(img):
    """Upscales low-res documents cleanly to target printing parameters."""
    w, h = img.size
    ls = max(w, h)
    if ls >= TARGET_LONG_SIDE: return img
    f = min(TARGET_LONG_SIDE / ls, MAX_UPSCALE_FACTOR)
    return img.resize((int(w * f), int(h * f)), Image.Resampling.LANCZOS)

# ==============================================================================
# MAIN SEQUENTIAL WORKFLOW LOOPS
# ==============================================================================
uploaded_files = st.file_uploader(
    "Upload document photos or PDF files here:", 
    type=["png", "jpg", "jpeg", "jfif", "pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    if 'cropped_cache' not in st.session_state:
        st.session_state.cropped_cache = None
        st.session_state.base_name = "document"
        st.session_state.final_pdf_bytes = None

    # STATE 1: Execute and view crop previews
    if st.session_state.cropped_cache is None:
        if st.button("✂️ Step 1: Crop Background Tables", type="primary", use_container_width=True):
            with st.spinner("AI is calculating margins..."):
                cache = []
                name = "scanned_output"
                for idx, file in enumerate(uploaded_files):
                    ext = os.path.splitext(file.name)[1].lower()
                    if idx == 0: name = os.path.splitext(file.name)[0]
                    if ext == ".pdf":
                        doc = fitz.open(stream=file.read(), filetype="pdf")
                        for p_idx in range(len(doc)):
                            pix = doc[p_idx].get_pixmap(matrix=fitz.Matrix(PDF_ZOOM_FACTOR, PDF_ZOOM_FACTOR), colorspace=fitz.csRGB, alpha=False)
                            img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            cache.append(crop_only_pipeline(cv_img))
                        doc.close()
                    else:
                        cv_img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
                        cache.append(crop_only_pipeline(cv_img))
                st.session_state.cropped_cache = cache
                st.session_state.base_name = name
                st.rerun()

    # STATE 2: Show crop outcomes and trigger contrast enhancements
    elif st.session_state.cropped_cache is not None and st.session_state.final_pdf_bytes is None:
        st.subheader("🔍 Review AI Crop Previews")
        for idx, cv_img in enumerate(st.session_state.cropped_cache):
            st.image(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), caption=f"Cropped Page {idx + 1}", use_container_width=True)
        
        st.markdown("---")
        if st.button("💎 Step 2: Run Text Enhancer & Compile", type="primary", use_container_width=True):
            with st.spinner("Bleaching paper backgrounds and sharpening ink structures..."):
                compiler = fitz.open()
                for cv_img in st.session_state.cropped_cache:
                    pil_img = apply_commercial_grade_enhancements(smart_upscale(Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))))
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    buf.seek(0)
                    
                    # FIX: Changed compiler.open to fitz.open to resolve the core AttributeError
                    img_doc = fitz.open("pdf", fitz.open(stream=buf.getvalue(), filetype="png").convert_to_pdf())
                    compiler.insert_pdf(img_doc)
                
                out = io.BytesIO()
                compiler.save(out)
                compiler.close()
                st.session_state.final_pdf_bytes = out.getvalue()
                st.rerun()

    # STATE 3: Expose secure final download link
    else:
        st.balloons()
        st.success("🎉 Processing complete! Tables cropped and text enhanced flawlessly.")
        st.download_button(
            label="⬇️ DOWNLOAD COMPLETED PDF",
            data=st.session_state.final_pdf_bytes,
            file_name=f"bhavani_print_{st.session_state.base_name}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
        if st.button("🔄 Scan Another Batch", use_container_width=True):
            st.session_state.cropped_cache = None
            st.session_state.final_pdf_bytes = None
            st.rerun()
