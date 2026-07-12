import os
import json
import cv2
import numpy as np
import torch
import timm
import albumentations as A

from albumentations.pytorch import ToTensorV2
from skimage.metrics import structural_similarity as ssim

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_model (1).pth")
CLASS_JSON = os.path.join(BASE_DIR, "class_names.json")
TEMPLATE_DIR = os.path.join(BASE_DIR, "Template")

IMG_SIZE = 160
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESH = 0.5
AREA_THRESH = 200

# =========================================

# ---------- Load Classes ----------
with open(CLASS_JSON, "r") as f:
    CLASSES = json.load(f)

# ---------- Load Model ----------
_model = None

def load_model():
    global _model

    if _model is not None:
        return _model

    model = timm.create_model(
        "tf_efficientnet_b1_ns",
        pretrained=False,
        num_classes=len(CLASSES)
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    _model = model
    return _model

# ---------- Image Transform ----------
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# ---------- ROI Classification ----------
def classify_roi(model, roi_rgb):

    tensor = transform(image=roi_rgb)["image"] \
        .unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    class_probs = {
        CLASSES[i]: float(probs[i])
        for i in range(len(CLASSES))
    }

    best_idx = int(torch.argmax(probs))

    return (
        CLASSES[best_idx],
        float(probs[best_idx]),
        class_probs
    )

# ---------- Load Templates ----------
def load_templates():

    templates = []

    for f in os.listdir(TEMPLATE_DIR):

        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            path = os.path.join(TEMPLATE_DIR, f)

            img = cv2.imread(path)

            if img is not None:
                templates.append((f, img))

    return templates

# ---------- Find Best Template ----------
def find_best_template(test_bgr):

    test_gray = cv2.cvtColor(
        test_bgr,
        cv2.COLOR_BGR2GRAY
    )

    best_score = -1
    best_tpl = None
    best_name = None

    for name, tpl in load_templates():

        tpl_gray = cv2.cvtColor(
            tpl,
            cv2.COLOR_BGR2GRAY
        )

        resized_test = cv2.resize(
            test_gray,
            (tpl.shape[1], tpl.shape[0])
        )

        score = ssim(
            tpl_gray,
            resized_test
        )

        if score > best_score:
            best_score = score
            best_tpl = tpl
            best_name = name

    return best_name, best_tpl

# =========================================================
# ---------- NEW: IMAGE ALIGNMENT ----------
# =========================================================

def align_images(template_bgr, test_bgr):

    """
    Align rotated PCB image with template
    using ORB + Homography
    """

    gray_tpl = cv2.cvtColor(
        template_bgr,
        cv2.COLOR_BGR2GRAY
    )

    gray_test = cv2.cvtColor(
        test_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # ORB detector
    orb = cv2.ORB_create(5000)

    kp1, des1 = orb.detectAndCompute(
        gray_tpl,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray_test,
        None
    )

    # Safety check
    if des1 is None or des2 is None:
        return test_bgr

    # Matcher
    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = matcher.match(des1, des2)

    # Safety check
    if len(matches) < 10:
        return test_bgr

    # Sort matches
    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    matches = matches[:500]

    src_pts = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    # Homography
    H, mask = cv2.findHomography(
        dst_pts,
        src_pts,
        cv2.RANSAC,
        5.0
    )

    # Safety
    if H is None:
        return test_bgr

    # Warp image
    aligned = cv2.warpPerspective(
        test_bgr,
        H,
        (
            template_bgr.shape[1],
            template_bgr.shape[0]
        )
    )

    return aligned

# =========================================================
# ---------- Defect Detection ----------
# =========================================================

def detect_defects(template, test):

    model = load_model()

    # Resize
    test = cv2.resize(
        test,
        (
            template.shape[1],
            template.shape[0]
        )
    )

    # ---------- ALIGNMENT ----------
    aligned_test = align_images(
        template,
        test
    )

    # ---------- Grayscale ----------
    gray_t = cv2.cvtColor(
        template,
        cv2.COLOR_BGR2GRAY
    )

    gray_i = cv2.cvtColor(
        aligned_test,
        cv2.COLOR_BGR2GRAY
    )

    # ---------- Difference ----------
    diff = cv2.absdiff(
        gray_t,
        gray_i
    )

    # ---------- Threshold ----------
    _, mask = cv2.threshold(
        diff,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ---------- Morphology ----------
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    # ---------- Contours ----------
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for c in contours:

        if cv2.contourArea(c) < AREA_THRESH:
            continue

        x, y, w, h = cv2.boundingRect(c)

        roi = aligned_test[
            y:y+h,
            x:x+w
        ]

        # Safety
        if roi.size == 0:
            continue

        roi_rgb = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        )

        label, conf, class_probs = classify_roi(
            model,
            roi_rgb
        )

        if conf < CONF_THRESH:
            continue

        detections.append({

            "x1": int(x),
            "y1": int(y),
            "x2": int(x+w),
            "y2": int(y+h),

            "label": label,
            "conf": round(conf, 4),

            "class_probs": class_probs

        })

    return detections

# =========================================================
# ---------- AUTO MODE ----------
# =========================================================

def infer_auto(test_bgr):

    template_name, template = \
        find_best_template(test_bgr)

    detections = detect_defects(
        template,
        test_bgr
    )

    return template_name, detections

# =========================================================
# ---------- MANUAL MODE ----------
# =========================================================

def infer_manual(test_bgr, template_bgr):

    detections = detect_defects(
        template_bgr,
        test_bgr
    )

    return detections