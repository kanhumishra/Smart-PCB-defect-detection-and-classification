import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from inference_backend import infer_auto, infer_manual

st.set_page_config(page_title="PCB Defect Detection System", layout="wide")
st.title("🔍 PCB Defect Detection System")

mode = st.radio("Template Mode", ["Auto", "Manual"])

# ✅ MULTIPLE FILE UPLOAD
pcb_files = st.file_uploader(
    "Upload PCB Images",
    ["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

tpl_file = None
tpl_img = None

if mode == "Manual":
    tpl_file = st.file_uploader("Upload Template Image", ["jpg", "jpeg", "png"])

    # ⚠️ Read template ONCE (important for parallel)
    if tpl_file:
        tpl_img = cv2.imdecode(
            np.frombuffer(tpl_file.read(), np.uint8), 1
        )

# ================= PROCESS FUNCTION =================
def process_image(pcb_file):
    pcb = cv2.imdecode(
        np.frombuffer(pcb_file.read(), np.uint8), 1
    )

    if mode == "Auto":
        template_used, detections = infer_auto(pcb)
    else:
        detections = infer_manual(pcb, tpl_img)
        template_used = "Manual"

    return {
        "name": pcb_file.name,
        "image": pcb,
        "detections": detections,
        "template": template_used
    }

# ================= MAIN =================
if pcb_files:

    if mode == "Manual" and tpl_img is None:
        st.warning("Upload template image")
        st.stop()

    results = []

    # ✅ PARALLEL EXECUTION
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_image, f) for f in pcb_files]

        for f in futures:
            results.append(f.result())

    # ================= DISPLAY =================
    for res in results:

        st.markdown("---")
        st.subheader(f"📸 {res['name']}")

        pcb = res["image"]
        detections = res["detections"]

        st.success(f"Template used: {res['template']}")

        output = pcb.copy()

        # ---------- DRAW ----------
        for d in detections:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]

            cv2.rectangle(output, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                output,
                f"{d['label']} {d['conf']:.2f}",
                (x1, max(15, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        st.image(
            cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        # ---------- RESULTS ----------
        st.subheader(f"Defects Found: {len(detections)}")

        if not detections:
            st.success("🎉 No defects detected")
            continue

        top = max(detections, key=lambda x: x["conf"])

        st.markdown(f"## Predicted Defect: **{top['label']}**")
        st.markdown(f"### Confidence: **{top['conf']*100:.2f}%**")

        # ---------- TABLE ----------
        df = pd.DataFrame(
            top["class_probs"].items(),
            columns=["Defect Type", "Confidence"]
        ).sort_values("Confidence", ascending=False)

        st.subheader("Prediction Confidence per Class")
        st.dataframe(df, use_container_width=True)

        # ---------- BAR CHART ----------
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(df["Defect Type"], df["Confidence"])
        ax.set_ylim(0,1)
        ax.set_ylabel("Confidence")
        ax.set_xlabel("Defect Type")
        ax.set_title("Prediction Confidence per Class")
        plt.xticks(rotation=45)

        st.pyplot(fig)