import os
import sys
import cv2
import torch
import numpy as np
import tensorflow as tf

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QInputDialog, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtCore import Qt

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input


# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DET_THRESH = 0.80
IMG_SIZE = 260
MC_RUNS = 30
# ==========================================


# ================= PREPROCESSING =================
def segment_breast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return img

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    segmented = cv2.bitwise_and(img, img, mask=mask)
    return segmented


def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def cv2_to_qpixmap(img):
    h, w, ch = img.shape
    bytes_per_line = ch * w
    qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg)


# ================= MAIN CLASS =================
class MammographyAI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mammography AI Hybrid CAD System")
        self.resize(1700, 950)

        self.original_image = None
        self.image = None
        self.final_box = None
        self.detector_score = None
        self.age = None
        self.birads = None
        self.density = None

        self.load_models()
        self.init_ui()

    # ================= LOAD MODELS =================
    def load_models(self):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        det_path = os.path.join(BASE_DIR, "fasterrcnn_pseudo_best.pth")

        self.detector = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

        ckpt = torch.load(det_path, map_location=DEVICE)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.detector.load_state_dict(state_dict)
        self.detector.to(DEVICE).eval()

        hybrid_path = os.path.join(BASE_DIR, "Hybrid CNN", "best_hybrid_model_overall.keras")
        self.hybrid_model = keras.models.load_model(hybrid_path, compile=False)

    # ================= UI =================
    def init_ui(self):

        self.setStyleSheet("""
            QMainWindow { background-color: black; }
            QLabel { color: white; }
            QPushButton {
                background-color: #222;
                color: white;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #444; }
        """)

        central = QWidget()
        main_layout = QVBoxLayout(central)

        title = QLabel("Mammography AI Hybrid CAD System")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        self.status_label = QLabel("Load image to begin")
        main_layout.addWidget(self.status_label)

        self.meta_label = QLabel("")
        main_layout.addWidget(self.meta_label)

        self.birads_label = QLabel("")
        self.birads_label.setFont(QFont("Arial", 13, QFont.Bold))
        main_layout.addWidget(self.birads_label)

        self.explain_label = QLabel("")
        main_layout.addWidget(self.explain_label)

        # Image Panels
        grid = QGridLayout()

        self.panel_original = QLabel()
        self.panel_gradcam = QLabel()
        self.panel_enhanced = QLabel()
        self.panel_edge = QLabel()

        panels = [
            self.panel_original,
            self.panel_gradcam,
            self.panel_enhanced,
            self.panel_edge
        ]

        for p in panels:
            p.setAlignment(Qt.AlignCenter)
            p.setMinimumSize(400, 350)
            p.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        grid.addWidget(QLabel("Original + Detection"), 0, 0)
        grid.addWidget(QLabel("Grad-CAM"), 0, 1)
        grid.addWidget(self.panel_original, 1, 0)
        grid.addWidget(self.panel_gradcam, 1, 1)

        grid.addWidget(QLabel("CLAHE Enhanced"), 2, 0)
        grid.addWidget(QLabel("Edge Map"), 2, 1)
        grid.addWidget(self.panel_enhanced, 3, 0)
        grid.addWidget(self.panel_edge, 3, 1)

        main_layout.addLayout(grid)

        btn_layout = QHBoxLayout()

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)

        infer_btn = QPushButton("Run Hybrid Analysis")
        infer_btn.clicked.connect(self.run_inference)

        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(infer_btn)

        main_layout.addLayout(btn_layout)

        self.setCentralWidget(central)

    # ================= LOAD IMAGE =================
    def load_image(self):

        path, _ = QFileDialog.getOpenFileName(self, "Select Mammogram")
        if not path:
            return

        age, ok = QInputDialog.getInt(self, "Patient Age", "Enter age:", 50, 10, 100)
        if not ok:
            return

        birads, ok = QInputDialog.getInt(self, "BIRADS", "Enter BIRADS (1-6):", 3, 1, 6)
        if not ok:
            return

        density, ok = QInputDialog.getInt(self, "Density", "Enter Density (1-4):", 2, 1, 4)
        if not ok:
            return

        self.age = age
        self.birads = birads
        self.density = density

        self.meta_label.setText(f"Age: {age} | BIRADS: {birads} | Density: {density}")

        self.original_image = cv2.imread(path)
        self.display_image(self.original_image, self.panel_original)

    # ================= DISPLAY =================
    def display_image(self, img, panel):

        pix = cv2_to_qpixmap(img)
        pix = pix.scaled(
            panel.width(),
            panel.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        panel.setPixmap(pix)

    # ================= HYBRID PREDICTION =================
    def hybrid_prediction(self, image):

        clinical = np.array([[self.age, self.birads, self.density]], dtype="float32")

        resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(resized)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = []

        for _ in range(MC_RUNS):
            p = self.hybrid_model([img_array, clinical], training=True)

            if isinstance(p, list):
                p = p[0]

            preds.append(p.numpy()[0][0])

        return float(np.mean(preds)), float(np.std(preds))

    # ================= GRAD-CAM =================
    def generate_gradcam(self, roi):

        resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(resized)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        clinical = np.array([[self.age, self.birads, self.density]], dtype="float32")

        grad_model = tf.keras.models.Model(
            inputs=self.hybrid_model.inputs,
            outputs=[
                self.hybrid_model.get_layer("top_conv").output,
                self.hybrid_model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model([img_array, clinical])

            if isinstance(predictions, list):
                predictions = predictions[0]

            loss = tf.reduce_mean(predictions)

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-8

        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (roi.shape[1], roi.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(roi, 0.6, heatmap, 0.4, 0)
        return overlay

    # ================= RUN =================
    def run_inference(self):

        if self.original_image is None:
            return

        self.image = self.original_image.copy()

        # Preprocessing
        self.image = segment_breast(self.image)
        enhanced = apply_clahe(self.image)

        # Detection
        img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).to(DEVICE)

        with torch.no_grad():
            pred = self.detector([img_tensor])[0]

        self.final_box = None
        self.detector_score = None

        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            if int(label) == 1 and score >= DET_THRESH:
                self.final_box = tuple(map(int, box.cpu().numpy()))
                self.detector_score = float(score)
                break

        roi = enhanced
        if self.final_box:
            x1, y1, x2, y2 = self.final_box
            roi = enhanced[y1:y2, x1:x2]
            cv2.rectangle(enhanced, (x1, y1), (x2, y2), (0, 0, 255), 9)
            # Draw thin inner white border
            cv2.rectangle(enhanced, (x1+3, y1+3), (x2-3, y2-3), (255, 255, 255), 2)
            cv2.putText(enhanced,f"Lesion ({self.detector_score:.2f})",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,0.8, 
            (0, 0, 255), 2)

        prob, uncertainty = self.hybrid_prediction(roi)

        self.status_label.setText(
            f"Cancer Probability: {int(prob*100)}% | Detection Score: {self.detector_score}"
        )
        self.explain_label.setText(f"Uncertainty: ± {uncertainty*100:.2f}%")

        gradcam = self.generate_gradcam(roi)
        edge = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 50, 150)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

        self.display_image(enhanced, self.panel_original)
        self.display_image(gradcam, self.panel_gradcam)
        self.display_image(enhanced, self.panel_enhanced)
        self.display_image(edge, self.panel_edge)


# ================= MAIN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MammographyAI()
    win.show()
    sys.exit(app.exec())
