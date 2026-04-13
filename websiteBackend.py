import os
import cv2
import torch
import numpy as np
import tensorflow as tf

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


# ================= LOAD MODELS =================
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    det_path = os.path.join(BASE_DIR, "fasterrcnn_pseudo_best.pth")

    detector = fasterrcnn_resnet50_fpn(weights=None)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    ckpt = torch.load(det_path, map_location=DEVICE)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    detector.load_state_dict(state_dict)
    detector.to(DEVICE).eval()

    hybrid_path = os.path.join(BASE_DIR, "Hybrid CNN", "best_hybrid_model_overall.keras")
    hybrid_model = keras.models.load_model(hybrid_path, compile=False)

    return detector, hybrid_model


# ================= HYBRID =================
def hybrid_prediction(model, image, age, birads, density):

    clinical = np.array([[age, birads, density]], dtype="float32")

    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = []

    for _ in range(MC_RUNS):
        p = model([img_array, clinical], training=True)
        if isinstance(p, list):
            p = p[0]
        preds.append(p.numpy()[0][0])

    return float(np.mean(preds)), float(np.std(preds))


# ================= GRAD-CAM =================
def generate_gradcam(model, roi, age, birads, density):

    resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    clinical = np.array([[age, birads, density]], dtype="float32")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer("top_conv").output, model.output]
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

    return cv2.addWeighted(roi, 0.6, heatmap, 0.4, 0)


# ================= FULL PIPELINE =================
def run_pipeline(detector, hybrid_model, image, age, birads, density):

    image = segment_breast(image)
    enhanced = apply_clahe(image)

    img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).to(DEVICE)

    with torch.no_grad():
        pred = detector([img_tensor])[0]

    final_box = None
    score_val = None

    for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
        if int(label) == 1 and score >= DET_THRESH:
            final_box = tuple(map(int, box.cpu().numpy()))
            score_val = float(score)
            break

    roi = enhanced
    if final_box:
        x1, y1, x2, y2 = final_box
        roi = enhanced[y1:y2, x1:x2]
        cv2.rectangle(enhanced, (x1, y1), (x2, y2), (0, 0, 255), 4)

    prob, uncertainty = hybrid_prediction(hybrid_model, roi, age, birads, density)

    gradcam = generate_gradcam(hybrid_model, roi, age, birads, density)

    edge = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 50, 150)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    return enhanced, gradcam, edge, prob, uncertainty, score_val
