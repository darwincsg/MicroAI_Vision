import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

def run_pose_inference(model, input_folder, output_folder, log_file):

    os.makedirs(output_folder, exist_ok=True)
    log = open(log_file, "w")

    total_start = time.time()

    # Iterar sobre las imágenes del dataset
    for img_path in sorted(Path(input_folder).glob("*")):
        start = time.time()

        # Cargar imagen original
        img = cv2.imread(str(img_path))
        
        img_padded = resize_keep_aspect(img)
        
        # Inferencia
        results = model.predict(source=img_padded)

        # Dibujar los resultados (YOLO ya trae método de visualización)
        annotated_img = results[0].plot()

        # Guardar imagen de salida
        out_path = Path(output_folder) / img_path.name
        cv2.imwrite(str(out_path), annotated_img)

        # Obtener detecciones y keypoints
        boxes = results[0].boxes
        keypoints = results[0].keypoints
        
        if boxes is None or len(boxes) == 0:
            log.write(f"{img_path.name}: No detections\n")
            continue
        
        # Loguear información
        log.write(f"\n=== {img_path.name} ===\n")
        for i, (box, kpts_xy, kpts_conf) in enumerate(
            zip(boxes, keypoints.xy, keypoints.conf)
        ):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            log.write(f"Person {i+1}: box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), conf={conf:.3f}\n")

            for j, (xy, c) in enumerate(zip(kpts_xy, kpts_conf)):
                kx, ky = xy
                log.write(f"   Keypoint {j:02d}: ({kx:.1f}, {ky:.1f}), conf={c:.3f}\n")

        end = time.time()
        print(f"{img_path.name} processed in {(end-start)*1000:.2f} ms")

    total_end = time.time()
    total_time = total_end - total_start
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    log.write(f"\nTotal processing time: {total_time:.2f} seconds\n")
    log.close()

def resize_keep_aspect(img, target_size=480):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Use INTER_NEAREST for faster resizing (slight quality loss)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    pad_x, pad_y = target_size - new_w, target_size - new_h
    padded = cv2.copyMakeBorder(resized, 0, pad_y, 0, pad_x,cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded


def main():
    #model = YOLO("yolo11n-pose.pt")  # Modelo YOLO Pose
    ov_model = YOLO('yolo11n-pose_int8_openvino_model')  # OpenVino model 

    dummy_image = np.zeros((320, 320, 3), dtype=np.uint8)

    print("Calentando el modelo...")
    _ = ov_model.predict(source=dummy_image, imgsz=320)
    print("Calentamiento completado.")
    
    input_folder = "./Dataset/Tests/"   # Carpeta con tus imágenes
    output_folder = "./Results/Test7/Images"
    log_file = "./Results/Test7/logs/logs.txt"
    run_pose_inference(ov_model, input_folder, output_folder, log_file)
