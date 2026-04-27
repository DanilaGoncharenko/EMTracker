import os
import sys
import cv2
import yaml
import numpy as np
import motmetrics as mm
from ultralytics import YOLO
import warnings
import torch

from src.reid import SoftAttentionReIDExtractor
from src.tracker import InterpolativeTracker
from src.metrics import HOTACalculator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


warnings.filterwarnings("ignore")
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda x, **kwargs: np.asarray(x, dtype=float, **kwargs)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_interpolated_benchmark(cfg):
    paths = cfg['paths']
    weights_cfg = cfg['weights']
    detect_cfg = cfg['detection']
    track_cfg = cfg['tracker']
    
    # ПРОВЕРКА CUDA: если в конфиге cuda, но она недоступна — переключаемся на cpu
    device = detect_cfg['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # Инициализация моделей
    model = YOLO(weights_cfg['active_model'])
    extractor = SoftAttentionReIDExtractor(device=device)
    
    my_tracker = InterpolativeTracker(
        sim_threshold=track_cfg['sim_threshold'],
        iou_threshold=track_cfg['iou_threshold'],
        dist_weight=track_cfg['dist_weight']
    )
    my_tracker.max_lost = track_cfg['max_lost']
    hota_calc = HOTACalculator()
    
    cap = cv2.VideoCapture(paths['video_source'])
   frames_cache = []
    f_idx = 1
    
   
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frames_cache.append(frame.copy())
        # Используем device и параметры из конфига
        res = model.predict(frame, conf=detect_cfg['conf_threshold'], imgsz=detect_cfg['imgsz'], device=device, verbose=False)[0]
        
        if res.boxes:
            b_xyxy = res.boxes.xyxy.cpu().numpy()
            embs, v_idx = extractor.get_embeddings(frame, b_xyxy)
            if len(v_idx) > 0:
                cls_names = [res.names[int(c)] for c in res.boxes.cls[v_idx]]
                my_tracker.update(b_xyxy[v_idx], embs, cls_names, f_idx)
        f_idx += 1
    cap.release()


    interpolated_history = my_tracker.get_interpolated_history(max_gap=cfg['interpolation']['max_gap'])

    acc = mm.MOTAccumulator(auto_id=True)
    gt_db = mm.io.loadtxt(paths['gt_path'], fmt="mot15-2D", min_confidence=1)
    
    h, w = frames_cache[0].shape[:2]
    output_path = os.path.join(paths['output_dir'], 'final_interpolated.mp4')
    os.makedirs(paths['output_dir'], exist_ok=True)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    for current_f, frame in enumerate(frames_cache, 1):
        t_ids, t_boxes_xywh = [], []
        
        if current_f in gt_db.index:
            gt_f = gt_db.loc[current_f]
            # Обработка случая, когда в кадре один объект (Series vs DataFrame)
            if len(gt_f.shape) == 1:
                g_ids = [gt_f.name] if hasattr(gt_f, 'name') else [0]
                g_boxes = gt_f[['X', 'Y', 'Width', 'Height']].values.reshape(1, 4)
            else:
                g_ids = gt_f.index.values
                g_boxes = gt_f[['X', 'Y', 'Width', 'Height']].values
        else:
            g_ids, g_boxes = [], np.empty((0, 4))

        for tid, f_map in interpolated_history.items():
            if current_f in f_map:
                box = f_map[current_f]
                if isinstance(box, (np.ndarray, list)) and len(box) >= 4:
                    t_ids.append(tid)
                    bw, bh = float(box[2] - box[0]), float(box[3] - box[1])
                    t_boxes_xywh.append([float(box[0]), float(box[1]), bw, bh])
                    
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"ID:{tid}", (x1, y1-10), 0, 0.6, (0, 255, 255), 2)

        # Обновление метрик
        dists = mm.distances.iou_matrix(g_boxes, t_boxes_xywh, max_iou=0.5)
        acc.update(g_ids, t_ids, dists)
        hota_calc.add(current_f, g_ids, g_boxes, t_ids, t_boxes_xywh)
        out.write(frame)
    
    out.release()

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches'], name='EMTracker')
    summary['hota'] = hota_calc.compute()
    
    print("\n" + "="*40)
    print("RESULTS:", summary)
    print("="*40)

if __name__ == "__main__":
    config = load_config("configs/tracker_config.yaml")
    run_interpolated_benchmark(config)
