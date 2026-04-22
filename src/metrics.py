import numpy as np

class HOTACalculator:
    def __init__(self, thresholds=np.linspace(0.05, 0.95, 19)):
        self.thresholds = thresholds
        self.data = []

    def add(self, frame, gt_ids, gt_boxes, t_ids, t_boxes):
        self.data.append({
            'f': frame, 
            'g_ids': gt_ids, 
            'g_b': np.array(gt_boxes), 
            't_ids': t_ids, 
            't_b': np.array(t_boxes)
        })

    def _iou(self, b1, b2):
        if b1.size == 0 or b2.size == 0: 
            return np.empty((len(b1), len(b2)))
        # Правильный расчет IoU с broadcasting для матриц разных размеров
        x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
        y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
        x2 = np.minimum(b1[:, None, 0] + b1[:, None, 2], b2[None, :, 0] + b2[None, :, 2])
        y2 = np.minimum(b1[:, None, 1] + b1[:, None, 3], b2[None, :, 1] + b2[None, :, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (b1[:, 2] * b1[:, 3])[:, None]
        area2 = (b2[:, 2] * b2[:, 3])[None, :]
        return inter / (area1 + area2 - inter + 1e-6)

    def compute(self):
        if not self.data: return 0.0
        hota_th = []
        for thr in self.thresholds:
            tp, fp, fn = 0, 0, 0
            for f in self.data:
                if f['g_b'].size == 0: 
                    fp += len(f['t_b']); continue
                if f['t_b'].size == 0:
                    fn += len(f['g_b']); continue
                
                iou = self._iou(f['g_b'], f['t_b'])
                matches = iou >= thr
                c_tp = min(np.any(matches, axis=1).sum(), np.any(matches, axis=0).sum())
                tp += c_tp
                fp += (len(f['t_b']) - c_tp)
                fn += (len(f['g_b']) - c_tp)
            hota_th.append(np.sqrt(tp / (tp + fp + fn + 1e-6)))
        return np.mean(hota_th) * 100
