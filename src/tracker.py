import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

class EmbeddingMemoryTracker:
    def __init__(self, sim_threshold=0.55, iou_threshold=0.15, dist_weight=0.05):
        self.sim_threshold = sim_threshold
        self.iou_threshold = iou_threshold
        self.dist_weight = dist_weight

        self.next_id = 1
        self.feature_banks = {}
        self.last_positions = {}
        self.lost_counters = {}
        self.max_lost = 1500

    def _get_center(self, bbox):
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    def _iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        areaA = (boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
        areaB = (boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
        return inter / (areaA + areaB - inter + 1e-6)

    def _get_similarity(self, emb, tid):
        return np.max(np.dot(self.feature_banks[tid], emb))

    def update(self, boxes, embeddings, labels):
        current_frame_output = {}
        assigned_dets, assigned_ids = set(), set()
        active_ids = list(self.feature_banks.keys())
        cost_matrix = np.zeros((len(active_ids), len(boxes)))
        for i, tid in enumerate(active_ids):
            prev_pos = self.last_positions[tid]
            prev_center = self._get_center(prev_pos)
            for j, det_box in enumerate(boxes):
                sim = self._get_similarity(embeddings[j], tid)
                dist = np.linalg.norm(prev_center - self._get_center(det_box))
                dist_factor = np.exp(-self.dist_weight * dist / 100)
                iou_bonus = 0.2 if self._iou(prev_pos, det_box) > self.iou_threshold else 0
                cost_matrix[i, j] = 1.0 - (sim * dist_factor + iou_bonus)

        rows, cols = linear_sum_assignment(cost_matrix)
        for r, c in zip(rows, cols):
            if 1.0 - cost_matrix[r, c] > self.sim_threshold:
                tid = active_ids[r]
                self._update_track(tid, boxes[c], embeddings[c])
                current_frame_output[tid] = (boxes[c], labels[c], 1.0 - cost_matrix[r, c])
                assigned_dets.add(c); assigned_ids.add(tid)

        # Второй этап: сопоставление неназначенных детекций
        for i in range(len(boxes)):
            if i in assigned_dets: continue
            
            best_id, max_sim = None, 0.65 # Сразу ставим порог
            for tid, bank in self.feature_banks.items():
                if tid in assigned_ids: continue
                sim = self._get_similarity(embeddings[i], tid)
                if sim > max_sim:
                    best_id, max_sim = tid, sim

            if best_id: # Если best_id не None, значит max_sim > 0.65
                self._update_track(best_id, boxes[i], embeddings[i])
                current_frame_output[best_id] = (boxes[i], labels[i], max_sim)
                assigned_ids.add(best_id)
            else:
                self._create_track(boxes[i], embeddings[i], labels[i])
                current_frame_output[self.next_id - 1] = (boxes[i], labels[i], 1.0)

        # Очистка потерянных треков
        for tid in list(self.feature_banks.keys()):
            if tid not in current_frame_output:
                self.lost_counters[tid] += 1
                if self.lost_counters[tid] > self.max_lost:
                    del self.feature_banks[tid], self.last_positions[tid], self.lost_counters[tid]

        return current_frame_output

    def _update_track(self, tid, box, emb):
        self.feature_banks[tid].append(emb)
        self.last_positions[tid] = box
        self.lost_counters[tid] = 0

    def _create_track(self, box, emb, label):
        self.feature_banks[self.next_id] = deque([emb], maxlen=100)
        self.last_positions[self.next_id] = box
        self.lost_counters[self.next_id] = 0
        self.next_id += 1

class InterpolativeTracker(EmbeddingMemoryTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_history = {}

    def update(self, boxes, embeddings, labels, f_idx):
        tracks = super().update(boxes, embeddings, labels)
        for tid, (box, label, score) in tracks.items(): # Распаковка кортежа сразу
            if tid not in self.full_history:
                self.full_history[tid] = {}
            self.full_history[tid][f_idx] = box
        return tracks

    def _compute_acceleration(self, frames, keys, idx):
        if idx < 2: return 0
        # Векторный расчет ускорения (центры боксов)
        pts = [(frames[keys[i]][:2] + frames[keys[i]][2:]) / 2 for i in range(idx-2, idx+1)]
        v1, v2 = pts[1] - pts[0], pts[2] - pts[1]
        return np.linalg.norm(v2 - v1)

    def get_interpolated_history(self, max_gap=30):
        final_data = {}
        for tid, frames in self.full_history.items():
            keys = sorted(frames.keys())
            if not keys: continue
            
            new_frames = frames.copy()
            for i in range(len(keys) - 1):
                f1, f2 = keys[i], keys[i+1]
                gap = f2 - f1
                if 1 < gap <= max_gap:
                    b1, b2 = frames[f1], frames[f2]
                    accel = self._compute_acceleration(frames, keys, i)
                    w = np.clip(accel / 15.0, 0, 1)

                    for f_miss in range(f1 + 1, f2):
                        t = (f_miss - f1) / gap
                        linear = b1 * (1 - t) + b2 * t
                        if i >= 1:
                            b0 = frames[keys[i-1]]
                            quad = (1-t)**2 * b0 + 2*(1-t)*t*b1 + t**2 * b2
                            interp = (1 - w) * linear + w * quad
                        else:
                            interp = linear
                        new_frames[f_miss] = interp
            final_data[tid] = new_frames
        return final_data