import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment


class EmbaddingMemmoyTracker:
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
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        areaA = (boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
        areaB = (boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
        return interArea / float(areaA + areaB - interArea + 1e-6)

    def _get_similarity(self, emb, tid):
        bank = np.array(self.feature_banks[tid])
        return np.max(np.dot(bank, emb))

    def update(self, boxes, embeddings, labels, f_idx=None): 
        current_frame_output = {}
        assigned_dets, assigned_ids = set(), set()
        active_ids = list(self.feature_banks.keys())
        
        if len(boxes) > 0 and active_ids:
            cost_matrix = np.zeros((len(active_ids), len(boxes)))
            for i, tid in enumerate(active_ids):
                prev_center = self._get_center(self.last_positions[tid])
                for j, det_box in enumerate(boxes):
                    sim = self._get_similarity(embeddings[j], tid)
                    curr_center = self._get_center(det_box)
                    dist = np.linalg.norm(prev_center - curr_center)
                    dist_factor = np.exp(-self.dist_weight * dist / 100)
                    iou_bonus = 0.20 if self._iou(self.last_positions[tid], det_box) > self.iou_threshold else 0
                    cost_matrix[i, j] = 1.0 - (sim * dist_factor + iou_bonus)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if 1.0 - cost_matrix[r, c] > self.sim_threshold:
                    tid = active_ids[r]
                    current_frame_output[tid] = (boxes[c], labels[c], 1.0 - cost_matrix[r, c])
                    assigned_dets.add(c)
                    assigned_ids.add(tid)
                    self.feature_banks[tid].append(embeddings[c])
                    self.last_positions[tid] = boxes[c]
                    self.lost_counters[tid] = 0

        for i in range(len(boxes)):
            if i not in assigned_dets:
                best_match_id, max_sim = None, -1
                for tid in self.feature_banks:
                    if tid not in assigned_ids:
                        sim = self._get_similarity(embeddings[i], tid)
                        if sim > max_sim:
                            max_sim, best_match_id = sim, tid
                
                if best_match_id and max_sim > 0.65:
                    current_frame_output[best_match_id] = (boxes[i], labels[i], max_sim)
                    assigned_ids.add(best_match_id)
                    self.feature_banks[best_match_id].append(embeddings[i])
                    self.last_positions[best_match_id] = boxes[i]
                    self.lost_counters[best_match_id] = 0
                else:
                    new_id = self.next_id
                    self.feature_banks[new_id] = deque([embeddings[i]], maxlen=100)
                    self.last_positions[new_id] = boxes[i]
                    self.lost_counters[new_id] = 0
                    current_frame_output[new_id] = (boxes[i], labels[i], 1.0)
                    self.next_id += 1

        for tid in list(self.feature_banks.keys()):
            if tid not in current_frame_output:
                self.lost_counters[tid] += 1
                if self.lost_counters[tid] > self.max_lost:
                    del self.feature_banks[tid], self.last_positions[tid], self.lost_counters[tid]
        return current_frame_output



class InterpolativeTracker(EmbaddingMemmoyTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_history = {} # {tid: {f_idx: box_xyxy}}

    def update(self, boxes, embeddings, labels, f_idx):
        # Вызываем базовый метод. f_idx пробрасывается, так как ваш класс его требует
        tracks = super().update(boxes, embeddings, labels, f_idx)
        
        # Безопасное сохранение истории
        for tid, data in tracks.items():
            if tid not in self.full_history:
                self.full_history[tid] = {}
            
            # Если данные пришли кортежем (box, label, score), берем box [0]
            # Если сразу массив координат, берем его целиком
            if isinstance(data, (np.ndarray, list)) and len(data) == 4:
                self.full_history[tid][f_idx] = data
            else:
                self.full_history[tid][f_idx] = data[0]
                
        return tracks
    
    def get_interpolated_history(self, max_gap=50):
        """Заполняет пропуски в данных до max_gap кадров"""
        final_data = {}
        for tid, frames in self.full_history.items():
            sorted_f = sorted(frames.keys())
            if not sorted_f: continue
            
            new_frames = frames.copy()
            for i in range(len(sorted_f) - 1):
                f1, f2 = sorted_f[i], sorted_f[i+1]
                gap = f2 - f1
                # Если разрыв небольшой, интерполируем промежуточные позиции
                if 1 < gap <= max_gap:
                    box1, box2 = frames[f1], frames[f2]
                    for f_miss in range(f1 + 1, f2):
                        alpha = (f_miss - f1) / gap
                        interp_box = box1 * (1 - alpha) + box2 * alpha
                        new_frames[f_miss] = interp_box
            final_data[tid] = new_frames
        return final_data