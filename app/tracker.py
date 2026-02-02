def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter + 1e-6)


class SimpleIOUTracker:
    def __init__(self, iou_threshold=0.35, max_lost=30, alpha=0.7):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.alpha = alpha  # Smoothing factor (0.7 means 70% new, 30% old)

        self.next_id = 1
        self.tracks = {}  # id -> {box, lost, gender}

    def update(self, detections):
        # detections is now expected to be a list of (box, gender_label)
        
        if not self.tracks:
            for box, gender in detections:
                self.tracks[self.next_id] = {"box": box, "lost": 0, "gender": gender}
                self.next_id += 1
            return [(tid, t["box"]) for tid, t in self.tracks.items()]

        assigned = set()
        results = []

        # Parse inputs
        # We need to match existing tracks to new detections
        # det_items = [(box, gender), ...]
        
        for det_idx, (box, gender) in enumerate(detections):
            best_tid = None
            best_score = 0.0

            for tid, tr in self.tracks.items():
                if tid in assigned:
                    continue
                score = iou(box, tr["box"])
                if score > best_score:
                    best_score = score
                    best_tid = tid

            if best_tid is not None and best_score >= self.iou_threshold:
                # HIT: Update track
                
                # Apply smoothing
                old_box = self.tracks[best_tid]["box"]
                new_box = [
                    self.alpha * box[0] + (1 - self.alpha) * old_box[0],
                    self.alpha * box[1] + (1 - self.alpha) * old_box[1],
                    self.alpha * box[2] + (1 - self.alpha) * old_box[2],
                    self.alpha * box[3] + (1 - self.alpha) * old_box[3]
                ]
                
                self.tracks[best_tid]["box"] = new_box
                self.tracks[best_tid]["lost"] = 0
                
                # Update gender if not set or if needed (here we prioritize known)
                if self.tracks[best_tid]["gender"] in [None, "unknown"] and gender not in [None, "unknown"]:
                    self.tracks[best_tid]["gender"] = gender
                
                assigned.add(best_tid)
                results.append((best_tid, new_box))
            else:
                # NEW TRACK
                self.tracks[self.next_id] = {"box": box, "lost": 0, "gender": gender}
                results.append((self.next_id, box))
                self.next_id += 1

        # Remove lost tracks
        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return results

    def set_gender(self, tid, gender):
        if tid in self.tracks:
            self.tracks[tid]["gender"] = gender

    def get_gender(self, tid):
        if tid in self.tracks:
            return self.tracks[tid]["gender"]
        return None
