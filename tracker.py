import numpy as np

DEFAULT_MAX_ABSENT = 15
IOU_MATCH_THRESHOLD = 0.30
CENTROID_FALLBACK_DIST = 80


def _iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def _centroid(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def _dist(c1, c2) -> float:
    return float(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))


class ObjectTracker:
    """IoU/centroid tracker. One instance handles all classes."""

    def __init__(self, max_absent: int = DEFAULT_MAX_ABSENT):
        self._tracks: dict = {}
        self._next_id: int = 0
        self._max_absent = max_absent

    def update(self, detections: list, frame_number: int) -> list:
        """Assign track_id and is_new to each Detection in-place."""
        track_ids = list(self._tracks.keys())
        matched_pairs = []
        matched_det_idxs = set()

        if track_ids and detections:
            scores = {}
            for ti, tid in enumerate(track_ids):
                tbox = self._tracks[tid]["bbox"]
                tlabel = self._tracks[tid]["label"]
                for di, det in enumerate(detections):
                    if det.label != tlabel:
                        continue
                    dbox = (det.x1, det.y1, det.x2, det.y2)
                    iou = _iou(tbox, dbox)
                    if iou > 0:
                        scores[(ti, di)] = iou
                    else:
                        d = _dist(_centroid(tbox), _centroid(dbox))
                        if d < CENTROID_FALLBACK_DIST:
                            scores[(ti, di)] = CENTROID_FALLBACK_DIST / (d + 1e-6) * 0.01

            matched_tracks = set()
            for (ti, di), score in sorted(scores.items(), key=lambda x: -x[1]):
                if score < IOU_MATCH_THRESHOLD:
                    break
                if ti in matched_tracks or di in matched_det_idxs:
                    continue
                tid = track_ids[ti]
                matched_pairs.append((di, tid))
                matched_tracks.add(ti)
                matched_det_idxs.add(di)

        for di, tid in matched_pairs:
            det = detections[di]
            self._tracks[tid].update(bbox=(det.x1, det.y1, det.x2, det.y2),
                                     centroid=det.center, last_frame=frame_number,
                                     is_new=False)
            det.track_id = tid
            det.is_new = False

        for di, det in enumerate(detections):
            if di in matched_det_idxs:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = dict(bbox=(det.x1, det.y1, det.x2, det.y2),
                                     centroid=det.center, label=det.label,
                                     last_frame=frame_number, is_new=True)
            det.track_id = tid
            det.is_new = True

        stale = [tid for tid, t in self._tracks.items()
                 if (frame_number - t["last_frame"]) > self._max_absent]
        for tid in stale:
            del self._tracks[tid]

        return detections

    def reset(self):
        self._tracks.clear()
        self._next_id = 0
