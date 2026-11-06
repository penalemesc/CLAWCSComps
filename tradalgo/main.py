import cv2
import numpy as np


def segment_vehicles(image, max_width=1000):
    """Classical rear-vehicle segmentation: red-light pairs seed GrabCut.

    Returns uint8 mask: 255 vehicle, 0 background.

    Run program with input images as parameters such as:
    ❯ python main.py car.jpg dashcam.jpg
    """
    H, W = image.shape[:2]
    scale = min(1.0, max_width / W)
    frame = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, (0, 75, 70), (12, 255, 255)) |
           cv2.inRange(hsv, (168, 75, 70), (179, 255, 255)))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lights = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        if 6 < area < w * h * 0.025 and y > 0.15 * h and bw > 2:
            lights.append((x + bw / 2, y + bh / 2, area))

    # Score candidate pairs, favoring similar elevation and blob area.
    candidates = []
    for i, (x1, y1, a1) in enumerate(lights):
        for j in range(i + 1, len(lights)):
            x2, y2, a2 = lights[j]
            d = abs(x2 - x1)
            if not 0.035 * w < d < 0.55 * w:
                continue
            if abs(y2 - y1) > max(8, 0.18 * d):
                continue
            ratio = min(a1, a2) / max(a1, a2)
            if ratio < 0.15 or min(a1, a2) / (d * d) < 0.0007:
                continue
            score = ratio + 0.15 * np.log1p(min(a1, a2)) - abs(y2 - y1) / d
            candidates.append((score, i, j))

    used = set()
    result = np.zeros((h, w), np.uint8)
    for _, i, j in sorted(candidates, reverse=True):
        if i in used or j in used:
            continue
        x1, y1, _ = lights[i]
        x2, y2, _ = lights[j]
        d = abs(x2 - x1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Loose car-sized ROI, derived from the taillight separation.
        x0 = max(0, int(cx - 0.70 * d))
        x3 = min(w, int(cx + 0.70 * d))
        y0 = max(0, int(cy - 0.52 * d))
        y3 = min(h, int(cy + 0.39 * d))
        if x3 - x0 < 12 or y3 - y0 < 12:
            continue
        roi = frame[y0:y3, x0:x3]
        rh, rw = roi.shape[:2]

        # Initial color model: center probably foreground; border background.
        gc = np.full((rh, rw), cv2.GC_PR_BGD, np.uint8)
        gc[int(rh*.10):int(rh*.90), int(rw*.08):int(rw*.92)] = cv2.GC_PR_FGD
        t = max(2, min(rh, rw) // 35)
        gc[:t] = cv2.GC_BGD
        gc[-t:] = cv2.GC_BGD
        gc[:, :t] = cv2.GC_BGD
        gc[:, -t:] = cv2.GC_BGD
        for k in (i, j):
            lx, ly = int(lights[k][0] - x0), int(lights[k][1] - y0)
            cv2.circle(gc, (lx, ly), max(2, int(.015*d)), cv2.GC_FGD, -1)

        try:
            cv2.grabCut(roi, gc, None, np.zeros((1, 65), np.float64),
                        np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            continue
        foreground = np.uint8((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) * 255
        # Select connected components containing either taillight.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        selected = set()
        for k in (i, j):
            lx, ly = int(lights[k][0] - x0), int(lights[k][1] - y0)
            if 0 <= lx < rw and 0 <= ly < rh and labels[ly, lx]:
                selected.add(labels[ly, lx])
        if selected:
            foreground = np.uint8(np.isin(labels, list(selected))) * 255
        result[y0:y3, x0:x3] |= foreground
        used.update((i, j))
    # Suppress tiny isolated false positives relative to largest detected vehicle.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(result, 8)
    if count > 1:
        largest = max(stats[1:, cv2.CC_STAT_AREA])
        valid = [k for k in range(1, count)
                 if stats[k, cv2.CC_STAT_AREA] >= max(80, 0.08 * largest)]
        result = np.uint8(np.isin(labels, valid)) * 255
    return cv2.resize(result, (W, H), interpolation=cv2.INTER_NEAREST)


if __name__ == '__main__':
    import sys
    for path in sys.argv[1:]:
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(path)
        mask = segment_vehicles(frame)
        target = path.rsplit('.', 1)[0] + '_mask.png'
        cv2.imwrite(target, mask)
        print(target, 'foreground pixels:', np.count_nonzero(mask))
