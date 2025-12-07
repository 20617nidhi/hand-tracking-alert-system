# hand_boundary_poc.py
# Prototype: Hand boundary alert system (classical CV, no MediaPipe/OpenPose)
# Requirements: opencv-python, numpy
# Run: python hand_boundary_poc.py

import cv2
import numpy as np
import time
import math

# -------------- Parameters (tune these) --------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# skin color HSV range (works reasonably well for many lighting conditions; tune if needed)
LOWER_HSV = np.array([0, 30, 60])    # lower bound for skin
UPPER_HSV = np.array([25, 255, 255]) # upper bound for skin

# morphological ops
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Virtual boundary rectangle coordinates (x1,y1) top-left, (x2,y2) bottom-right
# You can change these to any object/shape
rect_w, rect_h = int(FRAME_WIDTH * 0.4), int(FRAME_HEIGHT * 0.3)
rect_x1 = (FRAME_WIDTH - rect_w) // 2
rect_y1 = (FRAME_HEIGHT - rect_h) // 2
rect_x2 = rect_x1 + rect_w
rect_y2 = rect_y1 + rect_h

# distance thresholds expressed as fraction of frame diagonal
FRAME_DIAG = math.hypot(FRAME_WIDTH, FRAME_HEIGHT)
DANGER_THRESH = 0.06 * FRAME_DIAG   # closer than this => DANGER
WARNING_THRESH = 0.18 * FRAME_DIAG  # closer than this => WARNING (but > danger)

# smoothing for fingertip position (for stable visualization)
ALPHA = 0.6

# ------------------------------------------------------

def distance_point_to_rect(px, py, x1, y1, x2, y2):
    """Euclidean distance from point to rectangle (0 if inside)."""
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.hypot(dx, dy)

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:
        return None
    return largest

def contour_farthest_point_from_centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    # find contour point with maximum distance from centroid
    pts = cnt.reshape(-1, 2)
    dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
    idx = np.argmax(dists)
    fx, fy = pts[idx]
    return (fx, fy), (cx, cy)

def preprocess_frame(frame):
    # resize & blur
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # skin mask
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    # morphological cleaning
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # windows: use CAP_DSHOW for less lag
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Set camera resolution if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    prev_tip = None
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = preprocess_frame(frame)

        cnt = find_largest_contour(mask)
        fingertip = None
        centroid = None

        if cnt is not None:
            res = contour_farthest_point_from_centroid(cnt)
            if res is not None:
                (fx, fy), (cx, cy) = res
                fingertip = (int(fx), int(fy))
                centroid = (int(cx), int(cy))
                # smoothing
                if prev_tip is None:
                    smoothed = fingertip
                else:
                    sx = int(ALPHA * fingertip[0] + (1 - ALPHA) * prev_tip[0])
                    sy = int(ALPHA * fingertip[1] + (1 - ALPHA) * prev_tip[1])
                    smoothed = (sx, sy)
                prev_tip = smoothed
        else:
            prev_tip = None

        # draw virtual rectangle
        vis = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)).copy()
        cv2.rectangle(vis, (rect_x1, rect_y1), (rect_x2, rect_y2), (200, 200, 200), 2)

        # compute distance and state
        state = "NO HAND"
        color = (200, 200, 200)
        dist_px = None
        if prev_tip is not None:
            px, py = prev_tip
            dist_px = distance_point_to_rect(px, py, rect_x1, rect_y1, rect_x2, rect_y2)
            if dist_px <= DANGER_THRESH:
                state = "DANGER"
                color = (0, 0, 255)  # red
            elif dist_px <= WARNING_THRESH:
                state = "WARNING"
                color = (0, 165, 255)  # orange
            else:
                state = "SAFE"
                color = (0, 255, 0)  # green
            # draw fingertip and connecting line
            cv2.circle(vis, (px, py), 8, color, -1)
            # nearest point on rectangle for visualization
            nearest_x = min(max(px, rect_x1), rect_x2)
            nearest_y = min(max(py, rect_y1), rect_y2)
            cv2.line(vis, (px, py), (nearest_x, nearest_y), color, 2)
            cv2.circle(vis, (nearest_x, nearest_y), 6, color, -1)

            # draw contour and centroid for debug
            cv2.drawContours(vis, [cnt], -1, (255, 0, 255), 1)
            if centroid is not None:
                cv2.circle(vis, centroid, 4, (255, 0, 0), -1)

        # show state overlay
        overlay_text = f"State: {state}"
        cv2.putText(vis, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if state == "DANGER":
            # big warning
            cv2.putText(vis, "DANGER DANGER", (int(FRAME_WIDTH*0.15), int(FRAME_HEIGHT*0.55)),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 4)

        # FPS calc
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, FRAME_HEIGHT-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # show mask small window for debugging
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((cv2.resize(vis, (FRAME_WIDTH, FRAME_HEIGHT)), cv2.resize(mask_bgr, (FRAME_WIDTH, FRAME_HEIGHT))))
        # show only main view (uncomment to show combined)
        cv2.imshow("Hand Boundary Alert - Live", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
