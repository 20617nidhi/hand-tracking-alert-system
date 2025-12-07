# app.py – Streamlit version (WebRTC + OpenCV)
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

LOWER_HSV = np.array([0, 30, 60])
UPPER_HSV = np.array([25, 255, 255])
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

rect_w, rect_h = int(FRAME_WIDTH * 0.4), int(FRAME_HEIGHT * 0.3)
rect_x1 = (FRAME_WIDTH - rect_w) // 2
rect_y1 = (FRAME_HEIGHT - rect_h) // 2
rect_x2 = rect_x1 + rect_w
rect_y2 = rect_y1 + rect_h

FRAME_DIAG = np.hypot(FRAME_WIDTH, FRAME_HEIGHT)
DANGER_THRESH = 0.06 * FRAME_DIAG
WARNING_THRESH = 0.18 * FRAME_DIAG

ALPHA = 0.6

def distance_point_to_rect(px, py, x1, y1, x2, y2):
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return np.hypot(dx, dy)

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
    pts = cnt.reshape(-1, 2)
    dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
    idx = np.argmax(dists)
    return (pts[idx][0], pts[idx][1]), (cx, cy)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask

class HandBoundaryTracker(VideoTransformerBase):
    def __init__(self):
        self.prev_tip = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        mask = preprocess_frame(img)

        cnt = find_largest_contour(mask)
        fingertip = None

        if cnt is not None:
            res = contour_farthest_point_from_centroid(cnt)
            if res is not None:
                (fx, fy), (cx, cy) = res
                fingertip = (int(fx), int(fy))

                if self.prev_tip is None:
                    self.prev_tip = fingertip
                else:
                    sx = int(ALPHA * fingertip[0] + (1 - ALPHA) * self.prev_tip[0])
                    sy = int(ALPHA * fingertip[1] + (1 - ALPHA) * self.prev_tip[1])
                    self.prev_tip = (sx, sy)

        vis = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT)).copy()
        cv2.rectangle(vis, (rect_x1, rect_y1), (rect_x2, rect_y2), (200, 200, 200), 2)

        state = "NO HAND"
        color = (200, 200, 200)

        if self.prev_tip is not None:
            px, py = self.prev_tip
            dist_px = distance_point_to_rect(px, py, rect_x1, rect_y1, rect_x2, rect_y2)

            if dist_px <= DANGER_THRESH:
                state = "DANGER"
                color = (0, 0, 255)
            elif dist_px <= WARNING_THRESH:
                state = "WARNING"
                color = (0, 165, 255)
            else:
                state = "SAFE"
                color = (0, 255, 0)

            cv2.circle(vis, (px, py), 8, color, -1)
            nearest_x = min(max(px, rect_x1), rect_x2)
            nearest_y = min(max(py, rect_y1), rect_y2)
            cv2.line(vis, (px, py), (nearest_x, nearest_y), color, 2)

        cv2.putText(vis, f"STATE: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if state == "DANGER":
            cv2.putText(vis, "DANGER DANGER", (80, 260),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)

        return vis


# ---------------- UI ----------
st.title("Hand Boundary Detection — Streamlit Web App (POC)")

webrtc_streamer(
    key="hand-demo",
    video_transformer_factory=HandBoundaryTracker,
    media_stream_constraints={"video": True, "audio": False},
)
