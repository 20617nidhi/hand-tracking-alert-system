# hand_boundary.py
import cv2
import numpy as np
import time

# ----------------- CONFIG ----------------
BOUND_LEFT = 200
BOUND_RIGHT = 440
BOUND_TOP = 120
BOUND_BOTTOM = 360

SAVE_VIDEO = False       # Set True to save demo video to file
VIDEO_OUT = "demo_output.avi"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# ------------------------------------------

def get_hand_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

# Adjusted thresholds for SAFE/WARNING/DANGER
def classify_state(dist):
    if dist > 40:        # green
        return "SAFE", (0, 255, 0)
    elif dist > 20:      # yellow
        return "WARNING", (0, 255, 255)
    else:                # red
        return "DANGER DANGER", (0, 0, 255)

def main():
    global SAVE_VIDEO

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Make sure it’s not used by another application.")
        input("Press Enter to exit...")
        return

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(VIDEO_OUT, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))

    prev = time.time()
    fps = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Skin segmentation (YCrCb)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        mask = cv2.inRange(cr, 135, 180)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        hand = get_hand_contour(mask)

        state_text = "NO HAND DETECTED"
        color = (255, 255, 255)

        if hand is not None and cv2.contourArea(hand) > 2000:
            M = cv2.moments(hand)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Only consider hand **inside the rectangle**
                if BOUND_LEFT < cx < BOUND_RIGHT and BOUND_TOP < cy < BOUND_BOTTOM:
                    cv2.drawContours(frame, [hand], -1, (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                    # distances to rectangle boundary
                    dist_left = BOUND_LEFT - cx if cx < BOUND_LEFT else cx - BOUND_LEFT
                    dist_right = cx - BOUND_RIGHT if cx > BOUND_RIGHT else BOUND_RIGHT - cx
                    dist_top = BOUND_TOP - cy if cy < BOUND_TOP else cy - BOUND_TOP
                    dist_bottom = cy - BOUND_BOTTOM if cy > BOUND_BOTTOM else BOUND_BOTTOM - cy

                    min_dist = min(abs(dist_left), abs(dist_right), abs(dist_top), abs(dist_bottom))
                    state_text, color = classify_state(min_dist)

        # Draw virtual boundary
        cv2.rectangle(frame, (BOUND_LEFT, BOUND_TOP), (BOUND_RIGHT, BOUND_BOTTOM), (255, 255, 255), 2)

        # Draw state and FPS
        cv2.putText(frame, state_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # FPS calculation
        frame_count += 1
        if frame_count >= 10:
            now = time.time()
            fps = frame_count / (now - prev)
            prev = now
            frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        cv2.imshow("Hand Tracking Boundary System", frame)
        # cv2.imshow("Mask", mask)  # optional debug

        if SAVE_VIDEO and writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord('s'):  # toggle save
            SAVE_VIDEO = not SAVE_VIDEO

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
