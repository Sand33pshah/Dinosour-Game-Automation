import pyautogui
import cv2
import numpy as np
import time


def motion_detection_with_spacebar():
    # Define the detection zone (x, y, width, height) relative to the capture region
    detection_zone = (255, 65, 45, 105)  # Adjust these values as needed

    # Wait a moment before capturing background
    # print("Capturing background in 2 seconds...")
    # time.sleep(2)

    screenshot = pyautogui.screenshot(region=(0, 100, 900, 220))
    background = np.array(screenshot)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    background = cv2.GaussianBlur(background, (15, 15), 0)

    print("Motion Detection Active with Auto Spacebar")
    print("Detection zone highlighted in blue")
    print("Press 'r' to reset background")
    print("Press 'q' to quit")
    print("Press 's' to toggle spacebar auto-press")

    auto_spacebar = True

    while True:
        screenshot = pyautogui.screenshot(region=(0, 100, 900, 220))
        frame = np.array(screenshot)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        diff = cv2.absdiff(background, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Highlight the detection zone on the frame
        x, y, w, h = detection_zone
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h),
                      (255, 0, 0), 2)  # Blue rectangle
        cv2.putText(frame_bgr, 'Detection Zone', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Check for motion in the detection zone
        # Extract the detection zone from threshold image
        zone_thresh = thresh[y:y+h, x:x+w]
        motion_in_zone = cv2.countNonZero(
            zone_thresh) > 100  # Adjust sensitivity as needed

        if motion_in_zone:
            # Highlight detection zone in red when motion detected
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame_bgr, 'MOTION DETECTED!', (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Press spacebar if auto mode is on and cooldown has passed
            if auto_spacebar:
                pyautogui.press('space')
                print("Spacebar pressed!")

        # Status display
        status_text = f"Auto Spacebar: {'ON' if auto_spacebar else 'OFF'}"
        cv2.putText(frame_bgr, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Motion Detection', frame_bgr)
        # cv2.imshow('Threshold', thresh)
        # Show what's happening in the zone
        # cv2.imshow('Detection Zone', zone_thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            background = gray.copy()
            print("Background reset")
        elif key == ord('s'):
            auto_spacebar = not auto_spacebar
            print(f"Auto spacebar: {'ON' if auto_spacebar else 'OFF'}")

    cv2.destroyAllWindows()


motion_detection_with_spacebar()
