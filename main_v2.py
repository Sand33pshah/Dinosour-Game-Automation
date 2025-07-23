import pyautogui
import cv2
import numpy as np
import time


def preprocess_template(template_path):
    """Load and preprocess template for better matching"""
    template = cv2.imread(template_path, 0)  # Load as grayscale
    if template is None:
        print(f"Warning: Could not load {template_path}")
        return None

    # Apply Gaussian blur to reduce noise
    template = cv2.GaussianBlur(template, (3, 3), 0)

    # Optional: Apply edge detection for better matching
    # edges = cv2.Canny(template, 50, 150)
    # return edges

    return template


def match_and_highlight(frame, template, label, color=(0, 255, 0), threshold=0.7):
    """Improved template matching with multiple methods"""
    if template is None:
        return []

    # Ensure both images have the same type
    frame = frame.astype(np.uint8)
    template = template.astype(np.uint8)

    # Try multiple matching methods for better detection
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
    best_locations = []

    for method in methods:
        result = cv2.matchTemplate(frame, template, method)

        # Adjust threshold based on method
        if method == cv2.TM_CCORR_NORMED:
            threshold = 0.9  # Higher threshold for correlation

        loc = np.where(result >= threshold)

        # Convert to color for visualization
        if len(frame.shape) == 2:  # If grayscale
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_color = frame.copy()

        detection_locations = []
        for pt in zip(*loc[::-1]):
            h, w = template.shape
            cv2.rectangle(frame_color, pt, (pt[0] + w, pt[1] + h), color, 2)
            cv2.putText(frame_color, f"{label}_{method}", (pt[0], pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            detection_locations.append(pt)

        if detection_locations:
            best_locations.extend(detection_locations)

    return best_locations


def enhanced_cactus_detection(frame, templates, jump_zone_x=(200, 322)):
    """Enhanced detection with better preprocessing"""

    # Apply preprocessing to the frame
    processed_frame = cv2.GaussianBlur(frame, (3, 3), 0)

    # Optional: Apply histogram equalization for better contrast
    processed_frame = cv2.equalizeHist(processed_frame)

    detections = {}

    for name, template in templates.items():
        if template is None:
            continue

        # Use multiple scales for better detection
        scales = [0.8, 1.0, 1.2]

        for scale in scales:
            if scale != 1.0:
                h, w = template.shape
                scaled_template = cv2.resize(
                    template, (int(w * scale), int(h * scale)))
            else:
                scaled_template = template

            # Lower threshold for initial detection
            threshold = 0.6 if "cactus" in name else 0.7
            color = (0, 0, 255) if "cactus" in name else (0, 255, 0)

            locations = match_and_highlight(
                processed_frame, scaled_template,
                label=f"{name}_{scale:.1f}", color=color, threshold=threshold
            )

            if locations:
                detections[f"{name}_{scale}"] = locations

                # Check if cactus is in jump zone
                if "cactus" in name:
                    for loc in locations:
                        if jump_zone_x[0] < loc[0] < jump_zone_x[1]:
                            return True, detections

    return False, detections


# Load templates with preprocessing
templates = {}
template_files = [
    "dragon.png", "cactus1.png", "cactus2.png",
    "cactus3.png", "cactus4.png", "cactus5.png", "cactus6.png"
]

for file in template_files:
    name = file.replace('.png', '')
    templates[name] = preprocess_template(file)

print("Starting Dino Game Bot...")
print("Press 'q' in the game window to quit")

# Add delay to prevent multiple jumps
last_jump_time = 0
jump_cooldown = 0.5  # seconds

while True:
    try:
        # Capture specific game region
        screenshot = pyautogui.screenshot(region=(0, 100, 900, 300))
        img = np.array(screenshot)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Enhanced detection
        should_jump, detections = enhanced_cactus_detection(gray, templates)

        # Jump logic with cooldown
        current_time = time.time()
        if should_jump and (current_time - last_jump_time) > jump_cooldown:
            print("JUMPING! Cactus detected in danger zone")
            pyautogui.press("space")
            last_jump_time = current_time

        # Display debug information
        if detections:
            for name, locations in detections.items():
                for loc in locations:
                    print(f"Detected {name} at position: {loc}")

        # Show the processed frame
        display_frame = cv2.resize(gray, (950, 350))
        cv2.imshow("Dino Detection Feed", display_frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

cv2.destroyAllWindows()
print("Game bot stopped.")
