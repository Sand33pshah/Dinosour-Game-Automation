import pyautogui
import cv2
import numpy as np


# IMAGE
# screenshot = pyautogui.screenshot()
# img = np.array(screenshot)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Game Segment", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def match_and_highlight(frame, template, label, color=(0, 255, 0), threshold=0.9):

    # added to ensure both image have the same type
    frame = frame.astype(np.uint8)
    template = template.astype(np.uint8)

    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)
    # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    # return loc if len(loc[0]) > 0 else None

    detection_locations = []

    for pt in zip(*loc[::-1]):
        h, w = template.shape
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1]+h), color, 2)
        cv2.putText(frame, label, (pt[0], pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        detection_locations.append(pt)
    return detection_locations


templates = {
    # 0 tells that this is the grayscale image
    "dragon": cv2.imread("dragon.png", 0),
    "cactus_1": cv2.imread("cactus1.png", 0),
    "cactus_2": cv2.imread("cactus2.png", 0),
    "cactus_3": cv2.imread("cactus3.png", 0),
    "cactus_4": cv2.imread("cactus4.png", 0),
    "cactus_5": cv2.imread("cactus5.png", 0),
    "cactus_6": cv2.imread("cactus6.png", 0),


}

while True:
    screenshot = pyautogui.screenshot(region=(0, 100, 900, 300))
    img = np.array(screenshot)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for name, template in templates.items():

        template = cv2.GaussianBlur(
            template, (3, 3), 0)   # added to process image

        color = (0, 0, 255) if "cactus" in name else (0, 255, 0)
        locations = match_and_highlight(
            gray, template, label=name, color=color)
        for loc in locations:
            # print(f"{name} detected at {loc}")
            if "cactus" in name and loc[0] > 200 and loc[0] < 322:
                print(f"Loc: {loc} Horizontal position: {loc[0]}")
                pyautogui.press("space")

    resized_img = cv2.resize(gray, (950, 350))
    cv2.imshow("Dino Detection Feed", resized_img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()


'''
ERRORs

1.  result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
cv2.error: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\templmatch.cpp:1164: error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type() && _img.dims() <= 2 in function 'cv::matchTemplate

Both the feed and template should be in same scale either gray or 3 channel RGB




Dragon position is always fixed i.e. (175, 79)

'''
