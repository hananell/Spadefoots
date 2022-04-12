import cv2


# detect movements, that is a difference, between tw0 given frames
def detect_movements(frame1, frame2):
    frame1, frame2 = cv2.imread(frame1), cv2.imread(frame2)
    if frame1.shape != frame2.shape:
        print(f"different shapes  {frame1}  {frame2}")
        return []

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    move_rects = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 500:
            continue
        move_rects.append((x, y, x + w, y + h))
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    return move_rects
