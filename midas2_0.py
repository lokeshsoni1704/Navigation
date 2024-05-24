
import cv2
import torch
import numpy as np
import time
import pyttsx3

engine = pyttsx3.init("sapi5")
model_type = "MiDaS_small"
midas = torch.hub.load('intel-isl/MiDaS', model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transforms = midas_transforms.small_transform


def draw_dividers(image):
    height, width, _ = image.shape
    divider_width = width // 3
    cv2.line(image, (divider_width, 0), (divider_width, height), (0, 255, 0), 2)
    cv2.line(image, (2 * divider_width, 0), (2 * divider_width, height), (0, 255, 0), 2)
    return image


def navigate(intensities):
    threshold_intensity = 128
    max_intensity_index = np.argmax(intensities)
    if intensities[max_intensity_index] > threshold_intensity:
        directions = ["move to right", "move to left"]
        engine.say(
            f"High intensity detected in the {'left' if max_intensity_index == 0 else 'right' if max_intensity_index == 2 else 'middle'} section {directions[max_intensity_index == 2]}")
    else:
        engine.say("No significant intensity detected, staying in current section")
    engine.runAndWait()


cap = cv2.VideoCapture(0)
address = "http://100.96.32.4:8080/video"
cap.open(address)
# while cap.isOpened():
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    frame_with_dividers = draw_dividers(frame.copy())
    start = time.time()

    # Process all sections in one batch
    input_batch = torch.cat([
        transforms(
            cv2.cvtColor(frame[:, i * frame.shape[1] // 3: (i + 1) * frame.shape[1] // 3], cv2.COLOR_BGR2RGB)).to(
            device)
        for i in range(3)
    ])

    with torch.no_grad():
        predictions = midas(input_batch)
        outputs = [cv2.normalize(pred.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) for pred in predictions]
        intensities = [np.mean(output) for output in outputs]

    fps = 1 / (time.time() - start)
    print(f'FPS:',int(fps))
    for i, intensity in enumerate(intensities):
        cv2.putText(frame_with_dividers, f'{["Left", "Middle", "Right"][i]} Intensity: {intensity:.2f}',
                    (20, 120 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame with Dividers and Intensity', frame_with_dividers)
    navigate(intensities)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

