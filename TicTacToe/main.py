import time
import math
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

box_centers = {0:(240, 160), 1:(440, 160), 2:(640, 160), 3:(240, 360), 4:(440, 360), 5:(640, 360), 6:(240, 560), 7:(440, 560), 8:(640, 560)}

def check_end(circles, crosses):
  ends = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
  for end in ends:
    if (all(x in circles for x in end)):
      print('Circle player wins')
      return 1
    elif (all(x in crosses for x in end)):
      print('Cross player wins')
      return 2
  return 0


def add_winner(image, winner):
  if winner == 1: # Circle wins
    image = cv2.putText(image, 'Circle Player wins', org = (800, 200), color = (0, 0, 255), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, thickness = 2)
  elif winner == 2: # Cross wins
    image = cv2.putText(image, 'Cross Player wins', org = (800, 200), color = (0, 255, 0), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, thickness = 2)
  
  return image

def draw_shape(image, it, box_num):
  # box_num = get_box_number(coords)
  center = box_centers[box_num]
  if it == 0: # circle turn
    cv2.circle(image, center, 80, (0, 0, 255), 2)
  else: # draw cross
    cv2.line(image, (center[0] - 80, center[1] - 80), (center[0] + 80, center[1] + 80), (0, 255, 0), 2)
    cv2.line(image, (center[0] + 80, center[1] - 80), (center[0] - 80, center[1] + 80), (0, 255, 0), 2)
  return image

def draw_all_shapes(image, circles, crosses):
  if circles:
    for c in circles:
      image = draw_shape(image, 0, c)
  if crosses:
    for c in crosses:
      image = draw_shape(image, 1, c)
  return image

def get_box_number(coords):
  if coords[0] >= 140 and coords[0] < 340:
    if coords[1] >= 60 and coords[1] < 260:
      return 0
    elif coords[1] >= 260 and coords[1] < 460:
      return 3
    elif coords[1] >= 460 and coords[1] < 660:
      return 6
  elif coords[0] >= 340 and coords[0] < 540:
    if coords[1] >= 60 and coords[1] < 260:
      return 1
    elif coords[1] >= 260 and coords[1] < 460:
      return 4
    elif coords[1] >= 460 and coords[1] < 660:
      return 7
  elif coords[0] >= 540 and coords[0] < 740:
    if coords[1] >= 60 and coords[1] < 260:
      return 2
    elif coords[1] >= 260 and coords[1] < 460:
      return 5
    elif coords[1] >= 460 and coords[1] < 660:
      return 8
  else:
    return -1

# For webcam input:
cap = cv2.VideoCapture(0)
click_time = 0
done_game = False
end_at = math.inf
circles = []
crosses = []
it = 0  # circle start first
boxes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.rectangle(image, (140, 60), (740, 660), (255, 0, 0), 2)
    cv2.line(image, (340, 60), (340, 660), (255, 0, 0), 2)
    cv2.line(image, (540, 60), (540, 660), (255, 0, 0), 2)
    cv2.line(image, (140, 260), (740, 260), (255, 0, 0), 2)
    cv2.line(image, (140, 460), (740, 460), (255, 0, 0), 2)

    image = draw_all_shapes(image, circles, crosses)
    image = add_winner(image, done_game)
    if not(done_game):
      done_game = check_end(circles, crosses)
      end_at = time.time() + 2

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if not done_game:
          if abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z) > 0.1 and time.time() - click_time > 1:
            coords = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 1280, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 720)
            box_number = get_box_number(coords)
            if box_number != -1:
              if box_number not in boxes:
                continue
              if len(boxes) == 1:
                image = draw_shape(image, it, box_number)
                if not (done_game):
                  done_game = check_end(circles, crosses)
                  end_at = time.time() + 2

              if it == 0:
                circles.append(box_number)
              else:
                crosses.append(box_number)
              it = 1 - it
              boxes.remove(box_number)
        else:
          continue
          # click_time = time.time()
        

    if time.time() >= end_at:
      break
    if not(done_game):
      done_game = check_end(circles, crosses)
      end_at = time.time() + 2
    cv2.imshow('MediaPipe Hands', image)
    if boxes == [] and not(done_game):
      image = add_winner(image, done_game)
      print('Game Over !!!')
      end_at = time.time() + 2
      done_game = 3
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()