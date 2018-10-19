import warnings
import cv2
import unicodedata
import model
import utils
import data
from align import AlignDlib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import dlib

import face_recognition
from sklearn.externals import joblib
import pickle

# Load face embedding model
nn4_small2_pretrained = model.create_model()
nn4_small2_pretrained.load_weights('nn4.small2.v1.h5')

# Load face recognition model
face_rec_model = joblib.load('pretrain_model/bp_svc.joblib')
#Load encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy')
#enc = pickle.loads(open( "encoder.pkl", "rb" ).read())
# Load dlib face alignment
alignment = AlignDlib('landmarks.dat')

rectangleColor = (153, 51, 255)

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def stamp_label(point, text, image):
    right, y = point

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    margin = 5
    thickness = 2
    color = (0, 0, 0)

    size = cv2.getTextSize(text, font, font_scale, thickness)

    text_width = size[0][0]
    text_height = size[0][1]
    line_height = text_height + size[1] + margin

    _x = image.shape[1] - margin - text_width
    _y = margin + size[0][1]

    image = cv2.rectangle(image,(right, y),(right - text_width, y - text_height),rectangleColor, thickness=-1)
    cv2.putText(image, text, (right - text_width, y), font, font_scale, color, thickness)

    return image

def predict(model, frame):
    warnings.filterwarnings('ignore')
    boxes = face_recognition.face_locations(frame)
    res = []
    for i, c in enumerate(boxes):
        y = c[0]
        x = c[3]
        bottom = c[2]
        right = c[1]


        crop_img = frame[y: bottom, x: right]
        img = align_image(crop_img)
        if img is None or isinstance(img, float):
            continue
        img = (img / 255.).astype(np.float32)
        embedded = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
        pred = model.predict([embedded])
        enc = encoder.inverse_transform(pred)[0]
        name = enc
        if name == 'Rosé':name = 'Rose'

        result = {
            "name": name,
            "x": x,
            "y": y,
            "bottom": bottom,
            "right": right
        }

        res.append(result)
    return res

def recogize_faces(model, frame):
    result = predict(model, frame)

    image = frame.copy()

    for i in result:
        x = i['x']
        y = i['y']
        bottom = i['bottom']
        right = i['right']
        image = cv2.rectangle(image,(right, bottom),(x , y),rectangleColor, thickness=3)
        image = stamp_label((right, y), i['name'], image)

    return image


def track_face(model, frame):
  # Numerical encoding of identities
  trackingFace = 0
  #print(len(boxes))
  #top, right, bottom, left
  tracker = dlib.correlation_tracker()

  name = ''

  resultImage = frame.copy()
  if not trackingFace:
      result = predict(model, frame)
      for i in result:
          if i['name'] == 'Jennie':
              x = i['x']
              y = i['y']
              w = i['right'] - i['x']
              h = i['bottom'] - i['y']
              tracker.start_track(frame,dlib.rectangle( x-10, y-20, x+w+10, y+h+20))
              trackingFace = 1
              #track_face(image, x, y, right - x, bottom - y)

  if trackingFace:
      trackingQuality = tracker.update( frame )

      quality = 8.75
      if trackingQuality >= quality:
          tracked_position =  tracker.get_position()

          x = int(tracked_position.left())
          y = int(tracked_position.top())
          w = int(tracked_position.width())
          h = int(tracked_position.height())

          resultImage = cv2.rectangle(resultImage, (x, y), (x + w , y + h), rectangleColor ,2)
          resultImage = stamp_label((x+w, y), 'Tracked Jennie', resultImage)
          #cv2.imshow('frame',resultImage)
          print("Tracking")
      else:
          print("Cannot Track !!!!")
          trackingFace = 0




      #name = unicodedata.normalize('NFKD', enc)

      #

      #print(name)

  return resultImage


cap = cv2.VideoCapture('bp_live.mp4')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
if cap.isOpened():
    # get vcap property
    width = cap.get(3)   # float
    height = cap.get(4)

    out = cv2.VideoWriter('output_track.mp4', -1, 20.0, (int(width/2), int(height/2)))

while(cap.isOpened()):
    ret, frame = cap.read()
    (w, h) = (int(frame.shape[1]/2), int(frame.shape[0]/2))

    if ret == True:
        resized_image = cv2.resize(frame, (w, h), 0, 0, cv2.INTER_CUBIC);
        #image = predict(face_rec_model, enc, resized_image)
        image = track_face(face_rec_model, resized_image)
        #image = recogize_faces(face_rec_model, resized_image)
        #cv2.imshow('frame',image)
        out.write(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
