import os
import numpy as np
import cv2
from pytube import YouTube

def download_youtube_video(url, resolution='360p', path='./'):
    yt = YouTube(url)
    return yt.streams.filter(file_extension="mp4").get_by_resolution(resolution).download(path)

def extract_customized_frame(video_path, save_path, interval=None, position=None, resize=None):
  video = cv2.VideoCapture(video_path)
  length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  if not interval:
    interval = range(0, length)
  for i in interval:  
    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    stt, frame = video.read()
    if not stt:
      raise Exception('frame_read_error')
    if position:
      (x, y, w, h) = position[i - interval[0]]
    cv2_imshow(frame)
    cv2_imshow(frame[y:y + h + 1, x:x + w + 1])
    cv2_imshow(frame[x:x + w + 1, y:y + h + 1])
    cv2.imwrite(os.path.join(save_path, 'frame{}.jpg'.format(i - interval[0])), frame)