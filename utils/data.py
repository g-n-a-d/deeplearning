import os
from shutil import rmtree
import numpy as np
import torch
import cv2
from pytube import YouTube

def partition_sampler(df, size):
    indexes = np.random.choice(df.shape[0], size, replace=True)
    return df.iloc[indexes, :]

def download_youtube_video(url, resolution='360p', save_path_video='./videos'):
    yt = YouTube(url)
    return yt.streams.filter(file_extension="mp4").get_by_resolution(resolution).download(save_path_video)

def extract_customized_frame(video_path, interval, bounding_box, size_init=(720, 1280), size_out=(256, 256)):
    video = cv2.VideoCapture(video_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_out = []
    for i in range(*interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        stt, frame = video.read()
        if not stt:
            raise Exception('frame_read_error')
        (left, top, right, bot) = bounding_box
        left = int(left*width/size_init[1])
        top = int(top*height/size_init[0])
        right = int(right*width/size_init[1])
        bot = int(bot*height/size_init[0])
        frame_out.append(cv2.resize(frame[top:bot, left:right], dsize=size_out, interpolation=cv2.INTER_CUBIC))
    return frame_out

def load_data(df, num_sample=10, resolution='360p', size_out=(256, 256), save_path_video='./videos'):
    os.mkdir(save_path_video)
    data_raw = []
    for i in range(df.shape[0]):
        path_video = download_youtube_video('https://www.youtube.com/watch?v=' + df['video_id'].values[i], resolution, save_path_video)
        frame_sampled = extract_customized_frame(path_video, (df['start'].values[i], df['end'].values[i]), list(map(int, df['bbox'].values[i].split('-'))), (df['height'].values[i], df['width'].values[i]), size_out)
        for ii in range(num_sample):
            index_sampled = np.sort(np.random.choice(range(df['end'].values[i] - df['start'].values[i]), 2, replace=False))
            data_raw.append(np.concatenate((np.expand_dims(frame_sampled[index_sampled[0]], axis=0), np.expand_dims(frame_sampled[index_sampled[1]], axis=0)), axis=0))
    rmtree(save_path_video)
    return np.array(data_raw)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_raw, device=torch.device('cuda')):
        self.data_raw = torch.tensor(data_raw, dtype=torch.float)
        self.frame_source = torch.cat((self.data_raw[:, 0, :, :, 0].unsqueeze(-3), self.data_raw[:, 0, :, :, 0].unsqueeze(-3), self.data_raw[:, 0, :, :, 0].unsqueeze(-3)), dim=-3)
        self.frame_driving = torch.cat((self.data_raw[:, 1, :, :, 0].unsqueeze(-3), self.data_raw[:, 1, :, :, 0].unsqueeze(-3), self.data_raw[:, 1, :, :, 0].unsqueeze(-3)), dim=-3)
        self.frame_source = self.frame_source.to(device)
        self.frame_driving = self.frame_driving.to(device)

    def __len__(self):
        return len(self.data_raw)

    def __getitem__(self, index):
        return self.frame_source[index], self.frame_driving[index]