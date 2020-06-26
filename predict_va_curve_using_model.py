import sys
import time
import torchaudio
import pandas as pd
import xml.etree.ElementTree as ET
import ranking
import transform as T
import numpy as np
import torchvision
import torch
print(torch.__version__)
print(torchvision.__version__)

normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989])
# def normalize(tensor):
#     # Subtract the mean, and scale to the interval [-1,1]
#     tensor_minusmean = tensor - tensor.mean()
#     return tensor_minusmean/tensor_minusmean.abs().max()
transform_video = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128, 171)),
    T.RandomHorizontalFlip(),
    normalize,
    T.RandomCrop((112, 112))
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


root = ET.parse(
    '/root/yangsen-data/LIRIS-ACCEDE-movies/ACCEDEmovies.xml').getroot()
movie_length = {}


def get_sec(time_str: str) -> int:
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


for i in root:
    name = i.find('movie').text
    length = get_sec(i.find('length').text)
    movie_length[name] = length

# fpsMovie = [['After_The_Rain',23.976]]
fpsMovie = [['After_The_Rain', 23.976],
            ['Attitude_Matters', 29.97],
            ['Barely_legal_stories', 23.976],
            ['Between_Viewings', 25],
            ['Big_Buck_Bunny', 24],
            ['Chatter', 24],
            ['Cloudland', 25],
            ['Damaged_Kung_Fu', 25],
            ['Decay', 23.976],
            ['Elephant_s_Dream', 24],
            ['First_Bite', 25],
            ['Full_Service', 29.97],
            ['Islands', 23.976],
            ['Lesson_Learned', 29.97],
            ['Norm', 25],
            ['Nuclear_Family', 23.976],
            ['On_time', 30],
            ['Origami', 24],
            ['Parafundit', 24],
            ['Payload', 25],
            ['Riding_The_Rails', 23.976],
            ['Sintel', 24],
            ['Spaceman', 23.976],
            ['Superhero', 29.97],
            ['Tears_of_Steel', 24],
            ['The_room_of_franz_kafka', 29.786],
            ['The_secret_number', 23.976],
            ['To_Claire_From_Sonny', 23.976],
            ['Wanted', 25],
            ['You_Again', 29.97]]

print(torchaudio.__version__)
path_prefix = '/root/yangsen-data/LIRIS-ACCEDE-movies/movies/'


class MOVIE(torch.utils.data.Dataset):
    def __init__(self, name: str):
        super(MOVIE).__init__()
        self.name = name
        self.max_length = 200
        self.max_audio_length = 400000
        self.mfcc_transformer = torchaudio.transforms.MFCC()
        self.seconds_length = movie_length[name]

    def __len__(self):
        return self.seconds_length // 8

    def __getitem__(self, idx: int):
        data = torchvision.io.read_video(
            path_prefix + name, start_pts=idx * 8, end_pts=idx * 8 + 8, pts_unit='sec')
        transformed_video = transform_video(data[0])
        if transformed_video.shape[1] > self.max_length:
            transformed_video = transformed_video[:, :self.max_length, :, :]
        elif self.max_length > transformed_video.shape[1]:
            last_frame = transformed_video[:, -1:, :, :]
            frames = last_frame.repeat(
                1, self.max_length-transformed_video.shape[1], 1, 1)
            transformed_video = torch.cat([transformed_video, frames], 1)
#         print(transformed_video.shape)
        # audio
        sample_rate = data[2]['audio_fps']
        audio = data[1].to(device)
        downsample_rate = 2000
        downsample_resample = torchaudio.transforms.Resample(
            sample_rate, downsample_rate, resampling_method='sinc_interpolation')
        audio = normalize(audio)
        if audio.shape[0] > 2:
            audio = audio[0:2, :]
        elif audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        if audio.shape[1] > self.max_audio_length:
            audio = audio[:, :self.max_audio_length]
        elif audio.shape[1] < self.max_audio_length:
            last_frame = audio[:, -1:]
            frames = last_frame.repeat(
                1, self.max_audio_length - audio.shape[1])
            audio = torch.cat([audio, frames], 1)

        return {'name': self.name,
                'start': idx*8,
                'video': transformed_video,
                'audio': self.mfcc_transformer(audio), }


if __name__ == '__main__':
    assert(len(sys.argv) >= 2)
    mode = sys.argv[1]
    time1 = time.time()
    if mode == 'video':
        model = torchvision.models.video.r3d_18(
            pretrained=False, progress=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load('video_model.model'))
    elif mode == 'audio':
        # audio_model
        audio_model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = audio_model.fc.in_features
        audio_model.fc = torch.nn.Linear(num_ftrs, 2)
        audio_model.conv1 = torch.nn.Conv2d(2, 64, (2, 10))
        model.load_state_dict(torch.load('audio_model.model'))
        model = audio_model

    elif mode == 'fusion':
        fusion_model = FusionNetwork().to(device)
        fusion_model.load_state_dict(torch.load('fusion_model.model'))
        model = fusion_model
    else:
        assert(False)
    model.eval()
    model = model.to(device)

    VA = {}
    for i in fpsMovie:
        name = i[0]+'.mp4'
        dataset = MOVIE(name)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        line = torch.tensor([]).to(device)
        for i, data in enumerate(dataloader, 0):
            if mode == 'fusion':
                inputs = data
            else:
                inputs = data['mode']
            for i in inputs.shape:
                if i == 0:
                    continue
            with torch.set_grad_enabled(False):
                # print (inputs.shape)
                outputs = model(inputs)
                line = torch.cat([line, outputs])
        VA[name] = line.cpu().numpy()

    VA_list = {}
    for i in VA:
        length = VA[i].shape[0]
        valence = []
        arousal = []
        for j in range(length):
            valence.append(float(VA[i][j][0]))
            arousal.append(float(VA[i][j][1]))
        VA_list[i] = [valence, arousal]

    print(f'answer: {VA_list}')
    import json
    with open(f'predictVA-{mode}.json', 'w') as f:
        json.dump(VA_list, f)
    time2 = time.time()
    print(f'mode predict: {time2 - time1}s')
