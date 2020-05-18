import time
import torchaudio
import pandas as pd
import traceback
import transform as T
import numpy as np
import torchvision
import torch
print(torch.__version__)
print(torchvision.__version__)

normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989])
transform_video = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128, 171)),
    T.RandomHorizontalFlip(),
    normalize,
    T.RandomCrop((112, 112))
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torchaudio.__version__)


def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()


class LIRIS_ACCEDE(torch.utils.data.Dataset):
    def __init__(self, settype: str):
        super(LIRIS_ACCEDE).__init__()
        self.max_length = 200
        self.max_audio_length = 400000
        annotation_directory_path = '/root/yangsen-data/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/'
        sets_file_path = annotation_directory_path + 'ACCEDEsets.txt'
        sets_df = pd.read_csv(sets_file_path, sep='\t')
        if settype == 'train':
            set_id = 1
        elif settype == 'validation':
            set_id = 2
        elif settype == 'test':
            set_id = 0
        else:
            assert(false)
        self.videos_name = sets_df[sets_df['set'] == set_id]['name']
        self.data_path = '/root/yangsen-data/LIRIS-ACCEDE-data/data/'
        label_file = '/root/yangsen-data/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/ACCEDEranking.txt'
        self.label_df = pd.read_csv(label_file, sep='\t')
        self.mfcc_transformer = torchaudio.transforms.MFCC()

    def __len__(self):
        return len(self.videos_name)

    def uniform_rescale(self, x):
        return (x / 9800) * 2 - 1

    def getAffective(self, name, column='Value'):
        if column == 'Value':
            return np.array([self.label_df[self.label_df['name'] == name]['valenceValue'].iloc[0],
                             self.label_df[self.label_df['name'] == name]['arousalValue'].iloc[0]])
        elif column == 'Rank':
            return np.array([self.uniform_rescale(self.label_df[self.label_df['name'] == name]['valenceRank'].iloc[0]),
                             self.uniform_rescale(self.label_df[self.label_df['name'] == name]['arousalRank'].iloc[0])])
        else:
            raise Exception('need  column')

    def __getitem__(self, idx):
        name = self.videos_name.iloc[idx]
        data = torchvision.io.read_video(self.data_path + name)
        sample_rate = data[2]['audio_fps']
        audio = data[1]
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
        transformed_video = transform_video(data[0])
        if transformed_video.shape[1] > self.max_length:
            transformed_video = transformed_video[:, :self.max_length, :, :]
        elif self.max_length > transformed_video.shape[1]:
            last_frame = transformed_video[:, -1:, :, :]
            frames = last_frame.repeat(
                1, self.max_length-transformed_video.shape[1], 1, 1)
            transformed_video = torch.cat([transformed_video, frames], 1)
#         print(transformed_video.shape)

        return {'name': name,
                'video': transformed_video,
                'audio': self.mfcc_transformer(audio),
                'label': self.getAffective(name, column='Rank')
                }


# dataset
trainset = LIRIS_ACCEDE(settype='train')
test_data_set = LIRIS_ACCEDE(settype='test')
validateset = LIRIS_ACCEDE(settype='validation')
train_data_set = torch.utils.data.ConcatDataset([trainset, validateset])
batch_size = 8
train_dataloader = torch.utils.data.DataLoader(
    train_data_set, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(
    test_data_set, batch_size=batch_size)
# video_model
video_model = torchvision.models.video.r3d_18(pretrained=False, progress=True)
num_ftrs = video_model.fc.in_features
video_model.fc = torch.nn.Linear(num_ftrs, 2)
video_model = video_model.to(device)
# audio_model
audio_model = torchvision.models.resnet152(pretrained=True)
num_ftrs = audio_model.fc.in_features
audio_model.fc = torch.nn.Linear(num_ftrs, 2)
audio_model.conv1 = torch.nn.Conv2d(2, 64, (2, 10))
audio_model = audio_model.to(device)


def trainProcess(model, modal='video'):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            try:
                video_inputs = data[modal].to(device)
                labels = torch.tensor(data['label']).float().to(device)

                optimizer.zero_grad()

                outputs = model(video_inputs)
    #             print(outputs.shape, labels.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 5 == 4:
                    print('[%d. %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 5))
                    running_loss = 0.0
            except Exception as e:
                traceback.print_exc()

        evalProcess(model)


def evalProcess(model, medal='video'):
    criterion = torch.nn.MSELoss()
    arousal_loss_test = 0.0
    valence_loss_test = 0.0
    mse_list = []
    for i, data in enumerate(test_dataloader, 0):
        try:
            # get the inputs
            video_inputs = data[modal].to(device)
            valence = data['label'][:, 0:1]
            arousal = data['label'][:, 1:2]
            ground_truth = data['label']
            inputs, valence, arousal, ground_truth = inputs.to(device), valence.to(
                device), arousal.to(device), ground_truth.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                valence_loss = criterion(outputs[:, 0:1], valence)
                arousal_loss = criterion(outputs[:, 1:2], arousal)
                arousal_loss_test += arousal_loss.item()
                valence_loss_test += valence_loss.item()
        except Exception as e:
            traceback.print_exc()

    print(f'arousal mse average: {arousal_loss_test / len(test_dataloader)}')
    print(f'valence mse average: {valence_loss_test / len(test_dataloader)}')
    mse_list.append((arousal_loss_test / len(test_dataloader),
                     valence_loss_test / len(test_dataloader)))

# import time
# t1 = time.time()
# trainProcess(video_model, 'video')
# t2 = time.time()
# print (f'video train: {t2 - t1}')


time1 = time.time()
trainProcess(audio_model, modal='audio')
torch.save(audio_model.state_dict(), 'audio_model.model')
time2 = time.time()
print(f'audio train: {time2 - time1}s')
time3 = time.time()
evalProcess(audio_model)
print(f'audio eval: {time3 - time2}s')

