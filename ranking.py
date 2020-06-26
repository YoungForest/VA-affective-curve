import time
import torchaudio
import pandas as pd
import traceback
import transform as T
import numpy as np
import torchvision
import torch
from tensorboardX import SummaryWriter
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
trainIter = 0
evalIter = 0

def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()


class LIRIS_ACCEDE(torch.utils.data.Dataset):
    def __init__(self, settype: str):
        super(LIRIS_ACCEDE).__init__()
        self.max_length = 50
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
            assert(False)
        self.videos_name = sets_df[sets_df['set'] == set_id]['name']
        self.data_path = '/root/yangsen-data/LIRIS-ACCEDE-data/data/'
        label_file = '/root/yangsen-data/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/ACCEDEranking.txt'
        self.label_df = pd.read_csv(label_file, sep='\t')
        self.mfcc_transformer = torchaudio.transforms.MFCC().to(device)

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
        # vedio
        transformed_video = transform_video(data[0][::4]).to(device)
        if transformed_video.shape[1] > self.max_length:
            transformed_video = transformed_video[:, :self.max_length, :, :]
        elif self.max_length > transformed_video.shape[1]:
            last_frame = transformed_video[:, -1:, :, :]
            frames = last_frame.repeat(
                1, self.max_length-transformed_video.shape[1], 1, 1)
            transformed_video = torch.cat([transformed_video, frames], 1)
#         print(transformed_video.shape)
        del data
        return {'name': name,
                'video': transformed_video,
                'audio': self.mfcc_transformer(audio),
                'label': self.getAffective(name, column='Value')
                }


# dataset
trainset = LIRIS_ACCEDE(settype='train')
test_data_set = LIRIS_ACCEDE(settype='test')
validateset = LIRIS_ACCEDE(settype='validation')
train_data_set = torch.utils.data.ConcatDataset([trainset, validateset])
batch_size = 16
train_dataloader = torch.utils.data.DataLoader(
    train_data_set, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(
    test_data_set, batch_size=batch_size)
# video_model
video_model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
num_ftrs = video_model.fc.in_features
video_model.fc = torch.nn.Linear(num_ftrs, 2)
video_model = video_model.to(device)
# audio_model
audio_model = torchvision.models.resnet18(pretrained=True)
num_ftrs = audio_model.fc.in_features
audio_model.fc = torch.nn.Linear(num_ftrs, 2)
audio_model.conv1 = torch.nn.Conv2d(2, 64, (2, 10))
audio_model = audio_model.to(device)

class FusionNetwork(torch.nn.Module):
    def __init__(self):
        super(FusionNetwork, self).__init__()
        video_model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        num_ftrs = video_model.fc.in_features
        video_model.fc = torch.nn.Linear(num_ftrs, 128)
        self.video = video_model
        audio_model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = audio_model.fc.in_features
        audio_model.fc = torch.nn.Linear(num_ftrs, 128)
        audio_model.conv1 = torch.nn.Conv2d(2, 64, (2, 10))
        self.audio = audio_model
        self.fusion_fc = torch.nn.Linear(128*2, 2)
    def forward(self, x):
        self.video_feature = self.video(x['video'])
        self.audio_feature = self.audio(x['audio'])
        return self.fusion_fc(torch.cat([self.video_feature, self.audio_feature], dim=1))

print(f'train_dataloader length: {len(train_dataloader)}')


def trainProcess(model, modal='video'):
    global trainIter
    model.train()
    criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    for epoch in range(50):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            labels = data['label'].clone().detach().float().to(device)
            optimizer.zero_grad()

            if modal == 'fusion':
                outputs = model(data)
            else:
                outputs = model(data[modal])
#             print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:
                print('[%d. %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 5))
                writer.add_scalar(f'data/train-{mode}', running_loss / 5, trainIter)
                trainIter += 1
                running_loss = 0.0

        evalProcess(model, modal=modal)


def evalProcess(model, modal='video'):
    global evalIter
    model.eval()
    criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    arousal_loss_test = 0.0
    valence_loss_test = 0.0
    mse_list = []
    for i, data in enumerate(test_dataloader, 0):
        # get the inputs
        valence = data['label'][:, 0:1].to(device)
        arousal = data['label'][:, 1:2].to(device)
        ground_truth = data['label']
        with torch.no_grad():
            if modal == 'fusion':
                outputs = model(data)
            else:
                outputs = model(data[modal])
            valence_loss = criterion(outputs[:, 0:1], valence)
            arousal_loss = criterion(outputs[:, 1:2], arousal)
            arousal_loss_test += arousal_loss.item()
            valence_loss_test += valence_loss.item()

    print(f'arousal mse average: {arousal_loss_test / len(test_dataloader)}')
    print(f'valence mse average: {valence_loss_test / len(test_dataloader)}')
    writer.add_scalar(f'data/testArousal-{mode}', arousal_loss_test / len(test_dataloader), evalIter)
    writer.add_scalar(f'data/testValence-{mode}', valence_loss_test / len(test_dataloader), evalIter)
    evalIter += 1
    mse_list.append((arousal_loss_test / len(test_dataloader),
                     valence_loss_test / len(test_dataloader)))

import time
import sys

if __name__ == '__main__':
    assert(len(sys.argv) >= 2)
    mode = sys.argv[1]
    writer = SummaryWriter()
    if mode == 'video':
        t1 = time.time()
        # video_model.load_state_dict(torch.load('video_model.model'))
        # video_model.eval()
        trainProcess(video_model, 'video')
        torch.save(video_model.state_dict(), 'video_model.model')
        t2 = time.time()
        print (f'video train: {t2 - t1}')
    elif mode == 'audio':
        time1 = time.time()
        trainProcess(audio_model, modal='audio')
        torch.save(audio_model.state_dict(), 'audio_model.model')
        time2 = time.time()
        print(f'audio train: {time2 - time1}s')
    elif mode == 'fusion':
        fusion_model = FusionNetwork().to(device)
        time1 = time.time()
        trainProcess(fusion_model, modal='fusion')
        torch.save(fusion_model.state_dict(), 'fusion_model.model')
        time2 = time.time()
        print(f'fusion train: {time2 - time1}s')
    else:
        assert(False)
    # export scalar data to JSON for external processing
    writer.export_scalars_to_json(f'./{mode}_scalars.json')
    writer.close()
