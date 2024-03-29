import os
import argparse
import pickle
from multiprocessing import Pool
from random import shuffle
import math
import numpy as np
from pyedflib import highlevel
from scipy.signal import stft
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--sample_rate', type=int, default=250)
parser.add_argument('--data_directory', type=str, default='/.../TUH/tuh_eeg_seizure/v1.5.4') # Change it to your TUH EDF data files
parser.add_argument('--save_directory', type=str,
                    default='/.../Data/pckl') # Change it to where you want the pckl to be stored
parser.add_argument('--label_type', type=str, default='csv_bi')
parser.add_argument('--cpu_num', type=int, default=32)
parser.add_argument('--data_type', type=str, default='train', choices=['train', 'eval', 'dev'])
parser.add_argument('--task_type', type=str, default='binary', choices=['binary'])
parser.add_argument('--slice_length', type=int, default=12)
parser.add_argument('--eeg_type', type=str, default='stft', choices=['original', 'bipolar', 'stft'])
args = parser.parse_args()

GLOBAL_INFO = {}


def search_walk(info):
    searched_list = []
    root_path = info.get('path')
    extensions = info.get('extensions')

    for (path, dir, files) in os.walk(root_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext in extensions:
                list_file = ('%s/%s' % (path, filename))
                searched_list.append(list_file)

    return searched_list


def bipolar_signals_func(signals):
    bipolar_signals = []
    bipolar_signals.append(signals[0] - signals[4])  # fp1-f7
    bipolar_signals.append(signals[1] - signals[5])  # fp2-f8
    bipolar_signals.append(signals[4] - signals[9])  # f7-t3
    bipolar_signals.append(signals[5] - signals[10])  # f8-t4
    bipolar_signals.append(signals[9] - signals[15])  # t3-t5
    bipolar_signals.append(signals[10] - signals[16])  # t4-t6
    bipolar_signals.append(signals[15] - signals[13])  # t5-o1
    bipolar_signals.append(signals[16] - signals[14])  # t6-o2
    bipolar_signals.append(signals[9] - signals[6])  # t3-c3
    bipolar_signals.append(signals[7] - signals[10])  # c4-t4
    bipolar_signals.append(signals[6] - signals[8])  # c3-cz
    bipolar_signals.append(signals[8] - signals[7])  # cz-c4
    bipolar_signals.append(signals[0] - signals[2])  # fp1-f3
    bipolar_signals.append(signals[1] - signals[3])  # fp2-f4
    bipolar_signals.append(signals[2] - signals[6])  # f3-c3
    bipolar_signals.append(signals[3] - signals[7])  # f4-c4
    bipolar_signals.append(signals[6] - signals[11])  # c3-p3
    bipolar_signals.append(signals[7] - signals[12])  # c4-p4
    bipolar_signals.append(signals[11] - signals[13])  # p3-o1
    bipolar_signals.append(signals[12] - signals[14])  # p4-o2

    return bipolar_signals


def spectrogram_unfold_feature(signals):
    nperseg = 250
    noverlap = 50
    freq_resolution = 2
    nfft = args.sample_rate * freq_resolution
    freqs, times, spec = stft(signals, fs=args.sample_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                              boundary=None, padded=False)

    spec = spec[:, :spec.shape[1] - 1, :]
    spec = np.reshape(spec, (-1, spec.shape[2]))
    amp = (np.log(np.abs(spec) + 1e-10)).astype(np.float32)

    return freqs, times, amp


class TUHDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.file_length = len(self.file_list)
        self.transform = transform

    def __len__(self):
        return self.file_length

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'rb') as f:
            data_pkl = pickle.load(f)

            signals = np.asarray(bipolar_signals_func(data_pkl['signals']))
            # print(signals.shape)

            if args.eeg_type == 'stft':
                f, t, signals = spectrogram_unfold_feature(signals)
                # print(signals.shape)
                # exit()

            signals = self.transform(signals)
            label = data_pkl['label']
            label = 0. if label == "bckg" else 1.

            patient_id = data_pkl['patient id']
            confidence = data_pkl['confidence']

        return signals, label


def get_data_loader(batch_size):
    file_dir = {'train': os.path.join(args.save_directory, 'task-binary_datatype-train'),
                'val': os.path.join(args.save_directory, 'task-binary_datatype-eval'),
                'test': os.path.join(args.save_directory, 'task-binary_datatype-dev')}
    file_lists = {'train': {'bckg': [], 'seiz': []}, 'val': {'bckg': [], 'seiz': []}, 'test': {'bckg': [], 'seiz': []}}

    for dirname in file_dir.keys():
        filenames = os.listdir(file_dir[dirname])
        for filename in filenames:
            if 'bckg' in filename:
                file_lists[dirname]['bckg'].append(os.path.join(file_dir[dirname], filename))
            elif 'seiz' in filename:
                file_lists[dirname]['seiz'].append(os.path.join(file_dir[dirname], filename))
            else:
                print('------------------------  error  ------------------------')
                exit(-1)

    print('--------------------  file_lists  --------------------')
    for dirname in file_lists.keys():
        print('--------------------  {}'.format(dirname))
        for classname in file_lists[dirname].keys():
            print('{} num: {}'.format(classname, len(file_lists[dirname][classname])))

    train_data = file_lists['train']['bckg'] + file_lists['train']['seiz'] * \
                 int(len(file_lists['train']['bckg']) / len(file_lists['train']['seiz']))
    shuffle(train_data)
    print('len(train_data): {}'.format(len(train_data)))

    bckg_data = file_lists['val']['bckg'] + file_lists['test']['bckg']
    shuffle(bckg_data)

    seiz_data = file_lists['val']['seiz'] + file_lists['test']['seiz']
    shuffle(seiz_data)

    val_data = bckg_data[:int(len(bckg_data) / 2)] + seiz_data[:int(len(seiz_data) / 2)]
    shuffle(val_data)
    print('len(val_data): {}'.format(len(val_data)))

    test_data = bckg_data[int(len(bckg_data) / 2):] + seiz_data[int(len(seiz_data) / 2):]
    shuffle(test_data)
    print('len(test_data): {}'.format(len(test_data)))

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_data = TUHDataset(train_data, transform=train_transforms)
    val_data = TUHDataset(val_data, transform=val_transforms)
    test_data = TUHDataset(test_data, transform=test_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=math.ceil(len(val_data) / 50), shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=math.ceil(len(test_data) / 50), shuffle=False)

    return train_loader, val_loader, test_loader


def generate_lead_wise_data(edf_file):
    label_lines = open(edf_file.replace('.edf', '.' + GLOBAL_INFO['label_type']), 'r').readlines()
    assert label_lines[5].strip().startswith('channel,')
    label_lists = [line.strip().split(',') for line in label_lines[6:]]

    channels_all = []
    channels_use = []
    signal_list = []
    signal_list_ordered = []

    signals, signal_headers, header = highlevel.read_edf(edf_file.replace('/edf/', '/edf_resampled/'))
    for idx, signal in enumerate(signals):
        channel_no_ref = signal_headers[idx]['label'].strip().split("-")[0]
        channels_all.append(channel_no_ref)
        if channel_no_ref not in GLOBAL_INFO['channel_list']:
            continue

        channels_use.append(channel_no_ref)
        signal_list.append(signal)

    if not all(channel in channels_use for channel in GLOBAL_INFO['channel_list']):
        print(channels_use)
        print(GLOBAL_INFO['channel_list'])
        return

    for channel in GLOBAL_INFO['channel_list']:
        signal_list_ordered.append(signal_list[channels_use.index(channel)])

    signal_list_ordered = np.asarray(signal_list_ordered)

    for label_list in label_lists:
        start_time = float(label_list[1].strip())
        end_time = float(label_list[2].strip())
        if end_time - start_time < GLOBAL_INFO['slice_length']:
            continue

        label = label_list[3].strip()
        confidence = float(label_list[4].strip())

        for i in range(int((end_time - start_time) / GLOBAL_INFO['slice_length'])):
            slice_eeg = signal_list_ordered[:,
                        int(start_time + i * GLOBAL_INFO['slice_length']) * GLOBAL_INFO['sample_rate']:
                        int(start_time + (i + 1) * GLOBAL_INFO['slice_length']) * GLOBAL_INFO['sample_rate']]

            with open("{}/{}_label_{}_confidence_{}_index_{}.pkl".format(GLOBAL_INFO['save_directory'],
                                                                         edf_file.split('/')[-1].split('.')[0], label,
                                                                         confidence, i), 'wb') as f:
                pickle.dump({'signals': slice_eeg, 'patient id': edf_file.split('/')[-1].split('.')[0].split('_')[0],
                             'label': label, 'confidence': confidence}, f)


def run_multi_process(f, l: list, n_processes=40):
    n_processes = min(n_processes, len(l))
    print('processes num: {}'.format(n_processes))

    results = []
    pool = Pool(processes=n_processes)
    for r in tqdm(pool.imap_unordered(f, l), total=len(l), ncols=75):
        results.append(r)

    pool.close()
    pool.join()

    return results


def main(args):
    channel_list = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG C3', 'EEG C4', 'EEG CZ',
                    'EEG T3', 'EEG T4', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG T5', 'EEG T6', 'EEG PZ', 'EEG FZ']

    save_directory = "{}/task-{}_datatype-{}".format(args.save_directory, args.task_type, args.data_type)
    if os.path.isdir(save_directory):
        os.system("rm -rf {}".format(save_directory))
    os.system("mkdir -p {}".format(save_directory))

    data_directory = "{}/edf/{}".format(args.data_directory, args.data_type)

    if args.task_type == "binary":
        disease_labels = {'bckg': 0, 'seiz': 1}
    else:
        exit(-1)

    edf_list = search_walk({'path': data_directory, 'extensions': [".edf", ".EDF"]})

    GLOBAL_INFO['channel_list'] = channel_list
    GLOBAL_INFO['disease_labels'] = disease_labels
    GLOBAL_INFO['save_directory'] = save_directory
    GLOBAL_INFO['label_type'] = args.label_type
    GLOBAL_INFO['sample_rate'] = args.sample_rate
    GLOBAL_INFO['slice_length'] = args.slice_length
    GLOBAL_INFO['disease_type'] = args.disease_type

    print("Number of EDF files: ", len(edf_list))
    for i in GLOBAL_INFO:
        print("{}: {}".format(i, GLOBAL_INFO[i]))
    with open(data_directory + '/preprocess_info.pickle', 'wb') as pkl:
        pickle.dump(GLOBAL_INFO, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    run_multi_process(generate_lead_wise_data, edf_list, n_processes=args.cpu_num)


if __name__ == '__main__':
    main(args)
