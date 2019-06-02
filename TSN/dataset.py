import torch.utils.data as data
#This dataset right now gives question, video image frames and target as output.
#Later on, we will modify Glove embedding of question instead of question itself.
#transforms will be in dictionary.
"""
We are assuming that the data is structured as
Frames
    Video ID
        All frames

Optical FLow
    Video ID
        u
            Optical frames
        v
            Optical frames

corresponding optical flow index = (frame index / 2) - 1
For label, we would need to know the softmax index.
The file path for frames / optical flow + video also reqd.

"""

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd

"""
Videorecord is basically a set of frames for us, consisting of action into which we want to classify. (One row in 
QA_Individual_Segment)
So, it has one label, and one question.
For this class, args:
    frames path
    video id
    frame indices list
    label
    question
    modality

The path is going to give us the path where the frames are located.
So, it will be frames_path + video_id.
For RGB frames, in this path itself, you will find frames.
In case for flow, we may need to modify the paths as x and y paths for u and v vectors like
    path + u, path + v.
"""
class VideoRecord(object):
    def __init__(self, frames_path, video_id, frames, label, modality, question):
        self._data = [frames_path, video_id, frames, label, modality, question]

    @property
    def path(self):
        return os.path.join(self._data[0], self.video_id)

    @property
    def num_frames(self):
        frames = self._data[2]
        return frames[1] - frames[0] + 1
    
    @property
    def start(self):
        frames = self._data[2]
        return frames[0]

    @property
    def label(self):
        return int(self._data[3])
    
    @property
    def video_id(self):
        return self._data[1]    

    @property
    def modality(self):
        return self._data[4]

    @property
    def question(self):
        return self._data[5]
    
    @property
    def new_length(self):
        if self.modality == 'RGB':
            return 1
        elif self.modality == 'Flow':
            return 5
        elif self.modality == 'RGBDiff':
            return 6


class TSNDataSet(data.Dataset):
    def __init__(self, root_path,frames_path , optical_flow_path, QA_Individual_segments, SoftmaxIndex,
                 num_segments=3, 
                 image_tmpl='frame_{:010d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.QA_Individual_segments = QA_Individual_segments
        self.SoftmaxIndex = pd.DataFrame(SoftmaxIndex)
        self.SoftmaxIndex = self._pandas_to_dictionary(self.SoftmaxIndex)
        self.frames_path = frames_path
        self.optical_flow_path = optical_flow_path
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        #calling parse function
        self._parse_list()
    
    """
    This function is addition, to convert SoftmaxIndex dataframe to dictionary.
    condition is all rows must be unique
    
    The dictionary will be used in parse function to get softmax index.
    """
    def _pandas_to_dictionary(self, df):
        dictionary = {}
        for i in range(df.shape[0]):
            answer = np.array(df['answer'])[i]
            softmax_index = np.array(df['softmax_index'])[i]
            dictionary[answer] = softmax_index
        return dictionary
        
    def _load_image(self, directory, idx, modality):
        if modality == 'RGB' or modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif modality == 'Flow':
            x_img = Image.open(os.path.join(directory, 'u',self.image_tmpl.format(idx))).convert('L')
            y_img = Image.open(os.path.join(directory, 'v',self.image_tmpl.format(idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        """
        We will make two video lists over here, one for RGB, one for flow.
        list_file can be the pandas dataframe QA_Individual_segments

        QA_Individual_segments contains question, answer, time, frames, Qtype
        frames is seperated by hyphen like 563-672.
        flow index should be max(0, int(n/2 - 1))
        """
        self.rgb_video_list = []
        self.flow_video_list = []
        for i in range(self.QA_Individual_segments.shape[0]):
            rgb_frames = np.array(self.QA_Individual_segments['frames'])[i]
            rgb_frames = rgb_frames.split('-')
            rgb_frames = [int(frame) for frame in rgb_frames]
            #compute flow frames
            flow_frames = [max(0, int(frame / 2 - 1)) for frame in rgb_frames]
            """compute label"""
            answer = np.array(self.QA_Individual_segments['answer'])[i]
            video_id = np.array(self.QA_Individual_segments['video_id'])[i]
            label = self.SoftmaxIndex[answer]
            question = np.array(self.QA_Individual_segments['question'])[i]
            """Appending to the frame and flow lists with corresponding arguments"""
            
            self.rgb_video_list.append(VideoRecord(self.frames_path, video_id, rgb_frames, label, modality = 'RGB', question = question))
            self.flow_video_list.append(VideoRecord(self.optical_flow_path, video_id, flow_frames, label, modality = 'Flow', question = question))
        #self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.QA_Individual_segments)]
        
    """
    In all the indices related function, instead of adding with 1, we will add with record.start.
    """
    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - record.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - record.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + record.start

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + record.new_length - 1:
            tick = (record.num_frames - record.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + record.start

    def _get_test_indices(self, record):

        tick = (record.num_frames - record.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + record.start

    """
    In getitem, instead of returning only 1 item, we want to return 2 items, one for RGB, one for Flow.
    So, dataloader[5] will return a dictionary.
    dataloader[5]['RGB'] will return RGB data and label
    dataloader[5]['Flow'] will return Flow data and label.
    """
    def __getitem__(self, index):
        rgb_record = self.rgb_video_list[index]
        flow_record = self.flow_video_list[index]

        if not self.test_mode:
            rgb_segment_indices = self._sample_indices(rgb_record) if self.random_shift else self._get_val_indices(rgb_record)
            flow_segment_indices = self._sample_indices(flow_record) if self.random_shift else self._get_val_indices(flow_record)
        else:
            rgb_segment_indices = self._get_test_indices(rgb_record)
            flow_segment_indices = self._get_test_indices(flow_record)
        
        #dictionary = {}
        data_rgb = self.get(rgb_record, rgb_segment_indices)
        data_flow = self.get(flow_record, flow_segment_indices)
        data_question = flow_record.question
        data_label = flow_record.label
        return data_question, data_rgb, data_flow, data_label
    
    """Adding extra element - record.question"""
    
    def get(self, record, indices):

        images = list()
        modality = record.modality
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(record.new_length):
                seg_imgs = self._load_image(record.path, p, record.modality)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        if self.transform is not None:
            process_data = self.transform[modality](images)
        else:
            process_data = images
        return process_data#, record.label

    def __len__(self):
        return len(self.rgb_video_list)
