import numpy as np
import torch
import torchio
from torchio.transforms import Transform


class NormalizeMaxSignal(Transform):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        """
        normalize signals Image of subject
        Args:
            subject: Subject

        Returns:
            subject: Subject
        """
        images_dict = subject.get_images_dict()
        signals = images_dict['signals'].numpy()
        signals = self.normalize_signals(signals)
        subject.add_image(torchio.Image(tensor=torch.Tensor(signals)), 'signals')
        return subject

    @staticmethod
    def normalize_signals(signals):
        """
         normalize signals
         Args:
             signals: signals array to normalize

         Returns:
             normalized_signals: normalized signals array

         """
        maxsignal = np.nanmax(signals, axis=0)
        signals /= maxsignal

        return signals


