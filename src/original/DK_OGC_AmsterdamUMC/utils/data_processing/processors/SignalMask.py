import numpy as np
import torch
import torchio
from torchio.transforms import Transform


class SignalMask(Transform):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        """
        curates signals Image of subject
        Args:
            subject: Subject

        Returns:
            subject: Subject
        """
        images_dict = subject.get_images_dict()
        signals = images_dict['signals'].numpy()
        signal_mask = self.signal_mask(signals)
        subject.add_image(torchio.Image(tensor=torch.Tensor(np.expand_dims(signal_mask, 0))), 'signal_mask')
        return subject

    @staticmethod
    def signal_mask(signals):
        """
        returns mask with nonzero element for signal vectors with nonzero entries
        Args:
            signals: signals

        Returns:
            masked_signals: signals containing nonzero elements
        """

        return np.sum(signals, axis=0) > 0
