import torch
import torchio
import numpy as np

from torchio.transforms import Transform


class NormalizeXvals(Transform):

    def __init__(self, normalization_factor, **kwargs):
        self.normalization_factor = normalization_factor
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        """
        normalize xvals Image of subject
        Args:
            subject: Subject

        Returns:
            subject: Subject
        """
        images_dict = subject.get_images_dict()
        xvals = np.squeeze(images_dict['xvals'].numpy())
        xvals = self.normalize_xvals(xvals, self.normalization_factor)
        subject.add_image(torchio.Image(tensor=torch.Tensor(np.reshape(xvals, (xvals.shape[0], 1, 1, 1)))), 'xvals')
        return subject

    @staticmethod
    def normalize_xvals(xvals, normalization_factor):
        """
        normalize signals
        Args:
            xvals: xvalue array
            normalization_factor: factor to multiply xvals with

        Returns:
            normalized_xvals: normalized signals array
        """
        normalized_xvals = xvals * normalization_factor
        return normalized_xvals
