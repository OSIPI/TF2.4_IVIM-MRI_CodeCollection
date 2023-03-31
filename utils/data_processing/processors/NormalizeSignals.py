import numpy as np
import torch
import torchio
from torchio.transforms import Transform


class NormalizeSignals(Transform):

    def __init__(self, xval_threshold, **kwargs):
        self.xval_threshold = xval_threshold
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
        signals = images_dict['signals'].numpy()
        xvals = np.squeeze(images_dict['xvals'].numpy())
        signals = self.normalize_signals(signals, xvals, self.xval_threshold)
        subject.add_image(torchio.Image(tensor=torch.Tensor(signals)), 'signals')
        return subject

    @staticmethod
    def normalize_signals(signals, xvals, xval_threshold):
        """
         normalize signals
         Args:
             signals: signals array to normalize
             xvals: xval array
             xval_threshold: threshold below which bvals are considered b0

         Returns:
             normalized_signals: normalized signals array

         """
        # get average b0 signals and set signals with S0 of 0 to nan
        mean_S0 = np.nanmean(signals[xvals <= xval_threshold, :, :, :], axis=0)
        signals[:, mean_S0 == 0] = np.nan

        # normalize signals to S0 intensity
        normalized_signals = signals / mean_S0
        normalized_signals[np.isnan(normalized_signals)] = 0

        return normalized_signals


