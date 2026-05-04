import numpy as np
import torch
import torchio
from torchio.transforms import Transform


class AverageSignalsOfEqualXvals(Transform):

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        """
        normalize signals
        Args:
            signals: signals array to normalize
            xvals: xval array

        Returns:
            normalized_signals: normalized signals array

        """
        images_dict = subject.get_images_dict()
        signals = images_dict['signals'].numpy()
        xvals = np.squeeze(images_dict['xvals'].numpy())
        signals, xvals = self.average_signal_of_equal_xvals(signals, xvals)
        subject.add_image(torchio.Image(tensor=torch.Tensor(signals)), 'signals')
        subject.add_image(torchio.Image(tensor=torch.Tensor(np.reshape(xvals, (xvals.shape[0], 1, 1, 1)))), 'xvals')
        return subject

    @staticmethod
    def average_signal_of_equal_xvals(signals, xvals):
        """
        average the signal of equal xvals
        Args:
            signals: signal matrix [signals X xvals]
            xvals: array of xvals

        Returns:
            averaged_signal_matrix: averaged signal matrix [signals X unique_xvals]
            unique_xval_arrays: unique xvals in averaged signal matrix

        """
        unique_xvals = np.unique(xvals)
        averaged_signals = np.zeros((unique_xvals.shape[0], *signals.shape[1:]))
        for xval_idx, unique_xval in enumerate(unique_xvals):
            averaged_signals[xval_idx, ...] = np.squeeze(np.mean(signals[np.where(xvals == unique_xval), ...], axis=1))
        return averaged_signals, unique_xvals
