import numpy as np
import torch
import torchio
from torchio.transforms import Transform


class SortSignalOnXval(Transform):

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        """
        Sorts signals and xvals on ascending xvals
        Args:
            subject: Subject

        Returns:
            subject: Subject
        """
        images_dict = subject.get_images_dict()
        signals = images_dict['signals'].numpy()
        xvals = np.squeeze(images_dict['xvals'].numpy())
        signals, xvals = self.sort_signals_on_xval_array(signals, xvals)
        subject.add_image(torchio.Image(tensor=torch.Tensor(signals)), 'signals')
        subject.add_image(torchio.Image(tensor=torch.Tensor(np.reshape(xvals, (xvals.shape[0], 1, 1, 1)))), 'xvals')
        return subject

    @staticmethod
    def sort_signals_on_xval_array(signals, xvals):
        """
        Sorts signals and xvals on ascending xvals
        Args:
            signals: signals to sort
            bval: bvalues to use for sorting

        Returns:
            sorted_signals: sorted signals
            sorted_bvals: sorted bvals

        """
        sorted_xval_idcs = np.argsort(xvals)
        sorted_xvals = xvals[sorted_xval_idcs]
        sorted_signals = signals[sorted_xval_idcs, ...]
        return sorted_signals, sorted_xvals