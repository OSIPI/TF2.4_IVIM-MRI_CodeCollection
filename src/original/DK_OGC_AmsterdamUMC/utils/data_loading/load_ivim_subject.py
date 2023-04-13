import os
import logging
import torch
import numpy as np
import torchio as tio


def load_ivim_subject(study_subject_path):
    """
    loads torchio subject with IVIM data (signals and bvalues)
    Args:
        study_subject_path: path in which subject data is located

    Returns:

    """
    subject_dict = {}

    # find all files that match study path and subject id
    for file in os.listdir(study_subject_path):
        file_path = os.path.join(study_subject_path, file)
        logging.info(f'start loading data from {file_path}')

        # Check file extension for image file
        if file_path[-2:] == "gz" or file_path[-2:] == "ii":

            # load nifti image
            image = tio.Image(file_path)
            image.set_data(image.data.to(dtype=torch.float32))
            subject_dict['signals'] = image

        # Check if file contains bvalues
        elif file_path[-2:] == "al":
            text_file = np.genfromtxt(file_path)
            bvals = np.array(text_file)
            subject_dict["xvals"] = tio.Image(tensor=torch.Tensor(np.reshape(bvals, (bvals.shape[0], 1, 1, 1))))

        else:
            logging.info(f'skipping loading of file {file_path}, no appropriate file extension. ')

        # Create subject
    if 'xvals' in subject_dict.keys() and 'signals' in subject_dict.keys():
        return tio.Subject(subject_dict)
