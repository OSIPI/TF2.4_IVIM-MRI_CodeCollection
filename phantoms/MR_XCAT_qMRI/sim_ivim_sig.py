import numpy as np
from scipy.io import loadmat
import nibabel as nib
import json

##########
# code written by Oliver J Gurney-Champion
# code adapted from MAtlab code by Eric Schrauben: https://github.com/schrau24/XCAT-ERIC
# This code generates a 4D IVIM phantom as nifti file

def phantom(bvalue, noise, TR=8000, TE=80, motion=False, rician=False, interleaved=False):
    np.random.seed(42)
    if motion:
        states = range(1,21)
    else:
        states = [1]
    for state in states:
        # Load the .mat file
        mat_data = loadmat('XCAT_MAT_RESP/XCAT5D_RP_' + str(state) + '_CP_1.mat')

        # Access the variables in the loaded .mat file
        XCAT = mat_data['IMG']
        XCAT = XCAT[0:-1:2,0:-1:2,10:160:4]

        D, f, Ds = contrast_curve_calc()
        S, Dim, fim, Dpim, legend = XCAT_to_MR_DCE(XCAT, TR, TE, bvalue, D, f, Ds)
        if state == 1:
            Dim_out = Dim
            fim_out = fim
            Dpim_out = Dpim

        if rician:
            S = np.abs(S + np.random.normal(loc=0, scale = noise, size = np.shape(S)) + 1j * np.random.normal(loc=0, scale = noise, size = np.shape(S)))
        else:
            S = S + np.random.normal(loc=0, scale = noise, size = np.shape(S))
        try:
            totsig=np.append(totsig,np.expand_dims(S, 4), axis=4)
        except:
            totsig = np.expand_dims(S, 4)
    if motion:
        state = 0
        state2 = 0
        slices_per_resp_phase = round(np.shape(XCAT)[2]/20*6000/TR)
        for b in range(np.shape(totsig)[3]):
            if not interleaved:
                for a in range(np.shape(totsig)[2]):
                    S[:,:,a,b]=totsig[:,:,a,b,state]
                    if state2 == slices_per_resp_phase:
                        state2 = 0
                        state = state + 1
                        if state == 20:
                            state = 0
                    else:
                        state2 = state2+1
            else:
                for a in range(0,np.shape(totsig)[2],2):
                    S[:,:,a,b]=totsig[:,:,a,b,state]
                    if state2 == slices_per_resp_phase:
                        state2 = 0
                        state = state + 1
                        if state == 20:
                            state = 0
                    else:
                        state2 = state2+1
                for a in range(1,np.shape(totsig)[2],2):
                    S[:,:,a,b]=totsig[:,:,a,b,state]
                    if state2 == slices_per_resp_phase:
                        state2 = 0
                        state = state + 1
                        if state == 20:
                            state = 0
                    else:
                        state2 = state2+1
    else:
        S=np.squeeze(totsig)
    return S, XCAT, Dim_out, fim_out, Dpim_out, legend


def ivim(bvalues,D,f,Ds):
    return (1-f) * np.exp(-D * bvalues) + f * np.exp(-Ds * bvalues)

def contrast_curve_calc():

    D = np.full(74, np.nan)
    D[1] = 2.4e-3  # 1 Myocardium LV : Delattre et al. doi: 10.1097/RLI.0b013e31826ef901
    D[2] = 2.4e-3  # 2 myocardium RV: Delattre et al. doi: 10.1097/RLI.0b013e31826ef901
    D[3] = 2e-3  # 3 myocardium la
    D[4] = 1.5e-3  # 4 myocardium ra
    D[5] = 3e-3  # 5 Blood LV
    D[6] = 3e-3  # 6 Blood RV
    D[7] = 3e-3  # 7 Blood la
    D[8] = 3e-3  # 8 Blood ra
    D[13] = 1.5e-3  # 13 liver: Delattre et al. doi: 10.1097/RLI.0b013e31826ef901
    D[17] = 1.67e-3  # 17 esophagus : Huang et al. doi: 10.1259/bjr.20170421
    D[18] = 1.67e-3  # 18 esophagus cont : Huang et al. doi: 10.1259/bjr.20170421
    D[20] = 1.5e-3  # 20 stomach wall: Li et al. doi: 10.3389/fonc.2022.821586
    D[22] = 1.3e-3  # 22 Pancreas (from literature)
    D[23] = 2.12e-3  # 23 right kydney cortex : van Baalen et al. Doi: jmri.25519
    D[24] = 2.09e-3  # 23 right kydney medulla : van Baalen et al. Doi: jmri.25519
    D[25] = 2.12e-3  # 23 left kydney cortex : van Baalen et al. Doi: jmri.25519
    D[26] = 2.09e-3  # 23 left kydney medulla : van Baalen et al. Doi: jmri.25519
    D[30] = 1.3e-3  # 30 spleen : Taimouri et al. Doi: 10.1118/1.4915495
    D[36] = 3e-3  # 36 artery
    D[37] = 3e-3  # 37 vein
    D[40] = 1.31e-3  # 40 asc lower intestine : Hai-Jing et al. doi: 10.1097/RCT.0000000000000926
    D[41] = 1.31e-3  # 41 trans lower intestine : Hai-Jing et al. doi: 10.1097/RCT.0000000000000926
    D[42] = 1.31e-3  # 42 desc lower intestine : Hai-Jing et al. doi: 10.1097/RCT.0000000000000926
    D[43] = 1.31e-3  # 43 small intestine : Hai-Jing et al. doi: 10.1097/RCT.0000000000000926
    D[50] = 3e-3  # 50 pericardium
    D[73] = 1.8e-3  # 73 Pancreatic tumor (advanced state, from literature)

    f = np.full(74, np.nan)
    f[1] = 0.15  # 1 Myocardium LV : Delattre et al. doi: 10.1097/RLI.0b013e31826ef901
    f[2] = 0.15  # 2 myocardium RV : Delattre et al. doi: 10.1097/RLI.0b013e31826ef901
    f[3] = 0.07  # 3 myocardium la
    f[4] = 0.07  # 4 myocardium ra
    f[5] = 1.00  # 5 Blood LV
    f[6] = 1.00  # 6 Blood RV
    f[7] = 1.00  # 7 Blood la
    f[8] = 1.00  # 8 Blood ra
    f[13] = 0.11  # 13 liver : Delattre et al. doi: 10.1097/RLI.0b013e31826ef901
    f[17] = 0.32  # 17 esophagus : Huang et al. doi: 10.1259/bjr.20170421
    f[18] = 0.32  # 18 esophagus cont : Huang et al. doi: 10.1259/bjr.20170421
    f[20] = 0.3  # 20 stomach wall: Li et al. doi: 10.3389/fonc.2022.821586
    f[22] = 0.15  # 22 Pancreas (from literature)
    f[23] = 0.097  # 23 right kydney cortex : van Baalen et al. Doi: jmri.25519
    f[24] = 0.158  # 23 right kydney medulla : van Baalen et al. Doi: jmri.25519
    f[25] = 0.097  # 23 left kydney cortex : van Baalen et al. Doi: jmri.25519
    f[26] = 0.158  # 23 left kydney medulla : van Baalen et al. Doi: jmri.25519
    f[30] = 0.2  # 30 spleen : Taimouri et al. Doi: 10.1118/1.4915495
    f[36] = 1.0  # 36 artery
    f[37] = 1.0  # 37 vein
    f[40] = 0.69  # 40 asc lower intestine : Hai-Jing et al. doi: 10.1097/RCT.0000000000000926
    f[41] = 0.69  # 41 trans lower intestine : Hai-Jing et al. doi: 10.1097/RCT.0000000000000926
    f[42] = 0.69  # 42 desc lower intestine : Hai-Jing et al. doi: 10.1097/RCT.0000000000000926
    f[43] = 0.69  # 43 small intestine : Hai-Jing et al. doi: 10.1097/RCT.0000000000000926
    f[50] = 0.07  # 50 pericardium
    f[73] = 0.37  # 73 Pancreatic tumor (advanced state, from literature)

    Ds = np.full(74, np.nan)
    Ds[1] = 0.08  # 1 Myocardium LV: Delattre et al. doi: 10.1097/RLI.0b013e31826ef901
    Ds[2] = 0.08  # 2 myocardium RV: Delattre et al. doi: 10.1097/RLI.0b013e31826ef901
    Ds[3] = 0.07  # 3 myocardium la
    Ds[4] = 0.07  # 4 myocardium ra
    Ds[5] = 0.1  # 5 Blood LV
    Ds[6] = 0.1  # 6 Blood RV
    Ds[7] = 0.1  # 7 Blood la
    Ds[8] = 0.1  # 8 Blood ra
    Ds[13] = 0.1  # 13 liver: Delattre et al. doi: 10.1097/RLI.0b013e31826ef901
    Ds[17] = 0.03  # 17 esophagus : Huang et al. doi: 10.1259/bjr.20170421
    Ds[18] = 0.03  # 18 esophagus cont : Huang et al. doi: 10.1259/bjr.20170421
    Ds[20] = 0.012  # 20 stomach wall: Li et al. doi: 10.3389/fonc.2022.821586
    Ds[22] = 0.01  # 22 Pancreas (from literature)
    Ds[23] = 0.02  # 23 right kydney cortex : van Baalen et al. Doi: jmri.25519
    Ds[24] = 0.019  # 23 right kydney medulla : van Baalen et al. Doi: jmri.25519
    Ds[25] = 0.02  # 23 left kydney cortex : van Baalen et al. Doi: jmri.25519
    Ds[26] = 0.019  # 23 left kydney medulla : van Baalen et al. Doi: jmri.25519
    Ds[30] = 0.03  # 30 spleen : Taimouri et al. Doi: 10.1118/1.4915495
    Ds[36] = 0.1  # 36 artery
    Ds[37] = 0.1  # 37 vein
    Ds[40] = 0.029  # 40 asc lower intestine : Hai-Jing et al. doi: 10.1097/RCT.0000000000000926
    Ds[41] = 0.029  # 41 trans lower intestine : Hai-Jing et al. doi: 10.1097/RCT.0000000000000926
    Ds[42] = 0.029  # 42 desc lower intestine : Hai-Jing et al. doi: 10.1097/RCT.0000000000000926
    Ds[43] = 0.029  # 43 small intestine : Hai-Jing et al. doi: 10.1097/RCT.0000000000000926
    Ds[50] = 0.01  # 50 pericardium
    Ds[73] = 0.01  # 73 Pancreatic tumor (advanced state, from literature)
    # Return values

    return D, f, Ds


def XCAT_to_MR_DCE(XCAT, TR, TE, bvalue, D, f, Ds, b0=3, ivim_cont = True):
    ###########################################################################################
    # This script converts XCAT tissue values to MR contrast based on the SSFP signal equation.
    # Christopher W. Roy 2018-12-04 # fetal.xcmr@gmail.com
    # T1 and T2 values are stored in a table with the format:
    # [T1 @ 1.5 T, T2 @ 1.5 T, T1 @ 3.0 T, T2 @ 3.0 T]
    # Note the current implimentation is for 1.5T
    # Relaxation values are based off of:
    # Stanisz GJ, Odrobina EE, Pun J, Escaravage M, Graham SJ, Bronskill MJ, Henkelman RM. T1, T2 relaxation and magnetization transfer in tissue at 3T. Magnetic resonance in medicine. 2005;54:507�12.
    # Portnoy S, Osmond M, Zhu MY, Seed M, Sled JG, Macgowan CK. Relaxation properties of human umbilical cord blood at 1.5 Tesla. Magnetic Resonance in Medicine. 2016;00:1�13.
    # https://www.itis.ethz.ch/virtual-population/tissue-properties/XCATbase/relaxation-times/ #Tissue legend:
    legend = {
        1: 'Myocardium LV',
        2: 'myocardium RV', #
        3: 'myocardium la',
        4: 'myocardium ra', # 5 Blood LV
        6: 'Blood RV',
        7: 'Blood LA',
        8: 'Blood RA',
        9: 'body',
        10: 'muscle',
        11: 'Brain',
        12: 'Sinus',
        13: 'Liver',
        14: 'gall bladder',
        15: 'Right Lung',
        16: 'Left Lung',
        17: 'esophagus',
        18: 'esophagus cont',
        19: 'laryngopharynx',
        20: 'st wall',
        21: 'Stomach Contents',
        22: 'pancreas',
        23: 'Right kydney cortex',
        24: 'right kidney medulla',
        25: 'Left kidney cortex',
        26: 'left kidney medulla',
        27: 'adrenal',
        28: 'Right Renal Pelvis',
        29: 'Left Renal Pelvis',
        30: 'spleen',
        31: 'Ribs',
        32: 'Cortical Bone',
        33: 'Spine',
        34: 'spinal cord',
        35: 'Bone Marrow',
        36: 'Artery',
        37: 'Vein',
        38: 'bladder',
        39: 'prostate',
        40: 'asc lower intestine',
        41: 'trans lower intestine',
        42: 'desc lower intestine',
        43: 'small intestine',
        44: 'rectum',
        45: 'seminal vescile',
        46: 'vas deference',
        47: 'testicles',
        48: 'epididymus',
        49: 'ejac duct',
        50: 'pericardium',
        51: 'Cartilage',
        52: 'Intestine Cavity',
        53: 'ureter',
        54: 'urethra',
        55: 'Lymph',
        56: 'lymph abnormal',
        57: 'trach bronch',
        58: 'Airway',
        59: 'uterus',
        60: 'vagina',
        61: 'right ovary',
        62: 'left ovary',
        63: 'FAllopian tubes',
        64: 'Parathyroid',
        65: 'Thyroid',
        66: 'Thymus',
        67: 'salivary',
        68: 'Pituitary',
        69: 'Eye',
        70: 'eye lens',
        71: 'lesion',
        72: 'Fat',
        73: 'Pancreas tumor',
    }
    ###############################################################################
    np.random.seed(42)
    Tissue = np.zeros((74, 4))
    Tissue[1] = [1030, 40, 1471, 47]
    Tissue[2] = [1030, 40, 1471, 47]
    Tissue[3] = [1030, 40, 1471, 47]
    Tissue[4] = [1030, 40, 1471, 47]
    Tissue[5] = [1441, 290, 1932, 275]
    Tissue[6] = [1441, 290, 1932, 275]
    Tissue[7] = [1441, 290, 1932, 275]
    Tissue[8] = [1441, 290, 1932, 275]
    #Tissue[9] = [1008, 44, 1412, 50.00]
    Tissue[9] = [9999999999, 0.00000001, 999999999, 0.00000001] ## fat suppression
    Tissue[10] = [981.5, 36, 1232.9, 37.20]
    Tissue[11] = [884, 72, 1084, 69]
    Tissue[12] = [0, 0, 0, 0]
    Tissue[13] = [576, 46, 812, 42]
    Tissue[14] = [576, 46, 812, 42]
    Tissue[15] = [0, 0, 0, 0]
    Tissue[16] = [0, 0, 0, 0]
    Tissue[17] = [576, 46, 812, 42]
    Tissue[18] = [576, 46, 812, 42]
    Tissue[19] = [1045.5, 37.3, 1201, 44]
    Tissue[20] = [981.5, 36, 1232.9, 37.20]
    #Tissue[20] = [981.5, 36, 1232.9, 37.20]
    Tissue[21] = [0, 0, 0, 0]
    Tissue[22] = [584, 46, 725, 43]
    Tissue[23] = [828, 71, 1168, 66]
    Tissue[24] = [1412, 85, 1545, 81]
    Tissue[25] = [828, 71, 1168, 66]
    Tissue[26] = [1412, 85, 1545, 81]
    Tissue[27] = [576, 46, 812, 42]
    Tissue[28] = [200, 0.5, 302, 0.25]
    Tissue[29] = [200, 0.5, 302, 0.25]
    Tissue[30] = [1057, 79, 1328, 61]
    Tissue[31] = [200, 0.5, 302, 0.25]
    Tissue[32] = [200, 0.5, 302, 0.25]
    Tissue[33, :] = [200, 0.5, 302, 0.25]
    Tissue[34, :] = [745, 74, 993, 78]
    Tissue[35, :] = [549, 49, 586, 49]
    Tissue[36, :] = [1585, 254, 1664, 147]
    Tissue[37, :] = [1582, 181, 1584, 66]
    Tissue[38, :] = [576, 46, 812, 42]
    Tissue[39, :] = [1317, 88, 1597, 74]
    Tissue[40, :] = [576, 46, 812, 42]
    Tissue[41, :] = [576, 46, 812, 42]
    Tissue[42, :] = [576, 46, 812, 42]
    Tissue[43, :] = [576, 46, 812, 42]
    Tissue[44, :] = [576, 46, 812, 42]
    Tissue[45, :] = [1317, 88, 1597, 74]
    Tissue[46, :] = [576, 46, 812, 42]
    Tissue[47, :] = [576, 46, 812, 42]
    Tissue[48, :] = [576, 46, 812, 42]
    Tissue[49, :] = [576, 46, 812, 42]
    Tissue[50, :] = [981.5, 36, 1232.9, 37.20]
    Tissue[51, :] = [1024, 30, 1168, 27]
    # Tissue[52, :] = [;;;]
    Tissue[54, :] = [1434.5, 164, 1498.3, 164]
    Tissue[55, :] = [1434.5, 164, 1498.3, 164]
    Tissue[56, :] = [5053, 468, 5053, 468]
    # Tissue[57, :] = [;;;]
    Tissue[58, :] = [1045.5, 37.3, 1201, 44]
    Tissue[59, :] = [0, 0, 0, 0]
    Tissue[60, :] = [1309, 117, 1514, 79]
    Tissue[61, :] = [1135, 58, 1616, 83]
    Tissue[62, :] = [576, 46, 812, 42]
    Tissue[63, :] = [576, 46, 812, 42]
    Tissue[64, :] = [576, 46, 812, 42]
    Tissue[65, :] = [1317, 88, 1597, 74]
    Tissue[66, :] = [1317, 88, 1597, 74]
    Tissue[67, :] = [1317, 88, 1597, 74]
    Tissue[68, :] = [1317, 88, 1597, 74]
    Tissue[69, :] = [1317, 88, 1597, 74]
    Tissue[69, :] = [5053, 468, 5053, 468]
    Tissue[70, :] = [1138, 26, 1138, 26]
    # Tissue[71, :] = [;;;]
    # Tissue[72, :] = [312.4, 117.4, 377, 97.5]
    Tissue[72, :] = [9999999999, 0.00000001, 999999999, 0.00000001] ## fat suppression
    Tissue[73, :] = [0, 0, 1676, 64]
    Dim= np.zeros(XCAT.shape, dtype=np.float32)
    fim= np.zeros(XCAT.shape, dtype=np.float32)
    Dpim= np.zeros(XCAT.shape, dtype=np.float32)
    MR = np.zeros(XCAT.shape+(len(bvalue),), dtype=np.float32)
    for iTissue in range(len(Tissue)):
        if iTissue != 0:
            if b0 == 1.5:
                T1 = Tissue[iTissue, 0]
                T2 = Tissue[iTissue, 1]
            else:
                T1 = Tissue[iTissue, 2]
                T2 = Tissue[iTissue, 3]

            if ivim_cont and not np.isnan([D[iTissue], f[iTissue], Ds[iTissue]]).any():
                # note we are assuming blood fraction has the same T1 as tissue fraction here for simplicity. Can be changed in future.
                Dtemp=D[iTissue]
                ftemp=f[iTissue]
                Dstemp=Ds[iTissue]
            else:
                Dtemp=5e-4+np.random.rand(1)*3e-3
                ftemp=np.random.rand(1)*0.5
                Dstemp=5e-3+np.random.rand(1)*1e-1
            S0 = ivim(bvalue,Dtemp,ftemp,Dstemp)
            if T1 > 0 or T2 > 0:
                MR = MR + np.tile(np.expand_dims(XCAT == iTissue,3),len(S0)) * S0 * (1 - 2 * np.exp(-(TR - TE / 2) / T1) + np.exp(-TR / T1)) * np.exp(-TE / T2)
            Dim = Dim + (XCAT == iTissue) * Dtemp
            fim = fim + (XCAT == iTissue) * ftemp
            Dpim = Dpim + (XCAT == iTissue) * Dstemp
    return MR, Dim, fim, Dpim, legend

if __name__ == '__main__':
    bvalue = np.array([0., 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 250, 350, 400, 550, 700, 850, 1000])
    noise = 0.0005
    motion = False
    interleaved = False
    sig, XCAT, Dim, fim, Dpim, legend = phantom(bvalue, noise, motion=motion, interleaved=interleaved)
    # sig = np.flip(sig,axis=0)
    # sig = np.flip(sig,axis=1)
    res=np.eye(4)
    res[2]=2

    voxel_selector_fraction = 0.5
    D, f, Ds = contrast_curve_calc()
    ignore = np.isnan(D)
    generic_data = {}
    for level, name in legend.items():
        if len(ignore) > level and ignore[level]:
            continue
        selector = XCAT == level
        voxels = sig[selector]
        if len(voxels) < 1:
            continue
        signals = np.squeeze(voxels[int(voxels.shape[0] * voxel_selector_fraction)]).tolist()
        generic_data[name] = {
            'noise': noise,
            'D': np.mean(Dim[selector], axis=0),
            'f': np.mean(fim[selector], axis=0),
            'Dp': np.mean(Dpim[selector], axis=0),
            'data': signals
        }
    generic_data['config'] = {
        'bvalues': bvalue.tolist()
    }
    with open('generic.json', 'w') as f:
        json.dump(generic_data, f, indent=4)


    nifti_img = nib.Nifti1Image(sig, affine=res)  # Replace affine if necessary
    # Save the NIfTI image to a file
    nifti_img.header.set_data_dtype(np.float64)
    if not motion:
        output_file = 'output.nii.gz'  # Replace with your desired output file name
    elif interleaved:
        output_file = 'output_resp_int.nii.gz'  # Replace with your desired output file name
    else:
        output_file = 'output_resp.nii.gz'  # Replace with your desired output file name

    nib.save(nifti_img, output_file)


    nifti_img = nib.Nifti1Image(XCAT, affine=res)  # Replace affine if necessary
    # Save the NIfTI image to a file
    output_file = 'output_xcat.nii.gz'  # Replace with your desired output file name
    nib.save(nifti_img, output_file)

    nifti_img = nib.Nifti1Image(Dim, affine=res)  # Replace affine if necessary
    # Save the NIfTI image to a file
    nifti_img.header.set_data_dtype(np.float64)
    output_file = 'D.nii.gz'  # Replace with your desired output file name
    nib.save(nifti_img, output_file)

    nifti_img = nib.Nifti1Image(fim, affine=res)  # Replace affine if necessary
    # Save the NIfTI image to a file
    nifti_img.header.set_data_dtype(np.float64)
    output_file = 'f.nii.gz'  # Replace with your desired output file name
    nib.save(nifti_img, output_file)

    nifti_img = nib.Nifti1Image(Dpim, affine=res)  # Replace affine if necessary
    # Save the NIfTI image to a file
    nifti_img.header.set_data_dtype(np.float64)
    output_file = 'Dp.nii.gz'  # Replace with your desired output file name
    nib.save(nifti_img, output_file)

    np.savetxt('bvals.txt', bvalue)