import os
import re
import shutil
import subprocess
from typing import Optional, Tuple


class TOPED:
    def __init__(self, distribution: str, user: str, linux_workdir: str):
        """ Specify WSL distribution, user, and working directory for FSL.
        Parameters:
            distribution : str                      The WSL distribution where FSL is installed (e.g., "Ubuntu").
            user : str                              The Linux username to run commands as.
            linux_workdir : str                     The directory in the Linux filesystem to use for temporary files.
        """

        if not os.path.exists(linux_workdir):
            #raise ValueError(f"Linux working directory not found: {linux_workdir}")
            print(f"Linux working directory not found, creating: {linux_workdir}")
            os.makedirs(linux_workdir)

        self.distribution = distribution
        self.user = user
        self.linux_workdir = linux_workdir

    # CORE COMMAND FUNCTION
    def run_command(self, command: str, workdir: str):
        """ Run a generic FSL command through WSL.
        Parameters:
            command : str                           The FSL command to execute.
            workdir : str                           The working directory in the Linux filesystem where the command should be executed.
        """
        # print the full command in the IDE output when running the fsl_runner.
        print(f" command: {command}")

        full_cmd = f"source ~/.profile; source ~/.bashrc; {command}"
        return subprocess.run(
            ['wsl', '-d', self.distribution, '-u', self.user, '-e', 'bash', '-c', full_cmd],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True
        )

    # TOPUP
    def run_topup(
        self,
        b0_path: str,
        acq_parameter_path: str,
        config_file_path: Optional[str] = None,
        nthr: Optional[int] = None,
        logout: Optional[str] = None, 
        dfout: Optional[str] = None, 
        jacout: Optional[str] = None, 
        patient_id: Optional[str] = None,
        output_dir: Optional[str] = None 
    ) -> str:
        """ Run FSL TOPUP on a stack of b0 images.
        Parameters:
            b0_path : str                           The path to the b0 image to process with TOPUP.
            acq_parameter_path : str                Path to the acquisition parameters file.
            config_file_path : str, optional        Path to the TOPUP configuration file, by default None, which uses the default FSL config b02b0.cnf.
            nthr : int, optional                    Number of threads to use, by default None (uses default_threads)
        Returns:
            whatever specified in the TOPUP command output (dfout, iout, jacout)
            Corrected image, fieldmap, and Jacobian determinant paths are generated in the Linux working directory.
        """
        workdir = self.get_patient_workdir(patient_id)

        if not os.path.exists(b0_path):
            raise ValueError("b0_path invalid")
        if not os.path.exists(acq_parameter_path):
            raise ValueError("acq_parameter_path invalid")

        output_dir = output_dir or os.path.dirname(b0_path)

        # Copy inputs
        b0_name = self.copy_to_wsl(b0_path, workdir)
        b0_basename, _ = self.split_nifti_gz(b0_name)
    
        acqparam = self.copy_to_wsl(acq_parameter_path, workdir)
        if config_file_path:
            config_filename = self.copy_to_wsl(config_file_path, workdir)
            config_arg = f"--config={config_filename} "
        else:
            # FSL default config
            config_arg = "--config=b02b0.cnf "

        # Build command
        cmd = (
            f"topup "
            f"--imain={b0_name} "
            f"--datain={acqparam} "
            f"--out={b0_basename} "
            f"--iout={self.get_corrected_name(b0_basename)} "
        )

        if logout is not None: cmd += f"--logout={self.get_log_name(b0_basename)} "
        if dfout is not None: cmd += f"--dfout={self.get_warpfig_name(b0_basename)} "
        if jacout is not None: cmd += f"--jacout={self.get_jacobian_name(b0_basename)} "
        if nthr is not None: cmd += f"--nthr={nthr} "
        cmd += config_arg

        # Run
        self.run_command(cmd, workdir)

        # Copy corrected image back
        self.ensure_dir(output_dir)
        corrected_filename = self.get_corrected_name(b0_basename, ext=True)
        self.copy_from_wsl(corrected_filename, output_dir, workdir)

        # Return topup base for Eddy
        return b0_basename

    # Eddy
    def run_eddy(
        self,
        dwi_path: str,
        mask_path: str,
        acq_parameter_path: str,
        index_path: str,
        bvecs_path: str,
        bvals_path: str,
        topup_base: str,
        slspec_path: Optional[str] = None,
        b_range: Optional[int] = None,
        flm: Optional[str] = None,
        slm: Optional[str] = None,
        niter: Optional[int] = None,
        fwhm: Optional[str] = None,
        resamp: Optional[str] = None,
        fep: Optional[bool] = None,
        repol: Optional[bool] = None,
        estimate_move_by_susceptibility: Optional[bool] = None,
        mporder: Optional[int] = None,
        patient_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        json_path: Optional[str] = None, 
        interp: Optional[str] = None,
        nvoxhp: Optional[int] = None,
        ff: Optional[float] = None,
        dont_sep_offs_move: Optional[bool] = None,
        dont_peas: Optional[bool] = None,
        ol_nstd: Optional[float] = None,
        ol_nvox: Optional[int] = None,
        ol_type: Optional[str] = None,
        ol_ss: Optional[str] = None,
        ol_pos: Optional[bool] = None,
        ol_sqr: Optional[bool] = None,
        s2v_niter: Optional[int] = None,
        s2v_lambda: Optional[float] = None,
        s2v_interp: Optional[str] = None,
        mbs_niter: Optional[int] = None,
        mbs_lambda: Optional[float] = None,
        mbs_ksp: Optional[float] = None,
        cnr_maps: Optional[bool] = None,
        residuals: Optional[bool] = None,
        data_is_shelled: Optional[bool] = None,
        nthr: Optional[int] = None
    ) -> str:

        """ Run FSL Eddy.
        Parameters:
            dwi_path : str                          Path to the input DWI image. This is the stack to be corrected by Eddy & TOPUP.
            mask_path : str                         Path to the mask image.
            acqparams_path : str                    Path to acquisition parameters file
            index_path : str                        Path to index file
            bvecs_path : str                        Path to bvecs file
            bvals_path : str                        Path to bvals file
            topup_base : str                        Base name used for topup output. is is automatically set by TOUP in the linux working directory and is used as input for Eddy.
            output_dir : str                        Output directory
            bunch of optional Eddy flags (set to None to use FSL defaults).
        Returns:
            (Eddy + TOPUP) Corrected DWI image (path to output image)
        """

        workdir = self.get_patient_workdir(patient_id)

        for p in [dwi_path, mask_path, acq_parameter_path, index_path, bvecs_path, bvals_path]:
            if not os.path.exists(p):
                raise ValueError(f"Missing compulsory file: {p}")

        output_dir = output_dir or os.path.dirname(dwi_path)

        # Copy inputs
        dwi_name = self.copy_to_wsl(dwi_path, workdir)
        dwi_basename, _ = self.split_nifti_gz(dwi_name)

        mask = self.copy_to_wsl(mask_path, workdir)
        acqp = self.copy_to_wsl(acq_parameter_path, workdir)
        index = self.copy_to_wsl(index_path, workdir)
        bvecs = self.copy_to_wsl(bvecs_path, workdir)
        bvals = self.copy_to_wsl(bvals_path, workdir)

        slspec = self.copy_to_wsl(slspec_path, workdir) if slspec_path else None
        json_file = self.copy_to_wsl(json_path, workdir)   if json_path   else None

        # Build command
        cmd = (
            f"eddy "
            f"--imain={dwi_name} "
            f"--mask={mask} "
            f"--acqp={acqp} "
            f"--index={index} "
            f"--bvecs={bvecs} "
            f"--bvals={bvals} "
            f"--topup={topup_base} "
            f"--out={self.get_corrected_name_eddy(dwi_basename)} "
        )

        # Optional params
        if slspec: cmd += f"--slspec={slspec} "
        if json_file: cmd += f"--json={json_file} "
        if b_range is not None: cmd += f"--b_range={b_range} "

        if flm is not None: cmd += f"--flm={flm} "
        if slm is not None: cmd += f"--slm={slm} "
        if fwhm is not None: cmd += f"--fwhm={fwhm} "
        if niter is not None: cmd += f"--niter={niter} "
        if interp is not None: cmd += f"--interp={interp} "
        if resamp is not None: cmd += f"--resamp={resamp} "
        if fep: cmd += f"--fep "
        if nvoxhp is not None: cmd += f"--nvoxhp={nvoxhp} "
        if ff is not None: cmd += f"--ff={ff} "
        if dont_sep_offs_move: cmd += f"--dont_sep_offs_move "
        if dont_peas: cmd += f"--dont_peas "

        if repol: cmd += f"--repol "
        if ol_nstd is not None: cmd += f"--ol_nstd={ol_nstd} "
        if ol_nvox is not None: cmd += f"--ol_nvox={ol_nvox} "
        if ol_type is not None: cmd += f"--ol_type={ol_type} " 
        if ol_ss is not None: cmd += f"--ol_ss={ol_ss} "
        if ol_pos: cmd += f"--ol_pos "
        if ol_sqr: cmd += f"--ol_sqr "

        if mporder is not None: cmd += f"--mporder={mporder} "
        if s2v_niter is not None: cmd += f"--s2v_niter={s2v_niter} "
        if s2v_lambda is not None: cmd += f"--s2v_lambda={s2v_lambda} "
        if s2v_interp is not None: cmd += f"--s2v_interp={s2v_interp} "
        
        if estimate_move_by_susceptibility: cmd += f"--estimate_move_by_susceptibility "
        if mbs_niter is not None: cmd += f"--mbs_niter={mbs_niter} "
        if mbs_lambda is not None: cmd += f"--mbs_lambda={mbs_lambda} "
        if mbs_ksp is not None: cmd += f"--mbs_ksp={mbs_ksp} "

        if cnr_maps: cmd += f"--cnr_maps "
        if residuals: cmd += f"--residuals "
        if data_is_shelled: cmd += f"--data_is_shelled "
        if nthr is not None: cmd += f"--nthr={nthr} "   


        # Run
        self.run_command(cmd, workdir)

        # Copy output back
        self.ensure_dir(output_dir)
        output_filename = self.get_corrected_name_eddy(dwi_basename, ext=True)
        self.copy_from_wsl(output_filename, output_dir, workdir)

        return os.path.join(output_dir, output_filename)


    # FILE HANDLING between windows and wsl
    def copy_to_wsl(self, path: str, workdir: str) -> str:
        filename = os.path.basename(path)
        dst = os.path.join(workdir, filename)
        shutil.copy(path, dst)
        return filename

    def copy_from_wsl(self, filename: str, output_dir: str, workdir: str):
        src = os.path.join(workdir, filename)
        dst = os.path.join(output_dir, filename)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Expected output not found: {src}")
        shutil.copy(src, dst)


    # HELPERS - filenames and folders etc.
    def split_nifti_gz(self, filename: str) -> Tuple[str, str]:
        match = re.match(r"^(.*)(\.nii\.gz)$", filename)
        return match.group(1), match.group(2) if match else os.path.splitext(filename)

    def get_warpfig_name(self, base: str):
        return f"{base}_warpfig"

    def get_jacobian_name(self, base: str):
        return f"{base}_jacobian"
    
    def get_log_name(self, base: str):
        return f"{base}_log"   

    def get_corrected_name(self, base: str, ext=False):
        return f"{base}_corrected.nii.gz" if ext else f"{base}_corrected"

    def get_corrected_name_eddy(self, base: str, ext=False):
        return f"{base}_EddyCorrected.nii.gz" if ext else f"{base}_EddyCorrected"
    
    def ensure_dir(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_patient_workdir(self, patient_id: Optional[str]) -> str:
        if not patient_id:
            return self.linux_workdir
        path = os.path.join(self.linux_workdir, patient_id)
        if not os.path.exists(path):
            os.makedirs(path)
        return path