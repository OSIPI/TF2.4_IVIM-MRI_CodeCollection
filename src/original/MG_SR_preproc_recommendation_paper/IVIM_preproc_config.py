import yaml
import os


class DenoiseConfig:
    """ Specify denoising parameters as part of the IVIMPreprocConfig.

    Currently supported options are: mppca and localpca from the dipy library.
    Parameters to set:
        enabled:        boolean, whether to perform denoising (default: True)
        method:         mppca (default), localpca
        patch_radius:   integer specifying the patch radius used for denoising.
                        default 2
        
    """
    def __init__(self, enabled=True, method='mppca', patch_radius=2):
        self.enabled = enabled
        self.method = method
        self.patch_radius = patch_radius
        self.validate()

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("Enabled must be boolean.")
        if not isinstance(self.method, str):
            raise ValueError("Method must be a string.")
        if self.method not in ["mppca", "localpca"]:
            raise ValueError("Invalid denoising method: must be 'mppca' or 'localpca'")

        if self.patch_radius <= 0:
            raise ValueError("denoise.patch_radius must be > 0")


class MotionConfig:
    """ Specify motion correction parameters as part of the IVIMPreprocConfig.

    Currently supported parameters:
        enabled:         boolean, whether to perform motion correction (default: True)
        method:          rigid (default) or affine registration between image volumes and b0.
    """

    def __init__(self, enabled=True, method='rigid'):
        self.enabled = enabled
        self.method = method
        self.validate()

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("Enabled must be boolean.")
        if not isinstance(self.method, str):
            raise ValueError("Method must be a string.")
        if self.method not in ["rigid", "affine"]:
            raise ValueError("Invalid motion correction method: must be 'rigid' or 'affine'.")


class DistortionConfig:
    """ Specify distortion correction parameters.

    Currently supported parameters:
        enabled:         boolean, whether to perform distortion correction (default: False)
        blip_down_path:  path to the blip-down data nifti file (default not provided: '')
        acqparams_path:  path to the acqparams.txt file (default not provided: '')
        working_dir:     path to the working directory for temporary files (default not provided: '', falls back to system temp dir)
        b0_threshold:    threshold for identifying b0 volumes (default: 50), images with b-values below this threshold will be considered b0 images for distortion correction.
    
    NOTE: validate() does not check if FSL-based correction can be run, that step is done in the distortion() function 
    in IVIM_preproc_pipeline.py, which checks if the blip_down_path and acqparams_path are valid and if FSL is installed.
    FSL-based correction will always run if files are provided and valid.
    If these checks fail, the pipeline defaults to registration-based distortion correction, which does not require these files.
    Suggestion to adust this later..
    """

    def __init__(self, enabled=False, blip_down_path='', acqparams_path='',
                 working_dir='', b0_threshold=50):
        self.enabled = enabled
        self.blip_down_path = blip_down_path
        self.acqparams_path = acqparams_path
        self.working_dir = working_dir
        self.b0_threshold = b0_threshold
        self.validate()
 
    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("distortion.enabled must be boolean.")
 
        for name, value in [("blip_down_path", self.blip_down_path),
                             ("acqparams_path", self.acqparams_path),
                             ("working_dir", self.working_dir)]:
            if not isinstance(value, str):
                raise ValueError(
                    f"distortion.{name} must be a string, got {type(value).__name__}")
 
        # isinstance(x, (int, float)) is True for bools in Python, so check
        # for bool explicitly first to avoid accepting e.g. b0_threshold=True.
        if isinstance(self.b0_threshold, bool) or not isinstance(self.b0_threshold, (int, float)):
            raise ValueError("distortion.b0_threshold must be a number.")
        if self.b0_threshold < 0:
            raise ValueError("distortion.b0_threshold must be >= 0.")


class SignalVoidConfig:
    """ Specify signal void exclusion parameters.

    Currently supported parameters:
        enabled:         boolean (default: True) None, defaults are based on region: False for brain, True for body.
        threshold:       intensity threshold for identifying signal voids (default: 0.1), images with signal below this threshold will be considered signal voids and excluded from processing.
    """
    def __init__(self, enabled=True, threshold=0.1):
        self.enabled = enabled
        self.threshold = threshold
        self.validate()
 
    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("signal_void.enabled must be boolean.")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)):
            raise ValueError("signal_void.threshold must be a number.")


class IVIMPreprocConfig:
    """ IVIM Preprocessing Config
    Class to create, load, and validate a config file containing elements
    and options of IVIM pre-processing.

    Use IVIMPreprocConfig.create() to create a default config file. Optionally,
    parameters can be adjusted and specified manually with this function.

    Use IVIMPreprocConfig.save() to save the created config.

    To load an existing config, use IVIMPreprocConfig.load_or_create().
    However, if no config is specified or if the config file provided does
    not exist, a default config file will be created for region=brain.
    Parameters can not directly be adjusted with this function (use
    create instead).

    Args:
        region:     brain or body (default: brain). Sets enabled/disabled for all steps unless overridden.
                                brain:             body:
        denoise:                enabled            enabled
        motion (registration):  enabled            enabled
        signal_void:            disabled           enabled
        distortion:             enabled            disabled

    Authors:
    Susi Rauh
    Minoo Gandomi
    """

    def __init__(self,
                 region='brain',
                 denoise=None,
                 motion=None,
                 signal_void=None,
                 distortion=None,
                 ):
        self.region = region
        self.denoise = denoise or DenoiseConfig()
        self.motion = motion or MotionConfig()
        
        # motion & denoising is default True regardless of region. signal void and distortion is region dependant.
        
        #default, applied when SignalVoidConfig has not been provided. True for body.
        if signal_void is None:
            default_signal_void_enabled = (region != 'brain')
            signal_void = SignalVoidConfig(enabled=default_signal_void_enabled)
        self.signal_void = signal_void

        #default, applied when DistortionConfig has not been provided. True for brain.
        if distortion is None:
            default_distortion_enabled = (region == 'brain')
            distortion = DistortionConfig(enabled=default_distortion_enabled)
        self.distortion = distortion

        # Validate config
        self.validate()

    def validate(self):
        if self.region not in ["brain", "body"]:
            raise ValueError("Region must be 'brain' or 'body'")

        # validate sub-configs
        self.denoise.validate()
        self.motion.validate()
        self.signal_void.validate()
        self.distortion.validate() # does not check paths, wont crash?
        

    def to_dict(self):
        return {
            "region": self.region,
            "denoise": vars(self.denoise),
            "motion": vars(self.motion),
            "signal_void": vars(self.signal_void),
            "distortion": vars(self.distortion),
        }

    def save(self, filename):
        with open(filename, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)



    # Create default config
    @classmethod
    def default(cls):
        return cls()

    @classmethod
    def create(cls, **overrides):
        """Build Config from defaults, with optional overrides for any parameter.
        NOTE: 'enabled' paramter exists for MotionConfig, DistortionConfig, and SignalVoidConfig. 
        Overrides mare matched by field-name, thus 'create(enabled=False)' will only be applied to motion, and not set all three.
        'enabled' should be set explicitly e.g. 'IVIMPreprocConfig.create(motion=MotionConfig(enabled=False))'"""

        # Build with the requested region from the start (rather than
        # building a brain-default cfg and patching cfg.region afterward),
        # so region-dependent defaults -- currently just signal_void.enabled
        # -- are computed correctly. Patching `.region` post-construction
        # would NOT retroactively fix signal_void.enabled, since that default
        # is only ever applied inside __init__.
        # fix these to set daefautls better
        # WIP!
        region = overrides.get('region', 'brain')
        cfg = cls(region=region)
 
        for key, value in overrides.items():
            if key == 'region':
                continue  # already applied above via the constructor
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
            elif hasattr(cfg.denoise, key):
                setattr(cfg.denoise, key, value)
            elif hasattr(cfg.motion, key):
                setattr(cfg.motion, key, value)
            elif hasattr(cfg.distortion, key):
                setattr(cfg.distortion, key, value)
            elif hasattr(cfg.signal_void, key):
                setattr(cfg.signal_void, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")
 
        cfg.validate()
        return cfg

    # Load from YAML
    @classmethod
    def from_yaml(cls, filename):
        with open(filename) as f:
            raw = yaml.safe_load(f)

        denoise = DenoiseConfig(**raw.get("denoise", {}))
        motion = MotionConfig(**raw.get("motion", {}))

        distortion_raw = raw.get("distortion")
        distortion = DistortionConfig(**distortion_raw) if distortion_raw is not None else None
        signal_void_raw = raw.get("signal_void")
        signal_void = SignalVoidConfig(**signal_void_raw) if signal_void_raw is not None else None

        return cls(
            region=raw.get("region", "brain"),
            denoise=denoise,
            motion=motion,
            distortion=distortion,
            signal_void=signal_void,
        )

    @classmethod
    def load_or_create(cls, config_path=None):
        """
        Load config if exists. A yaml file is expected as input.
        Otherwise, create default, optionally save it, and return it.

        WIP SR: Maybe this is unnecessary, and a valid config needs to
        be specified in the pipeline (if not: throw error to create config
        first). Now, default config will automatically be created and used,
        which might result in unpredictable/unwanted behaviour.
        """

        if config_path is None:
            print("IVIM Preprocessing: No config specified. "
                  "Using default config.")
            return cls.default()

        if not os.path.exists(config_path):
            print(f"IVIM Preprocessing Config '{config_path} not found. "
                  f"Creating default config.")

            cfg = cls.default()
            cfg.save(config_path)

            print(f"IVIM Preprocessing Default config written to: "
                  f"{config_path}")
            return cfg

        # Load existing
        _, ext = os.path.splitext(config_path)
        if not ext == '.yml':
            raise Exception("Only yaml file is accepted as config file. "
                            "Got ", ext, " file.")

        print(f"IVIM Preprocessing Loading config: {config_path}")
        return cls.from_yaml(config_path)
