import yaml
import os


class DenoiseConfig:
    """ Specify denoising parameters as part of the IVIMPreprocConfig.

    Currently supported options are: mppca and localpca from the dipy library.
    Parameters to set:
        method:         mppca (default) or localpca
        patch_radius:   integer specifying the patch radius used for denoising.
                        default 2
    """
    def __init__(self, method='mppca', patch_radius=2):
        self.method = method
        self.patch_radius = patch_radius
        self.validate()

    def validate(self):
        if self.method not in ["mppca", "localpca", None]:
            raise ValueError("Invalid denoising method")

        if self.patch_radius <= 0:
            raise ValueError("denoise.patch_radius must be > 0")


class MotionConfig:
    """ Specify motion correction parameters as part of the IVIMPreprocConfig.

    TBD
    """
    def __init__(self, rigid=True):
        self.rigid = rigid

    def validate(self):
        if not isinstance(self.rigid, bool):
            raise ValueError("Rigid must be boolean.")


class DistortionConfig:
    """ Specify distortion correction parameters.

    TBD, currently placeholder
    """
    def __init__(self, blip_down_path='', acqparams_path='', working_dir=''):
        self.blip_down_path = blip_down_path
        self.acqparams_path = acqparams_path
        self.working_dir = working_dir

    def validate(self):
        if not os.path.isdir(self.blip_down_path):
            raise ValueError("Specified blip_down_path does not exist: ",
                             self.blip_down_path)


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
        region:     brain or body (default: brain)
        denoise:
        motion:
        signal_void:
        distortion:

    Authors:
    Susi Rauh
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

        # Region-dependent defaults
        if signal_void is None:
            if region == "brain":
                signal_void = False
            else:   # body
                signal_void = True

        self.signal_void = signal_void
        self.distortion = distortion or DistortionConfig()

        # Validate config
        self.validate()

    def validate(self):
        if self.region not in ["brain", "body"]:
            raise ValueError("Region must be 'brain' or 'body'")

        # validate sub-configs
        self.denoise.validate()
        self.motion.validate()
        # self.distortion.validate()  # WIP: Currently disabled, will crash
        # if provided paths don't exist (default set to empty)

    def to_dict(self):
        return {
            "region": self.region,
            "denoise": vars(self.denoise),
            "motion": vars(self.motion),
            "signalvoid": self.signal_void,
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
        cfg = cls.default()

        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            elif hasattr(cfg.denoise, key):
                setattr(cfg.denoise, key, value)
            elif hasattr(cfg.motion, key):
                setattr(cfg.motion, key, value)
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

        return cls(
            region=raw.get("region", "brain"),
            denoise=denoise,
            motion=motion,
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
