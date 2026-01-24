import sys
from unittest.mock import MagicMock

# --- 1. SETUP THE MOCKS BEFORE IMPORTING ---
# Create the fake function
mock_infer_func = MagicMock()
# Set the return value here directly so it can't be missed
# Returns: Dp (D*), Dt (D), Fp (f), S0
mock_infer_func.return_value = (0.2, 0.001, 0.1, 100)

# Create the fake module structure
mock_super_ivim_dc = MagicMock()
mock_super_ivim_dc.__spec__ = MagicMock() # Satisfy importlib

mock_infer_module = MagicMock()
mock_infer_module.infer_from_signal = mock_infer_func

# Inject them into Python's memory
sys.modules["super_ivim_dc"] = mock_super_ivim_dc
sys.modules["super_ivim_dc.infer"] = mock_infer_module

# --- 2. NOW IMPORT THE WRAPPER ---
from src.standardized.TCML_TechnionIIT_SuperIVIMDC import TCML_TechnionIIT_SuperIVIMDC
import numpy as np
import pytest
from unittest.mock import patch

class TestSuperIVIMDC:
    
    def test_init(self):
        """Test that the class initializes correctly."""
        model = TCML_TechnionIIT_SuperIVIMDC(model_path="dummy.pt")
        assert model.ACCEPTED_DIMENSIONS == (2, 4)
        assert model.model_path == "dummy.pt"

    @patch("src.standardized.TCML_TechnionIIT_SuperIVIMDC.os.path.exists")
    def test_osipi_fit_reordering(self, mock_exists):
        """
        Test that the wrapper correctly calls the external library 
        and reorders the outputs.
        """
        # 1. Setup the Mock for File Existence
        mock_exists.return_value = True
        
        # 2. Setup the Wrapper & Dummy Data
        model = TCML_TechnionIIT_SuperIVIMDC(model_path="fake_model.pt")
        data = np.zeros((10, 5)) 
        bvalues = np.array([0, 50, 100, 200, 800])
        
        # 3. Run the Fit
        f, D_star, D, S0 = model.osipi_fit(data, bvalues)
        
        # 4. Verify the Logic
        # Ensure our wrapper called the external function
        mock_infer_func.assert_called_once()
        
        # CHECK REORDERING:
        # We forced the mock to return (0.2, 0.001, 0.1, 100) -> (D*, D, f, S0)
        # The wrapper SHOULD return (f=0.1, D*=0.2, D=0.001, S0=100)
        assert f == 0.1
        assert D_star == 0.2
        assert D == 0.001
        assert S0 == 100
        print("\nSUCCESS: Wrapper logic verified!")