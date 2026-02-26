"""
Configuration and constants for Reality Fabric
Edge Case Analysis: All critical parameters are centralized for easy tuning
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
import firebase_admin
from firebase_admin import credentials, firestore
import logging

@dataclass
class GAMConfig:
    """Configuration for Generative Adversarial Market"""
    # Model architecture
    latent_dim: int = 100
    generator_hidden_dims: List[int] = (256, 512, 256)
    discriminator_hidden_dims: List[int] = (256, 128, 64)
    sequence_length: int = 100
    feature_dim: int = 5  # OHLCV
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 1000
    critic_iterations: int = 5
    lambda_gp: float = 10.0
    learning_rate: float = 0.0001
    
    # Stability parameters
    gradient_penalty_weight: float = 10.0
    wasserstein_weight: float = 1.0
    noise_std: float = 0.01  # Additive noise for robustness
    
    # Validation
    validation_fraction: float = 0.1
    early_stopping_patience: int = 50

@dataclass
class RegimeConfig:
    """Configuration for regime detection and generation"""
    n_regimes: int = 4  # Bull, Bear, Sideways, HighVol
    regime_minimum_samples: int = 100
    transition_smoothness: float = 0.3
    volatility_multipliers: List[float] = (0.5, 1.0, 2.0, 4.0)
    
    # Regime characteristics
    regime_features: List[str] = [
        'volatility', 
        'trend_strength', 
        'volume_profile',
        'correlation_structure'
    ]

class RealityFabricConfig:
    """Main configuration manager with Firebase integration"""
    
    def __init__(self, firebase_credentials_path: Optional[str] = None):
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Model configurations
        self.gam = GAMConfig()
        self.regime = RegimeConfig()
        
        # Path configurations
        self.data_dir = "./market_data"
        self.models_dir = "./trained_models"
        self.simulations_dir = "./generated_simulations"
        
        # Ensure directories exist
        self._initialize_directories()
        
        # Initialize Firebase if credentials provided
        self.firestore_client = None
        if firebase_credentials_path and os.path.exists(firebase_credentials_path):
            try:
                cred = credentials.Certificate(firebase_credentials_path)
                firebase_admin.initialize_app(cred)
                self.firestore_client = firestore.client()
                self.logger.info("Firebase Firestore initialized successfully")
            except Exception as e:
                self.logger.error(f"Firebase initialization failed: {str(e)}")
                self.firestore_client = None
        
        # Edge case: Validate configurations
        self._validate_configs()
    
    def _initialize_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        for directory in [self.data_dir, self.models_dir, self.simulations_dir]:
            os.makedirs(directory, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")
    
    def _validate_configs(self) -> None:
        """Validate all configuration parameters for consistency"""
        # Check GAM configuration
        assert self.gam.latent_dim > 0, "Latent dimension must be positive"
        assert self.gam.batch_size > 0, "Batch size must be positive"
        assert 0 < self.gam.validation_fraction < 1, "Validation fraction must be between 0 and 1"
        
        # Check regime configuration
        assert self.regime.n_regimes >= 2, "Must have at least 2 regimes"
        assert len(self.regime.volatility_multipliers) == self.regime.n_regimes, \
            "Volatility multipliers must match number of regimes"
        
        self.logger.info("All configurations validated successfully")
    
    def save_to_firestore(self, collection_name: str = "atsef_configs") -> bool:
        """Save configuration to Firebase Firestore for persistence"""
        if not self.firestore_client:
            self.logger.warning("Firestore client not initialized, skipping save")
            return False
        
        try:
            config_dict = {
                "gam_config": self.gam.__dict__,
                "regime_config": self.regime.__dict__,
                "paths": {
                    "data_dir": self.data_dir,
                    "models_dir": self.models_dir,
                    "simulations_dir": self.simulations_dir
                },
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            
            doc_ref = self.firestore_client.collection(collection_name).document("reality_fabric")
            doc_ref.set(config_dict)
            self.logger.info("Configuration saved to Firestore")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save config to Firestore: {str(e)}")
            return False