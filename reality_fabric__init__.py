"""
Reality Fabric - Layer 1 of ATSEF
Generative Adversarial Market Simulation Engine
"""
from reality_fabric.generative_adversarial_market import GenerativeAdversarialMarket
from reality_fabric.regime_aware_generator import RegimeAwareGenerator
from reality_fabric.multi_resolution_synthesizer import MultiResolutionSynthesizer
from reality_fabric.simulator import MarketSimulator

__version__ = "1.0.0"
__all__ = [
    "GenerativeAdversarialMarket",
    "RegimeAwareGenerator",
    "MultiResolutionSynthesizer",
    "MarketSimulator"
]