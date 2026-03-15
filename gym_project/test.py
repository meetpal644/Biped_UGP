import torch
from stable_baselines3 import PPO
from pathlib import Path
from torchviz import make_dot

# ============= CONFIGURATION =============
BASE_DIRECTORY = Path.cwd()
MODEL_PATH = BASE_DIRECTORY / "models" / "best_model.zip"

print("\nLoading model...")
model = PPO.load(MODEL_PATH, device="cpu")

# 1. Create dummy observation (4 features as per your model logs)
obs = torch.randn(1, 4)

# 2. Extract features (This is the FlattenExtractor part)
features = model.policy.features_extractor(obs)

# 3. Pass through the MLP Extractor to get latents
# This matches the 'mlp_extractor' in your printed model architecture
latent_pi, latent_vf = model.policy.mlp_extractor(features)

# 4. Get final outputs
# We'll visualize the 'actions' path (Policy Net)
actions = model.policy.action_net(latent_pi)

# 5. Generate and Render
params = dict(model.policy.named_parameters())
dot = make_dot(actions, params=params, show_attrs=True, show_saved=True)
dot.format = "png"
dot.render("model_arch.png")

print("Success! 'model_arch.png' has been generated.")