import os
# Fix for OpenMP duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Then your other imports
import gymnasium as gym
from stable_baselines3 import DQN  # Changed from SAC to DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import torch  # For GPU check

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

class LunarLanderTrainer:
    def __init__(self, env_name="LunarLander-v3"):  # Changed to discrete version
        # Create and normalize the environment (important for performance)
        self.env = DummyVecEnv([lambda: gym.make(env_name)])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        
        # Check for GPU availability with more detailed info
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n{'='*30}\nGPU STATUS")
        print(f"PyTorch sees CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Using: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("No GPU available, using CPU instead.")
        print(f"{'='*30}\n")
        
        # Force CUDA device if available
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU
            print(f"CUDA Device: {torch.cuda.current_device()}")
            print(f"Using: {torch.cuda.get_device_name(0)}")
        
        # Initialize DQN with optimized hyperparameters
        self.model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=1e-4,        # Learning rate
            buffer_size=100000,        # Replay buffer size
            batch_size=64,             # Batch size for training
            gamma=0.99,                # Discount factor
            exploration_fraction=0.2,  # Fraction of training to explore
            exploration_initial_eps=1.0,  # Initial exploration rate
            exploration_final_eps=0.05,   # Final exploration rate
            train_freq=4,              # Update the model every n steps
            gradient_steps=1,          # How many gradient steps per update
            learning_starts=1000,      # Collect this many steps before training
            target_update_interval=1000,  # Update target network every n steps
            verbose=1,
            tensorboard_log="./dqn_tensorboard/",
            device=device,             # Use GPU if available
            policy_kwargs=dict(
                net_arch=[256, 256]    # Network architecture
            )
        )
        
        # Create directories for model saving
        self.models_dir = "trained_models/dqn"  # Changed from sac to dqn
        os.makedirs(self.models_dir, exist_ok=True)
        self.vec_norm_path = os.path.join(self.models_dir, "vec_normalize.pkl")

    def train(self, total_timesteps=500000, save_interval=50000):
        """Train the model with periodic saving"""
        print("Starting DQN training...")  # Changed from SAC to DQN
        
        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_interval, 
            save_path=self.models_dir,
            name_prefix="dqn_lunar"  # Changed from sac to dqn
        )
        
        # Train the model
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        
        # Save the final model and normalization stats
        final_model_path = os.path.join(self.models_dir, "dqn_lunar_final")  # Changed from sac to dqn
        self.model.save(final_model_path)
        self.env.save(self.vec_norm_path)
        print(f"Training completed. Model saved to {final_model_path}")

    def evaluate(self, episodes=10, render=False):
        """Evaluate the model's performance"""
        # Create evaluation environment
        eval_env = gym.make("LunarLander-v3", render_mode="human" if render else None)
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # Load normalization stats
        eval_env = VecNormalize.load(self.vec_norm_path, eval_env)
        eval_env.training = False  # Don't update normalization stats during evaluation
        eval_env.norm_reward = False  # Don't normalize rewards during evaluation
        
        rewards = []
        for episode in range(episodes):
            obs = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                step_result = eval_env.step(action)
                
                # Handle both old and new gym API formats
                if len(step_result) == 4:  # Old format: obs, reward, done, info
                    obs, reward, done_array, _ = step_result
                    done = done_array[0]
                else:  # New format: obs, reward, terminated, truncated, info
                    obs, reward, terminated, truncated, _ = step_result
                    done = terminated[0] or truncated[0]
                
                episode_reward += reward[0]  # Unwrap the reward
            
            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        mean_reward = np.mean(rewards)
        print(f"\nAverage Reward over {episodes} episodes: {mean_reward:.2f}")
        return mean_reward

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'env'):
            self.env.close()

def main():
    print("=== Lunar Lander with Optimized DQN ===")  # Changed from SAC to DQN
    trainer = LunarLanderTrainer()
    
    # Print PyTorch and CUDA information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
    
    # Test tensor movement to GPU
    x = torch.rand(5, 3)
    if torch.cuda.is_available():
        y = x.cuda()
        print("Successfully moved tensor to GPU")
        print(y.device)  # Should show cuda:0
    else:
        print("CUDA not available")
    
    while True:
        print("\n1. Train new model")
        print("2. Test existing model")
        print("3. Exit")
        choice = input("\nYour choice: ")
        
        if choice == "1":
            timesteps = int(input("Enter total timesteps for training (default: 500000): ") or "500000")
            trainer.train(total_timesteps=timesteps)
            
        elif choice == "2":
            model_path = os.path.join(trainer.models_dir, "dqn_lunar_final.zip")  # Changed from sac to dqn
            if not os.path.exists(model_path):
                # Try to find latest checkpoint
                model_files = [f for f in os.listdir(trainer.models_dir) if f.startswith("dqn_lunar") and f.endswith(".zip")]  # Changed from sac to dqn
                if model_files:
                    latest_model = max(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]) if x.split("_")[-1].split(".")[0].isdigit() else 0)
                    model_path = os.path.join(trainer.models_dir, latest_model)
            
            if os.path.exists(model_path) and os.path.exists(trainer.vec_norm_path):
                trainer.model = DQN.load(model_path)  # Changed from SAC to DQN
                episodes = int(input("Enter number of test episodes (default: 10): ") or "10")
                render = input("Render environment? (y/n, default: n): ").lower() == "y"
                trainer.evaluate(episodes=episodes, render=render)
            else:
                print("No trained models found. Please train a model first.")
            
        elif choice == "3":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()