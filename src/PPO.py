import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import torch  # For GPU check

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

class LunarLanderTrainer:
    def __init__(self, env_name="LunarLander-v3"):
        # Create multiple environments for parallel sampling
        num_envs = 4  # Decrease environment count and increase steps
        
        # Create vectorized environments
        env_fns = [lambda: gym.make(env_name) for _ in range(num_envs)]
        self.env = DummyVecEnv(env_fns)
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        
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
        
        # Learning rate schedule - decreases over time
        def lr_schedule(progress):
            return 3e-4 * (1.0 - progress)
        
        # Initialize PPO with improved hyperparameters
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=lr_schedule,  # Use scheduled learning rate
            n_steps=2048,              # Decrease environment count and increase steps
            batch_size=256,            # Increase batch size
            n_epochs=10,               
            gamma=0.99,                # Discount factor
            gae_lambda=0.95,           # GAE lambda parameter
            clip_range=0.2,            # PPO clipping parameter
            ent_coef=0.005,            # Slightly lower entropy for more exploitation
            vf_coef=0.5,               # Value function coefficient
            max_grad_norm=0.5,         # Gradient clipping
            verbose=1,
            tensorboard_log="./ppo_tensorboard/",
            device=device,             # Use GPU if available
            policy_kwargs=dict(
                # Larger network architecture
                net_arch=[dict(pi=[512, 256], vf=[512, 256])],  
                # Add activation function specification
                activation_fn=torch.nn.ReLU
            )
        )
        
        # Create directories for model saving
        self.models_dir = "trained_models/ppo"
        os.makedirs(self.models_dir, exist_ok=True)
        self.vec_norm_path = os.path.join(self.models_dir, "vec_normalize.pkl")

    def train(self, total_timesteps=500000, save_interval=50000):
        """Train the model with periodic saving"""
        print("Starting PPO training...")
        
        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_interval, 
            save_path=self.models_dir,
            name_prefix="ppo_lunar"
        )
        
        # Train the model
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        
        # Save the final model and normalization stats
        final_model_path = os.path.join(self.models_dir, "ppo_lunar_final")
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
    print("=== Lunar Lander with Optimized PPO ===")
    trainer = LunarLanderTrainer()
    
    while True:
        print("\n1. Train new model")
        print("2. Test existing model")
        print("3. Exit")
        choice = input("\nYour choice: ")
        
        if choice == "1":
            timesteps = int(input("Enter total timesteps for training (default: 500000): ") or "500000")
            trainer.train(total_timesteps=timesteps)
            
        elif choice == "2":
            model_path = os.path.join(trainer.models_dir, "ppo_lunar_final.zip")
            if not os.path.exists(model_path):
                # Try to find latest checkpoint
                model_files = [f for f in os.listdir(trainer.models_dir) if f.startswith("ppo_lunar") and f.endswith(".zip")]
                if model_files:
                    latest_model = max(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]) if x.split("_")[-1].split(".")[0].isdigit() else 0)
                    model_path = os.path.join(trainer.models_dir, latest_model)
            
            if os.path.exists(model_path) and os.path.exists(trainer.vec_norm_path):
                trainer.model = PPO.load(model_path)
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
