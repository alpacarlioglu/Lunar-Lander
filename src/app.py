import gymnasium as gym
from stable_baselines3 import PPO, DQN, SAC  # Changed from A2C to SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # Add this line
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import glob
import re

class LunarLanderTrainer:
    def __init__(self, env_name=None, algo="PPO"):
        # Set appropriate environment based on algorithm
        if algo.upper() == "SAC" and env_name is None:
            # SAC requires continuous action space
            env_name = "LunarLanderContinuous-v3"
        elif env_name is None:
            # Default for discrete algorithms
            env_name = "LunarLander-v3"
            
        self.env = gym.make(env_name)
        self.algo = algo.upper()
        self.env_name = env_name
        
        # Initialize model based on chosen algorithm with optimized hyperparameters
        if self.algo == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=1e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.999,
                gae_lambda=0.98,
                ent_coef=0.01,
                verbose=1
            )
        elif self.algo == "DQN":
            self.model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=1e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=64,
                gamma=0.999,
                train_freq=4,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_final_eps=0.05,
                verbose=1
            )
        elif self.algo == "SAC":  # Changed from A2C to SAC
            self.model = SAC(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                ent_coef="auto",
                train_freq=1,
                gradient_steps=1,
                learning_starts=1000,
                verbose=1
            )
        else:
            raise ValueError("Unsupported algorithm. Choose among PPO, DQN, or SAC.")  # Changed A2C to SAC
        
        # Create directories for model saving
        self.models_dir = f"trained_models/{self.algo.lower()}"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Dictionary to store rewards during training for plotting
        self.training_rewards = []

    def train(self, total_timesteps=100000, save_interval=10000):
        """Train the model with periodic saving"""
        print(f"Starting training with {self.algo}...")
        
        timesteps_elapsed = 0
        while timesteps_elapsed < total_timesteps:
            # Train for save_interval timesteps
            current_steps = min(save_interval, total_timesteps - timesteps_elapsed)
            self.model.learn(total_timesteps=current_steps)
            timesteps_elapsed += current_steps
            
            # Save model checkpoint
            save_path = os.path.join(self.models_dir, f"{self.algo.lower()}_lunar_{timesteps_elapsed}")
            self.model.save(save_path)
            
            # Evaluate current performance
            mean_reward = self.evaluate(episodes=5)
            print(f"Timesteps: {timesteps_elapsed}/{total_timesteps}")
            print(f"Mean Reward: {mean_reward:.2f}")
            
            # Store reward for plotting
            self.training_rewards.append((timesteps_elapsed, mean_reward))

    def evaluate(self, episodes=10, render=False):
        """Evaluate the model's performance"""
        rewards = []
        
        env = self.env
        if render:
            env = gym.make(self.env_name, render_mode="human")
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        return np.mean(rewards)

    def __del__(self):
        """Clean up resources"""
        self.env.close()

def find_available_models():
    """Find all available trained models for each algorithm"""
    algorithms = ["ppo", "dqn", "sac"]
    available_models = {}
    
    # First, print what directories we're checking
    print("\nLooking for model files in:")
    for algo in algorithms:
        model_dir = f"trained_models/{algo}"
        print(f" - {model_dir}")
        
        if os.path.exists(model_dir):
            print(f"   Directory exists. Files found: {os.listdir(model_dir)}")
            
            # Be more flexible with model file detection - accept any .zip file
            model_files = glob.glob(os.path.join(model_dir, "*.zip"))
            
            if model_files:
                models = []
                for model_path in model_files:
                    # Extract timesteps from the filename, or use file modification time if no number found
                    match = re.search(r'(\d+)\.zip$', model_path)
                    if match:
                        timesteps = int(match.group(1))
                    else:
                        # Use modification time as a fallback
                        timesteps = int(os.path.getmtime(model_path))
                        
                    models.append((os.path.basename(model_path), timesteps))
                
                # Sort by timesteps
                models.sort(key=lambda x: x[1])
                available_models[algo] = models
                print(f"   Found {len(models)} model files")
            else:
                print("   No .zip files found")
        else:
            print("   Directory does not exist")
    
    # Also check in the main trained_models directory for misplaced files
    if os.path.exists("trained_models"):
        print("\nChecking main trained_models directory:")
        for file in os.listdir("trained_models"):
            file_path = os.path.join("trained_models", file)
            if os.path.isfile(file_path) and file.endswith(".zip"):
                print(f" - Found: {file}")
                
    return available_models

def compare_algorithms(render=False):
    """Compare the performance of all three algorithms"""
    print("Comparing PPO, DQN, and SAC performance...")
    
    available_models = find_available_models()
    algorithms = ["ppo", "dqn", "sac"]
    
    # Check if we have models for each algorithm
    missing_algos = [algo for algo in algorithms if algo not in available_models or not available_models[algo]]
    if missing_algos:
        print(f"Missing trained models for: {', '.join(missing_algos)}")
        print("Please train all algorithms first.")
        return
    
    # Evaluate best model from each algorithm
    results = {}
    for algo in algorithms:
        # Get the latest model (highest timesteps)
        latest_model = available_models[algo][-1]
        model_name, timesteps = latest_model
        
        print(f"\nEvaluating {algo.upper()} ({timesteps} timesteps)...")
        
        # Create appropriate environment based on algorithm
        env_name = "LunarLanderContinuous-v3" if algo == "sac" else "LunarLander-v3"
        
        # Create vectorized environment (important for normalization)
        eval_env = gym.make(env_name, render_mode="human" if render else None)
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # Try to load normalization statistics if they exist
        vec_norm_path = os.path.join(f"trained_models/{algo}", "vec_normalize.pkl")
        if os.path.exists(vec_norm_path):
            print(f"Loading normalization stats from {vec_norm_path}")
            eval_env = VecNormalize.load(vec_norm_path, eval_env)
            eval_env.training = False  # Don't update normalization stats during evaluation
            eval_env.norm_reward = False  # Don't normalize rewards during evaluation
        
        # Load model
        model_path = os.path.join(f"trained_models/{algo}", model_name)
        
        if algo == "ppo":
            model = PPO.load(model_path, env=eval_env)
        elif algo == "dqn":
            model = DQN.load(model_path, env=eval_env)
        else:  # SAC
            model = SAC.load(model_path, env=eval_env)
        
        # Evaluate
        rewards = []
        episodes = 10
        for episode in range(episodes):
            obs = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions!
                obs, reward, done_array, _ = eval_env.step(action)
                
                episode_reward += reward[0]  # Unwrap the reward
                done = done_array[0]  # Unwrap done
            
            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        mean_reward = np.mean(rewards)
        print(f"Average Reward: {mean_reward:.2f}")
        results[algo.upper()] = (mean_reward, rewards)
        
        eval_env.close()

    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Bar chart for average rewards
    plt.subplot(1, 2, 1)
    algos = list(results.keys())
    mean_rewards = [results[algo][0] for algo in algos]
    bars = plt.bar(algos, mean_rewards, color=['blue', 'green', 'red'])
    plt.title('Average Reward Comparison')
    plt.ylabel('Reward')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar, reward in zip(bars, mean_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                 f'{reward:.1f}', ha='center', va='bottom')
    
    # Box plot for reward distributions
    plt.subplot(1, 2, 2)
    reward_data = [results[algo][1] for algo in algos]
    plt.boxplot(reward_data, labels=algos)
    plt.title('Reward Distribution')
    plt.ylabel('Reward')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    plt.show()
    
    print(f"Comparison plot saved to 'algorithm_comparison.png'")

def main():
    print("=== Lunar Lander with Deep RL Algorithms ===")
    
    while True:
        print("\n1. Train new model")
        print("2. Test existing model")
        print("3. Compare algorithms")
        print("4. Exit")
        
        choice = input("\nYour choice: ")
        
        if choice == "1":
            algo_choice = input("Choose algorithm (PPO/DQN/SAC, default: PPO): ") or "PPO"
            trainer = LunarLanderTrainer(algo=algo_choice)
            timesteps = int(input("Enter total timesteps for training (default: 100000): ") or "100000")
            trainer.train(total_timesteps=timesteps)
            
        elif choice == "2":
            # Find all available models
            available_models = find_available_models()
            
            if not available_models:
                print("No trained models found. Please train a model first.")
                continue
                
            # Display available algorithms
            print("\nAvailable algorithms:")
            for i, algo in enumerate(available_models.keys(), 1):
                print(f"{i}. {algo.upper()}")
            
            algo_idx = int(input("\nSelect algorithm (number): ")) - 1
            if algo_idx < 0 or algo_idx >= len(available_models):
                print("Invalid selection.")
                continue
                
            selected_algo = list(available_models.keys())[algo_idx]
            
            # Display available models for the selected algorithm
            print(f"\nAvailable {selected_algo.upper()} models:")
            models = available_models[selected_algo]
            for i, (model_name, timesteps) in enumerate(models, 1):
                print(f"{i}. {model_name} ({timesteps} timesteps)")
                
            model_idx = int(input("\nSelect model (number): ")) - 1
            if model_idx < 0 or model_idx >= len(models):
                print("Invalid selection.")
                continue
                
            selected_model, _ = models[model_idx]
            model_path = os.path.join(f"trained_models/{selected_algo}", selected_model)
            
            # Load and evaluate the selected model
            env_name = "LunarLanderContinuous-v3" if selected_algo == "sac" else "LunarLander-v3"
            env = gym.make(env_name)
            
            if selected_algo == "ppo":
                model = PPO.load(model_path, env=env)
            elif selected_algo == "dqn":
                model = DQN.load(model_path, env=env)
            else:  # SAC
                model = SAC.load(model_path, env=env)
                
            trainer = LunarLanderTrainer(algo=selected_algo)
            trainer.model = model
            trainer.env = env
            
            episodes = int(input("Enter number of test episodes (default: 10): ") or "10")
            render = input("Render environment? (y/n, default: n): ").lower() == "y"
            mean_reward = trainer.evaluate(episodes=episodes, render=render)
            print(f"\nAverage Reward over {episodes} episodes: {mean_reward:.2f}")
            
        elif choice == "3":
            render = input("Render environment during evaluation? (y/n, default: n): ").lower() == "y"
            compare_algorithms(render=render)
            
        elif choice == "4":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
