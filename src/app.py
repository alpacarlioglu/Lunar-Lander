import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
import numpy as np
import os

class LunarLanderTrainer:
    def __init__(self, env_name="LunarLander-v2", algo="PPO"):
        self.env = gym.make(env_name)
        self.algo = algo.upper()
        
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
        elif self.algo == "A2C":
            self.model = A2C(
                "MlpPolicy",
                self.env,
                learning_rate=1e-4,
                n_steps=5,
                gamma=0.999,
                ent_coef=0.01,
                verbose=1
            )
        else:
            raise ValueError("Unsupported algorithm. Choose among PPO, DQN, or A2C.")
        
        # Create directories for model saving
        self.models_dir = f"trained_models/{self.algo.lower()}"
        os.makedirs(self.models_dir, exist_ok=True)

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

    def evaluate(self, episodes=10):
        """Evaluate the model's performance"""
        rewards = []
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        return np.mean(rewards)

    def __del__(self):
        """Clean up resources"""
        self.env.close()

def main():
    print("=== Lunar Lander with Deep RL Algorithms ===")
    print("1. Train new model")
    print("2. Test existing model")
    print("3. Exit")
    
    algo_choice = input("Choose algorithm (PPO/DQN/A2C, default: PPO): ") or "PPO"
    trainer = LunarLanderTrainer(algo=algo_choice)
    
    while True:
        choice = input("\nYour choice: ")
        
        if choice == "1":
            timesteps = int(input("Enter total timesteps for training (default: 100000): ") or "100000")
            trainer.train(total_timesteps=timesteps)
            chc = int(input("Wanna try different algorithm?\n If yes press 0 else 1\n"))
            if chc == 0:
                main()
            
        elif choice == "2":
            if os.path.exists(trainer.models_dir):
                # Load the latest model
                model_files = [f for f in os.listdir(trainer.models_dir) if f.startswith(f"{trainer.algo.lower()}_lunar")]
                if model_files:
                    latest_model = max(model_files, key=lambda x: int(x.split("_")[-1].replace(".zip", "")))
                    model_path = os.path.join(trainer.models_dir, latest_model)
                    if trainer.algo == "PPO":
                        trainer.model = PPO.load(model_path, env=trainer.env)
                    elif trainer.algo == "DQN":
                        trainer.model = DQN.load(model_path, env=trainer.env)
                    elif trainer.algo == "A2C":
                        trainer.model = A2C.load(model_path, env=trainer.env)
                    print(f"Loaded model: {latest_model}")
                    
                    episodes = int(input("Enter number of test episodes (default: 10): ") or "10")
                    mean_reward = trainer.evaluate(episodes=episodes)
                    print(f"\nAverage Reward over {episodes} episodes: {mean_reward:.2f}")
                else:
                    print("No trained models found. Please train a model first.")
            else:
                print("No trained models found. Please train a model first.")
            
        elif choice == "3":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
