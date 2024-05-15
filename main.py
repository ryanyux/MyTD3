import argparse
import gymnasium as gym
from buffer import *
from models import *
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

torch.set_printoptions(profile='full')

def eval(env, actor):
    eval_env = gym.make(env, continuous=True)
    s, _ = eval_env.reset()
    
    avg_reward = 0
    for _ in range(50):
        a = actor(torch.from_numpy(s))
        s, r, done, _ = eval_env.step(a.detach().numpy())
        avg_reward += r
        
        if done:
            s, _ = eval_env.reset()
    
    return avg_reward / 50

def main():
    parser = argparse.ArgumentParser()              # Policy name (TD3, DDPG or OurDDPG)
    #parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser.add_argument("--env", default="LunarLander-v2")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--episode", default=1e6, type=int)# Time steps initial random policy is used
    parser.add_argument("--warmup", default=1e2, type=int)
    parser.add_argument("--eval_freq", default=2e2, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=512, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=5, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
 
 
 
 
    env = gym.make(args.env, 
                   #render_mode = "human",
                   continuous=True)
    ob_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    
    actor = Actor(ob_space, action_space)
    actor_target = Actor(ob_space, action_space)
    
    critic = Critic(ob_space, action_space)
    critic_target = Critic(ob_space, action_space)
    critic2 = Critic(ob_space, action_space)
    critic2_target = Critic(ob_space, action_space)
    
    soft_update(actor, actor_target, 1)
    soft_update(critic, critic_target, 1)
    soft_update(critic2, critic2_target, 1)
    
    actor_optim = Adam(actor.parameters(), lr=1e-4)
    critic_optim = Adam(critic.parameters(), lr=1e-2)
    critic2_optim = Adam(critic2.parameters(), lr=1e-2)
    
    state, _ = env.reset(seed=42)
    
    buffer = ReplayBuffer(10000, ob_space, action_space)
    

    
    bar = tqdm(range(int(args.episode)))
    for step in bar:
        if step < args.warmup:
            action = env.action_space.sample()
        else:
            action = actor(torch.from_numpy(state))
            noise = np.random.normal(0, env.action_space.high * args.expl_noise, size=action_space,)
            action = action.cpu().data.numpy() + noise
            action = np.float32(action)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
        next_state, reward, terminated, truncated , _ = env.step(action)
        
        done = terminated or truncated
        
        buffer.add(state, next_state, action, reward, int(done))
        
        state = next_state
        
        if done:
            state, _ = env.reset()
            done = False
        
        
        
        if step > args.warmup:
            s, ns, a, r, d = buffer.sample(args.batch_size)
            
 
            with torch.no_grad():
                next_action = actor_target(ns)
                noise = (
                        torch.randn_like(a) * args.policy_noise
                    ).clamp(-args.noise_clip, args.noise_clip)
			
                na = (
                    next_action + noise
                ).clamp(
                    torch.from_numpy(env.action_space.low), 
                    torch.from_numpy(env.action_space.high)
                    )
                target_q1 = critic_target(ns, na)
                target_q2 = critic2_target(ns, na)
                q = torch.min(target_q1, target_q2)
                q = r + (1-d) * args.discount * q
            
            q1, q2 = critic(s, a), critic2(s, a)
            
            loss = F.mse_loss(q1, q) + F.mse_loss(q2, q)
            
            critic_optim.zero_grad()
            critic2_optim.zero_grad()
            
            loss.backward()
            
            # if loss > 100:
            #     print(q1 - q)
            #     print(q2 - q)
            
            if step % 50 == 0:
                bar.set_description("reward = {:.2f} loss = {:.2f}".format(reward, loss.item()))
            
            critic_optim.step()
            critic2_optim.step()
            
            if step % args.policy_freq == 0:
                
                actor_loss = -critic(
                    s, actor(s)
                ).mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()
                
                soft_update(actor, actor_target, args.tau)
                soft_update(critic, critic_target, args.tau)
                soft_update(critic2, critic2_target, args.tau)
            

        # if step % int(args.eval_freq) == int(args.eval_freq) - 1:
        #     print(f"EVAL: {eval(args.env, actor)}")

def soft_update(from_param, to_param, tau=1):
    for param, target_param in zip(from_param.parameters(), to_param.parameters()):
        target_param.data.copy_(target_param * (1 - tau) + param * tau)
            
            
            
            

    



if __name__ == "__main__":
    main()