#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:55:39 2019

@author: jeevan
"""
from collections import deque
import numpy as np
from path import Env
from DuelingDQNPrioritizedReplay import DDDQNNet,Memory
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import transform# Help us to preprocess the frames
from collections import deque# Ordered collection with ends
import random                # Handling random number generation
import time                  # Handling time calculation
LOAD=False
LOAD_PREV_WEIGHTS=False
game=Env()  
state_dim = game.state_dim
action_dim= game.action_dim      

### MODEL HYPERPARAMETERS
state_size = [100,130,2]      # Our input is a stack of 2 frames hence 100x120x2 (Width, height, channels) 
action_size = 5#game.get_available_buttons_size()              # 7 possible actions
learning_rate =  0.00001      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 14500         # Total episodes for training
max_steps =200        # Max possible steps in an episode
batch_size = 32           

# FIXED Q TARGETS HYPERPARAMETERS 
max_tau = 20000 #Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.1            # minimum exploration probability 
decay_rate = 0.000001#05            # exponential decay rate for exploration prob

# Q LEARNING hyperparameters
gamma = 0.9               # Discounting rate

### MEMORY HYPERPARAMETERS
## If you have GPU change to 1million
memory_size = 50000       # Number of experiences the Memory can keep
stack_size=2
s2_stack_size=2
## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False
possible_actions=np.identity(5,dtype=int).tolist()

# Instantiate the DQNetwork
DQNetwork = DDDQNNet(state_dim, action_dim, learning_rate, name="DQNetwork")
# Instantiate the target network
TargetNetwork = DDDQNNet(state_dim, action_dim, learning_rate, name="TargetNetwork")

def stack_frames(stacked_frames, frame, is_new_episode):
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((100,130), dtype=np.int) for i in range(stack_size)], maxlen=2)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=0)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=0) 
    return stacked_state, stacked_frames

def stack_states(stacked_state, state, is_new_episode):
    
    if is_new_episode:
        stacked_state = deque([np.zeros((2)) for i in range(s2_stack_size)], maxlen=2)
        stacked_state.append(state)
        stacked_state.append(state)
        stacked_state = np.stack(stacked_state, axis=0)

    else:
        stacked_state=np.append(stacked_state,state)
#        print(stacked_state)
#        stacked_state=stacked_state.reshape(4)
#        print(stacked_state)
        stacked_state = np.stack(stacked_state, axis=0) 
    return stacked_state,stacked_state

def predict_action(sess,explore_start, explore_stop, decay_rate, decay_step, state, s2,actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        choice=np.random.randint(0,action_dim)
        action = possible_actions[choice]
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_1: state.reshape((1,100,130,2)),\
                                                     DQNetwork.inputs_2: s2.reshape((1,2))})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
                
    return action, explore_probability ,choice
    



# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani
def update_target_graph():
    
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []
    
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder



# Instantiate memory
memory = Memory(memory_size)
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("1")#("/tensorboard/dddqn/1")
## Losses
tf.summary.scalar("Loss", DQNetwork.loss)
write_op = tf.summary.merge_all()

# Saver will help us to save our model
def train():
    reward_avg=deque(maxlen=200)
    avg_reward_over_episodes=[]
    saver = tf.train.Saver()
    
    np.set_printoptions(precision=2)

    with tf.Session() as sess:
        if LOAD_PREV_WEIGHTS==True:
            saver.restore(sess, "./models/model.ckpt")
        else:
            sess.run(tf.global_variables_initializer())
        decay_step = 0
        tau = 0
        filled_memory_size=0
        update_target = update_target_graph() 
        sess.run(update_target)
        
        for episode in range(total_episodes):
            memory.PER_b = np.min([1., memory.PER_b + memory.PER_b_increment_per_sampling])  # max = 1
            step = 0
            episode_rewards = []
            stacked_frames  =  deque([np.zeros((100,130), dtype=np.int) for i in range(stack_size)], maxlen=2) 
#            stacked_s2=deque([np.zeros(2) for i in range(s2_stack_size)], maxlen=2) 
            
            state,s2=game.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True)
#            s2, stacked_s2 = stack_states(stacked_s2, s2, True)
            
            loss=0
            while step < max_steps:
#                print(step)
                if (episode>1000) and (episode%500<20):
                    game.render()
#                if step%10==0:
#                    print("step=",step)
                step += 1
                tau += 1        # Increase the C step
                decay_step +=1  # Increase decay_step
                # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
                action, explore_probability,choice = predict_action(sess,explore_start, explore_stop,\
                                                                    decay_rate, decay_step, state,s2, possible_actions)
                # Do the action
                next_state,next_s2,reward,done = game.step(choice)
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
#                next_s2, stacked_s2 = stack_states(stacked_s2, s2, False)
                if game.end_episode==True :
                    total_reward = np.sum(episode_rewards)
                    reward_avg.append(total_reward)
                    print('Ep: {}'.format(episode),
                              'Steps:',step,
                              '|| Ret: {:.2f}'.format(total_reward),
                              '|| Loss: {:.4f}'.format(loss),
                              '|| epsi: {:.2f}'.format(explore_probability),
                              '|| beta: {:.2f}'.format(memory.PER_b))
                    break
                
                episode_rewards.append(reward)
                experience = state,s2, action,reward, next_state,next_s2, done
                memory.store(experience)
                filled_memory_size+=1
                print_step=step
                if done or (step==max_steps-1):
                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    reward_avg.append(total_reward)
                    print('Ep: {}'.format(episode),
                              '|| Steps:',print_step,
                              '|| Ret: {:.2f}'.format(total_reward),
                              '|| Loss: {:.4f}'.format(loss),
                              '|| epsi: {:.2f}'.format(explore_probability),
                              '|| beta: {:.2f}'.format(memory.PER_b))

                else:
                    state = next_state
                    s2=next_s2

                if filled_memory_size>memory_size:
                    ### LEARNING PART            
                    # Obtain random mini-batch from memory
                    tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
#                    print("1",ISWeights_mb)
                    
                    states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                    s2_mb = np.array([each[0][1] for each in batch])
                    actions_mb = np.array([each[0][2] for each in batch])
                    rewards_mb = np.array([each[0][3] for each in batch]) 
                    next_states_mb = np.array([each[0][4] for each in batch], ndmin=3)
                    next_s2_mb = np.array([each[0][5] for each in batch])
                    dones_mb = np.array([each[0][6] for each in batch])
#                    print("Action",actions_mb)
                    target_Qs_batch = []
                    ### DOUBLE DQN Logic
                    # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                    # Use TargetNetwork to calculate the Q_val of Q(s',a')
                    states_mb=next_states_mb.reshape((-1,100,130,2))
                    s2_mb=next_s2_mb.reshape((-1,2))
                    next_states_mb=next_states_mb.reshape((-1,100,130,2))
                    next_s2_mb=next_s2_mb.reshape((-1,2))
                    # Get Q values for next_state 
                    q_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_1: next_states_mb,DQNetwork.inputs_2: next_s2_mb})
                    
                    # Calculate Qtarget for all actions that state
                    q_target_next_state = sess.run(TargetNetwork.output, feed_dict = {TargetNetwork.inputs_1: next_states_mb,TargetNetwork.inputs_2: next_s2_mb})
                    
                    
                    # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a') 
                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]
                        
                        # We got a'
                        action = np.argmax(q_next_state[i])
    
                        # If we are in a terminal state, only equals reward
                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])
                            
                        else:
                            # Take the Qtarget for action a'
                            target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                            target_Qs_batch.append(target)
                            
    
                    targets_mb = np.array([each for each in target_Qs_batch])
#                    print("2",targets_mb.shape)
#                    print("Done",done)
#                    print("Weighgt",ISWeight s_mb[0].shape)
#                    print("Training")
                    _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                        feed_dict={DQNetwork.inputs_1: states_mb,
                                                   DQNetwork.inputs_2: s2_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb, 
                                                   DQNetwork.ISWeights_: ISWeights_mb})
                    
                    # Update priority
                    memory.batch_update(tree_idx, absolute_errors)

                    # Write TF Summaries
                    summary = sess.run(write_op, feed_dict={DQNetwork.inputs_1: states_mb,
                                                            DQNetwork.inputs_2: s2_mb,
                                                       DQNetwork.target_Q: targets_mb,
                                                       DQNetwork.actions_: actions_mb,
                                                  DQNetwork.ISWeights_: ISWeights_mb})
                    writer.add_summary(summary, episode)
                    writer.flush()
                    
                    if tau > max_tau:
                        # Update the parameters of our TargetNetwork with DQN_weights
                        update_target = update_target_graph()
                        sess.run(update_target)
                        tau = 0
                        print("Target Model updated")
    
                # Save model every 5 episodes
            if episode % 1000 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")
                
            
            avg_reward_over_episodes.append(np.average(reward_avg))
            if episode % 50==0:
                plt.plot(avg_reward_over_episodes)    
                plt.show()

    def eval_():
        saver.restore(sess, "./models/model.ckpt")
        with tf.Session() as sess:
            for i in range(10):
                step=0
                stacked_frames  =  deque([np.zeros((100,130), dtype=np.int) for i in range(stack_size)], maxlen=2) 
                stacked_s2  =  deque([np.zeros((100,130), dtype=np.int) for i in range(stack_size)], maxlen=2) 
                state=game.reset()
                state, stacked_frames = stack_frames(stacked_frames, state, True)
                s2, stacked_s2 = stack_states(stacked_s2, state, True)
                done=False
                while not (done or game.end_episode):
                    step += 1
                    Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_1: state.reshape((1, *state.shape)),\
                                                                 DQNetwork.inputs_2: s2_mb,})
                    choice = np.argmax(Qs)
                    next_state,reward,done=game.step(choice)
                    next_state, stacked_frames = stack_frames(stacked_frames, state, False)
                    next_s2, stacked_s2 = stack_states(stacked_s2, state, True)
                    if done or game.end_episode:
                        episode_rewards.append(reward)
                        break  
                        
                    else:
                        episode_rewards.append(reward)
                        state = next_state
                
                total_reward = np.sum(episode_rewards)
                print('Ep: {}'.format(episode),
                              'Steps:',step,
                              'Return: {:.2f}'.format(total_reward),)

if __name__=="__main__":
    
    if LOAD:
        eval_()
        
    else:
        train()
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
