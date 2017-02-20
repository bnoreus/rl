# This Python file uses the following encoding: utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys
from scipy.misc import imsave, imresize
from scipy.signal import lfilter
from moviepy.editor import VideoClip
import time
import random
from collections import deque
import gym

env = gym.make('Seaquest-v3')
num_actions = env.action_space.n


tf.set_random_seed(2016)
random.seed(2016)


# Config variables
batch_size = 1000

class ReplayBuffer:
	def __init__(self,max_len):
		self.data = []
		self.ptr = 0
		self.max_len = max_len
	def add(self,observation,action,reward,next_observation):
		if len(self.data) > self.max_len:
			self.ptr += 1
		self.data.append((observation,action,reward,next_observation))

		if self.ptr >= 10000:
			self.data = self.data[10000:]
			self.ptr -= 10000
			print "Data size is now",len(self.data)

	def size(self):
		return len(self.data)-self.ptr
	def sample(self,batch_size):
		batch = [self.data[random.randint(self.ptr,len(self.data)-1)] for _ in range(batch_size)]		
		observation_batch = np.array([_[0] for _ in batch])
		action_batch = np.array([_[1] for _ in batch])
		reward_batch = np.array([_[2] for _ in batch])
		next_observation_batch = np.array([_[3] for _ in batch])

		return observation_batch,action_batch,reward_batch,next_observation_batch

# FrameMaker creates the video 
class FrameMaker:
	def __init__(self,fps,frames):
		self.fps = fps
		self.frames = frames
	def duration(self):
		return float(len(self.frames)/self.fps)
	def make_frame(self,t):
		idx = int(t*self.fps)
		return self.frames[idx]

def discount(rewards,discount_rate):
	return lfilter([1],[1,-discount_rate],rewards[::-1])[::-1]

def downsample_image(img):
	img = img.astype(np.float32).mean(2)
	img = imresize(img,(110,84))
	img = img[18:102,:].astype(np.float32)
	img *= (1.0/255.0)
	img = np.reshape(img,[84,84,1])
	return img


class Model:
	def __init__(self):
		self.output_size = env.action_space.n
		self.critic_learning_rate = 0.01
		self.actor_learning_rate = 0.001
		# Gather all variables:
		num_variables = 0
		self.input_placeholder,self.state = self.convnet()
		self.convnet_variables = tf.trainable_variables()[num_variables:]
		num_variables += len(self.convnet_variables)
		self.action_softmax = self.actor(self.state)
		self.actor_variables = tf.trainable_variables()[num_variables:]
		num_variables += len(self.actor_variables)
		self.value_function,self.action_placeholder,self.actor_gradient_generator = self.critic(self.state)
		self.critic_variables = tf.trainable_variables()[num_variables:]

		# Optimize actor:
		
		self.actor_gradient_placeholder = tf.placeholder(tf.float32, [None, self.output_size])
		actor_gradients = tf.gradients(self.action_softmax, self.convnet_variables+self.actor_variables, -self.actor_gradient_placeholder)
		self.optimize_actor = tf.train.AdagradOptimizer(self.actor_learning_rate).apply_gradients(zip(actor_gradients,self.convnet_variables+self.actor_variables))

		# Optimize critic:
		self.reward_placeholder = tf.placeholder(tf.float32,shape=[None,1])
		self.value_loss = tf.reduce_mean(tf.square(self.reward_placeholder-self.value_function))
		self.optimize_critic = tf.train.AdagradOptimizer(self.critic_learning_rate).minimize(self.value_loss)

		# Start session:
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


	def train(self,observations,actions,rewards):
		actor_gradients,_ = self.sess.run([self.actor_gradient_generator,self.optimize_critic],
			feed_dict={self.input_placeholder:observations,self.action_placeholder:actions,self.reward_placeholder:rewards})
		actor_gradients = actor_gradients[0]
		self.sess.run(self.optimize_actor,feed_dict={self.input_placeholder:observations,self.actor_gradient_placeholder:actor_gradients})

	def actor(self,state):
		action_logits = slim.fully_connected(state,self.output_size,activation_fn=None,scope="fc_act")
		action_softmax = tf.nn.softmax(action_logits)

		return action_softmax
	def critic(self,state):
		batch_size = tf.cast(tf.shape(state)[0],tf.float32)
		action_placeholder = tf.placeholder(tf.float32,shape=[None,self.output_size])
		state_size = state.get_shape()[1]
		hidden_size = 64
		w_state_hidden = tf.get_variable("w_state_hidden",shape=[state_size,hidden_size])
		w_action_hidden = tf.get_variable("w_action_hidden",shape=[self.output_size,hidden_size])
		b = tf.get_variable("b_critic_hidden",shape=[hidden_size])
		hidden = tf.nn.relu(tf.matmul(state,w_state_hidden)+tf.matmul(action_placeholder,w_action_hidden)+b)

		out = slim.fully_connected(hidden,1,activation_fn=None,scope="out")

		gradient_generator = tf.div(tf.gradients(out,action_placeholder),batch_size)
		return out,action_placeholder,gradient_generator
	def convnet(self):
		hidden_1_size = 16
		hidden_2_size = 32
		hidden_3_size = 256
		x = tf.placeholder(tf.float32,[None,84,84,2],name="X") # [batch_size,height,width,time_lag]
		hidden_1 = slim.conv2d(x,hidden_1_size,[8,8],stride=4,padding="SAME",activation_fn=tf.nn.relu,scope="conv1")
		hidden_2 = slim.conv2d(hidden_1,hidden_2_size,[4,4],stride=2,padding="SAME",activation_fn=tf.nn.relu,scope="conv2")
		flat = slim.flatten(hidden_2)
		hidden_3 = slim.fully_connected(flat,hidden_3_size,activation_fn=tf.nn.relu,scope="fc_hidden")
		
		return x,hidden_3
	def predict_action(self,img):
		feed_dict={self.input_placeholder:np.expand_dims(img,0)}
		return self.sess.run(self.action_softmax,feed_dict)[0]



model = Model()

replay_buffer = ReplayBuffer(20000)
batch_size = 32
k = 4
output_size = env.action_space.n

for episode in xrange(10000):
	img = env.reset()
	img = downsample_image(img)
	img_old = img
	action = env.action_space.sample()
	episode_reward = 0
	
	# We need to buffer a little bit
	movie_images = []
	observations = []
	rewards = []
	actions = []
	observations_next = []
	noise_ratio = np.exp(-episode/1500.0)

	for step in xrange(1000):
		
		img_concat = np.concatenate((img,img_old),axis=2)
		img_old = img


		if step % k == 0:
			action = model.predict_action(img_concat)
			noise = np.random.rand(output_size)
			noise = noise / np.sum(noise)
			action = (1-noise_ratio)*action + noise_ratio*noise
			action /= np.sum(action)
			action_idx = np.argmax(action)
		else:
			pass # Just use the old prediction. 
		
		img,reward,done,info = env.step(action_idx)
		if episode % 20 == 0:
			movie_images.append(img)
		episode_reward += reward

		img = downsample_image(img)
		if reward > 0.0:
			reward = 1.0
		elif reward < 0.0:
			reward = -1.0

		observations.append(img_concat)
		actions.append(action)
		rewards.append(reward)
		observations_next.append(np.concatenate((img,img_old),axis=2))


		
		if replay_buffer.size() > 20000 and step % k == 0:
			o_batch, a_batch, r_batch, s2_batch = replay_buffer.sample(batch_size)
			model.train(o_batch,a_batch,np.expand_dims(r_batch,1))

		if done:
			break
	if episode % 20 == 0:
		frame_maker = FrameMaker(fps=24,frames=movie_images)
		animation = VideoClip(frame_maker.make_frame,duration=frame_maker.duration())
		animation.write_videofile("animation_step_"+str(episode)+".mp4",fps=24)
		print "Animation saved!"
	
	discounted_rewards = discount(rewards,0.99)
	for i in range(len(observations)):
		replay_buffer.add(observations[i],actions[i],rewards[i],observations_next[i])
	print "\nEpisode ",episode, ".  Reward",episode_reward," Noise ratio=",noise_ratio
	"""
	tmp = list(np.mean(np.array(actions),axis=0))
	tmp = map(lambda x: round(x,3),tmp)
	tmp = zip(tmp,range(len(tmp)))
	tmp = sorted(tmp,key=lambda x:-x[0])[:10]
	tmp = " , ".join(map(lambda x: str(x[1])+"->"+str(x[0]),tmp))
	print tmp
	"""
