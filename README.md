# Deep Deterministic Policy Gradient (DDPG) algorithm on OpenAI's LunarLander

## Summary
&nbsp;&nbsp;&nbsp;&nbsp;The goal of this application is to implement **DDPG algorithm**[[paper]](https://arxiv.org/pdf/1509.02971.pdf) on [Open AI LunarLanderContinuous enviroment](https://gym.openai.com/envs/LunarLanderContinuous-v2/).
  
![LunarLander Gif008](images/ll-ep8.gif) 
![LunarLander Gif125](images/ll-ep125.gif)
![LunarLander Gif240](images/ll-ep240.gif)

*DDPG: Episode 8 vs Episode 125 vs Episode 240*

## Environment
&nbsp;&nbsp;&nbsp;&nbsp;[LunarLanderContinuous](https://gym.openai.com/envs/LunarLanderContinuous-v2/) is [OpenAI Box2D enviroment](https://gym.openai.com/envs/#box2d) which corresponds to the rocket trajectory optimization which is a classic topic in Optimal Control. LunarLander enviroment contains the rocket and terrain with landing pad which is generated randomly. The lander has three engines: left, right and bottom. Goal is to, using these engines, land somewhere on landing pad with using as less fuel as possible. Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.

![LunarLander Enviroment](images/ll_env.png)

*LunarLander Enviroment [Image source](https://shiva-verma.medium.com/teach-your-ai-how-to-walk-5ad55fce8bca)*

&nbsp;&nbsp;&nbsp;&nbsp;State consists of the horizontal coordinate, the vertical coordinate, the horizontal speed, the vertical speed, the angle, the angular speed, 1 if first leg has contact, else 0, 1 if second leg has contact, else 0

&nbsp;&nbsp;&nbsp;&nbsp;Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points. If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points. Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame. Solved is 200 points.

&nbsp;&nbsp;&nbsp;&nbsp;Action is two real values vector from -1 to +1. First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power. Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.

&nbsp;&nbsp;&nbsp;&nbsp;The episode ends when the lander **lands on the terrain**. Goal is reached when algorithm achieves ***mean score of 200 or higher on last 100 episodes (games)***.

## Deep Deterministic Policy Gradient
&nbsp;&nbsp;&nbsp;&nbsp;Short summary, input output, why is different than  algorithms before, explain deterministic and mix of DQN and AC.

![DDPG algorithm](images/ddpg_algo.png)
*DDPG algorithm*

## Continuous action implementation and random noise function
&nbsp;&nbsp;&nbsp;&nbsp; TODO

## Improving DDPG
&nbsp;&nbsp;&nbsp;&nbsp; TODO

## Testing
&nbsp;&nbsp;&nbsp;&nbsp; To get accurate results, algorithm has additional class (test process) whose job is to occasionally test 100 episodes and calculate mean reward of last 100 episodes. By the rules, if test process gets 200 or higher mean score over last 100 games, goal is reached and we should terminate. If goal isn't reached, training process continues. Testing is done every 50,000 steps or when mean of last 40 returns is 200 or more.

## Results
&nbsp;&nbsp;&nbsp;&nbsp;One of the results can be seen on graph below, where X axis represents number of episodes in algorithm and Y axis represents episode reward, mean training return and mean test return (return = mean episode reward over last 100 episodes). Keep in mind that for goal to be reached mean test return has to reach 200.

![Results graph](images/results.png)

- ![#33bbee](https://via.placeholder.com/15/33bbee/000000?text=+) `Episode reward`
- ![#359a3c](https://via.placeholder.com/15/359a3c/000000?text=+) `Mean training return`
- ![#ee3377](https://via.placeholder.com/15/ee3377/000000?text=+) `Mean test return`

* During multiple runs, **mean test return is over 200**, therefore we can conclude that **goal is reached!**

&nbsp;&nbsp;&nbsp;&nbsp;Additional statistics

* **Fastest run reached the goal after TODO enviroment steps**.
* **Highest reward in a single episode achieved is TODO**.

## Rest of the data and TensorBoard
&nbsp;&nbsp;&nbsp;&nbsp; If you wish to use trained models, there are saved NN models in [/models](/models). You will have to modify `load.py` PATH parameters and run the script to see results of training.

&nbsp;&nbsp;&nbsp;&nbsp; **If you dont want to bother with running the script, you can head over to the [YouTube](TODO) or see best recordings in [/recordings](/recordings).**

&nbsp;&nbsp;&nbsp;&nbsp;Rest of the training data can be found at [/content/runs](/content/runs). If you wish to see it and compare it with the rest, I recommend using TensorBoard. After installation simply change the directory where the data is stored, use the following command
  
```python
LOG_DIR = "full\path\to\data"
tensorboard --logdir=LOG_DIR --host=127.0.0.1
```
and open http://localhost:6006 in your browser.
For information about installation and further questions visit [TensorBoard github](https://github.com/tensorflow/tensorboard/blob/master/README.md)
  


