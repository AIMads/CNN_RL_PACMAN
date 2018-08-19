# CNN_RL_PACMAN
This is a CNN model that with help of reinforcement learning and deepQlearning manages, too navigate pacman around in the maze.


What i have used for this model is OpenAI's gym enviroment, where they offer all atari games and their controls. Then i have made a model with the use of keras for the CNN model that analyzes the enviroment which is the raw pixels, and keras-rl for the Reinforcement learning. I have used deepQlearning too this problem, it is based around the theory of having an enviorment in pacman thats the maze, and an agent which is pacman in this case. The enviroment sends a state too the agent, the agent then makes an action which results in a new state and a reward. The purpose is then too optimize the reward, and this goes on in a loop. 
Too run this model you need tensorflow: pip install tensorflow (for the gpu version then it is tensorflow-gpu)
You will also need keras and keras-rl both pip installs, if you use linux then you can get the OpenAI gym enviroment easy with: 
Pip install cmake and then pip install gym[atari] else with windows do pip install cmake, pip install gym and pip install atari-py
Make a folder where you can save your training file and then read from it in test phase.

You train the model by going too the directory where you have the model file, and then type:
python model.py --mode train    (for training)
python model.py --mode test     (for testing)

If you don't want too train the model i have included trained weights that can be downloaded :)
