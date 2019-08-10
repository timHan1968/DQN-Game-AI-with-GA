import time
import random
from tkinter import *
from collections import namedtuple
import pyautogui
import numpy as np
import math
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import dqn

import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


root = Tk()
root.title("Bounce v1.0")
root.geometry("500x500")
root.resizable(height=False, width=False)
root.wm_attributes("-topmost",1)
canvas = Canvas(width=500, height=500,bd=0, highlightthickness=0)
canvas.pack()
#print(root.wm_attributes())




rectX=[
            [5,100],
            [100,200],
            [200,300],
            [300,400],
            [400,495],
            [5,100],
            [100,200],
            [200,300],
            [300,400],
            [400,495],
            [5,100],
            [100,200],
            [200,300],
            [300,400],
            [400,495],
            [5,100],
            [100,200],
            [200,300],
            [300,400],
            [400,495],
            [5,100],
            [100,200],
            [200,300],
            [300,400],
            [400,495],
            ]

rectY=[
            [40,60],
            [40,60],
            [40,60],
            [40,60],
            [40,60],
            [60,80],
            [60,80],
            [60,80],
            [60,80],
            [60,80],
            [80,100],
            [80,100],
            [80,100],
            [80,100],
            [80,100],
            [100,120],
            [100,120],
            [100,120],
            [100,120],
            [100,120],
            [20,40],
            [20,40],
            [20,40],
            [20,40],
            [20,40]
            ]


def oldRect():
    oldX=[
                [5,100],
                [100,200],
                [200,300],
                [300,400],
                [400,495],
                [5,100],
                [100,200],
                [200,300],
                [300,400],
                [400,495],
                [5,100],
                [100,200],
                [200,300],
                [300,400],
                [400,495],
                [5,100],
                [100,200],
                [200,300],
                [300,400],
                [400,495],
                [5,100],
                [100,200],
                [200,300],
                [300,400],
                [400,495],
                ]
    oldY=[
                [40,60],
                [40,60],
                [40,60],
                [40,60],
                [40,60],
                [60,80],
                [60,80],
                [60,80],
                [60,80],
                [60,80],
                [80,100],
                [80,100],
                [80,100],
                [80,100],
                [80,100],
                [100,120],
                [100,120],
                [100,120],
                [100,120],
                [100,120],
                [20,40],
                [20,40],
                [20,40],
                [20,40],
                [20,40]
                ]
    return oldX, oldY


def drawBricks(x1,y1,x2,y2,i,color):
      canvas.create_rectangle(x1,y1,x2,y2,fill=color,width=2,tag="brick"+str(i))


def setupBricks():
    for i in range(0,25):
          j=0
          if i==2:
                color="yellow"
          elif i==19:
                color="turquoise"
          elif i==5:
                color="lawn green"
          elif i==11:
                color="moccasin"
          else:
                color= "firebrick"
          drawBricks(rectX[i][j], rectY[i][j], rectX[i][j+1], rectY[i][j+1],i,color)



def restoreBricks():
    rectXStart, rectYStart = oldRect()
    for i in range(0,25):
          canvas.delete("brick"+str(i))
          j=0
          if i==2:
                color="yellow"
          elif i==19:
                color="turquoise"
          elif i==5:
                color="lawn green"
          elif i==11:
                color="moccasin"
          else:
                color= "firebrick"
          drawBricks(rectXStart[i][j], rectYStart[i][j], rectXStart[i][j+1], rectYStart[i][j+1],i,color)


class Ball:
      def __init__(self,canvas,paddle,color, name):
            self.name = name
            self.color = color
            self.canvas=canvas
            self.paddle=paddle
            #self.position=canvas.create_oval(245,460,255,470,fill=color,tag=name)
            self.position=canvas.create_oval(242.5,457,257.5,472,fill=color,tag=name)
            angle=[-3,-2,-1,1,2,3]
            random.shuffle(angle)
            self.direction=angle[0]
            self.speed=-3
            self.startPaddleModTime=0
            self.startPaddleSpeedModTime=0
            self.paddleSpeedPower=True
            self.paddleSpeedPower1=True
            self.startBallSpeedModTime=0
            self.ballSpeedPower=True
            self.lifeLineMod=False
            self.groundHit=False


      def restoreBall(self):
            #self.position=canvas.create_oval(245,460,255,470,fill=color,tag=name)
            self.position=canvas.create_oval(242.5,457,257.5,472,fill=self.color,tag=self.name)
            angle=[-3,-2,-1,1,2,3]
            random.shuffle(angle)
            self.direction=angle[0]
            self.speed=-3
            self.startPaddleModTime=0
            self.startPaddleSpeedModTime=0
            self.paddleSpeedPower=True
            self.paddleSpeedPower1=True
            self.startBallSpeedModTime=0
            self.ballSpeedPower=True
            self.lifeLineMod=False
            self.groundHit=False

      def restorePaddle(self,color):
            paddleCoordinates=paddle.canvas.coords(paddle.position)
            canvas.delete("paddleInitial")
            paddleCoordinates[0]+=13
            paddleCoordinates[2]-=13
            paddle.position=paddle.canvas.create_rectangle(paddleCoordinates[0],paddleCoordinates[1],paddleCoordinates[2],paddleCoordinates[3],fill=color,tag="paddleInitial")

      def draw(self,brickX,brickY):
            if (self.groundHit):
                return
            global score
            global penalty_score
            self.canvas.move(self.position,self.direction,self.speed)
            currentPosition=self.canvas.coords(self.position)
            paddlePosition=self.canvas.coords(self.paddle.position)
            if currentPosition[1]<=0:
                  self.speed*=-1
            if currentPosition[3]>=500:
                  self.canvas.delete(self.name)
                  self.groundHit=True
                  # score-=30
                  # self.canvas.itemconfig(score_id, text = "Current score: "+str(score))
                  num = numBallsGone(balls)
                  if num == 1:
                      penalty_score-=10
                  elif num == 2:
                      penalty_score-=30
                  elif num == 3:
                      penalty_score-=1000
            if currentPosition[0]<=0:
                  self.direction*=-1
            if currentPosition[2]>=500:
                  self.direction*=-1
            if self.lifeLineMod==True and currentPosition[3]>=477:
                  print("In Mod If")
                  canvas.delete("LifeLineMod")
                  self.speed*=-1
                  #self.lifeLineMod=False
                  for ball in balls:
                      ball.lifeLineMod=False

            if currentPosition[2]>=paddlePosition[0] and currentPosition[0]<=paddlePosition[2] and currentPosition[3]>=paddlePosition[1] and currentPosition[3]>=paddlePosition[3]:
                  print("Collide")
                  self.speed*=-1

            for i in range(0,25):
                  if currentPosition[2]>=brickX[i][0] and currentPosition[0]<=brickX[i][1] and currentPosition[1]>=brickY[i][0] and currentPosition[3]<=brickY[i][1] :#and currentPosition[3]>=brickY[i][1]:
                        self.speed*=-1
                        canvas.delete("brick"+str(i))
                        '''
                        print(brickX[i+1][0])
                        print(brickX[i+1][1])
                        print(brickY[i+1][0])
                        print(brickY[i+1][1])
                        '''
                        brickX[i][0]=brickX[i][1]=brickY[i][0]=brickY[i][1]=0
                        score+=10
                        penalty_score+=10
                        self.canvas.itemconfig(score_id, text = "Current score: "+str(score))

                        if i==19:
                              self.canvas.create_line(0,476,500,476,width=2,fill="turquoise",tag="LifeLineMod")
                              #self.lifeLineMod=True
                              for ball in balls:
                                  ball.lifeLineMod=True

                        if i==5:
                              print("Ball Speed Mod")
                              currentPosition=canvas.coords(self.position)
                              self.canvas.delete(self.name)
                              self.position=canvas.create_oval(currentPosition[0],currentPosition[1],currentPosition[2],currentPosition[3],fill="red",tag=self.name)

                              self.speed=2
                              self.startBallSpeedModTime=time.time()

                        if i==11:
                              print("Paddle Speed Mod")
                              paddleCoordinates=paddle.canvas.coords(paddle.position)
                              canvas.delete("paddleInitial")
                              paddle.position=paddle.canvas.create_rectangle(paddleCoordinates[0],paddleCoordinates[1],paddleCoordinates[2],paddleCoordinates[3],fill="moccasin",tag="paddleInitial",width=2)
                              paddle.paddleSpeed=8
                              self.startPaddleSpeedModTime=time.time()

                        if i==2:
                              paddleCoordinates=paddle.canvas.coords(paddle.position)
                              canvas.delete("paddleInitial")
                              paddleCoordinates[0]-=13
                              paddleCoordinates[2]+=13
                              paddle.position=paddle.canvas.create_rectangle(paddleCoordinates[0],paddleCoordinates[1],paddleCoordinates[2],paddleCoordinates[3],fill="yellow",tag="paddleInitial",width=2)
                              self.startPaddleModTime=time.time()

                        break



class Paddle:
      def __init__(self,canvas,color):
            self.canvas=canvas
            self.paddleSpeed=4
            self.paddleSpeedL=self.paddleSpeed
            self.paddleSpeedR=self.paddleSpeed
            self.direction=0
            # self.x1=215
            # self.x2=285
            # self.y1=470
            # self.y2=475
            self.x1=200
            self.x2=300
            self.y1=470
            self.y2=480
            self.position=self.canvas.create_rectangle(self.x1,self.y1,self.x2,self.y2,fill=color,tag="paddleInitial")

            # self.canvas.bind_all('<KeyPress-Right>',self.moveRight)
            # self.canvas.bind_all('<KeyRelease-Right>',self.moveStop)
            #
            # self.canvas.bind_all('<KeyPress-Left>',self.moveLeft)
            # self.canvas.bind_all('<KeyRelease-Left>',self.moveStop)

      def restorePaddle(self):
            self.canvas.delete("paddleInitial")
            self.paddleSpeed=4
            self.paddleSpeedL=self.paddleSpeed
            self.paddleSpeedR=self.paddleSpeed
            self.direction=0
            # self.x1=215
            # self.x2=285
            # self.y1=470
            # self.y2=475
            self.x1=200
            self.x2=300
            self.y1=470
            self.y2=480
            self.position=self.canvas.create_rectangle(self.x1,self.y1,self.x2,self.y2,fill="black",tag="paddleInitial")

      def moveLeft(self):
            currentPosition=self.canvas.coords(self.position)
            if currentPosition[0]>=2:
                  self.direction=self.paddleSpeedL
                  self.direction*=-1
                  self.paddleSpeedR=self.paddleSpeed

      def moveRight(self):
            currentPosition=self.canvas.coords(self.position)
            if currentPosition[2]<=498:
                  self.direction=self.paddleSpeedR
                  self.direction*=1
                  self.paddleSpeedL=self.paddleSpeed

      def moveStop(self):
            self.direction=0

      def changeSize(self):
            # self.x1=200
            # self.x2=320
            self.x1=185
            self.x2=335

      def draw(self):
            self.canvas.move(self.position,self.direction,0)
            currentPosition=self.canvas.coords(self.position)
            if currentPosition[0]<=0:
                  self.paddleSpeedL=0
                  self.direction=0
            if currentPosition[2]>=500:
                  self.paddleSpeedR=0
                  self.direction=0



def allHitGround(balls):
    for b in balls:
        if b.groundHit != True:
            return False
    return True


def numBallsGone(balls):
    num = 0
    for b in balls:
        if b.groundHit:
            num += 1
    return num


def allBricksGone(brickX):
    for t in brickX:
        if t != [0,0]:
            return False
    return True


def play(action):
      #screen = getScreen()
      global EPISODES
      global score
      global penalty_score
      global startPaddleModTime
      global rectX
      global rectY

      # print("Game Started")
      # canvas.delete("pressed")
      reward = penalty_score
      done = False
      startTime = time.time()

      if action == 1:
          paddle.moveLeft()
      elif action == 2:
          paddle.moveRight()
      else:
          paddle.moveStop()

      while (time.time()-startTime < 0.05):
            currentTime=time.time()
            for ball in balls:
                ball.draw(rectX,rectY)

            paddle.draw()
            root.update_idletasks()
            root.update()
            time.sleep(0.01)

            for ball in balls:
                if (ball.groundHit):
                    continue

                if ((currentTime-ball.startBallSpeedModTime)>=5 and (currentTime-ball.startBallSpeedModTime)<=6) and ball.ballSpeedPower==True:
                    print("ball speed mod over!!!")
                    if ball.speed>0:
                        ball.speed=5
                    elif ball.speed<0:
                        ball.speed=-5
                    ball.ballSpeedPower=False

                if ((currentTime-ball.startPaddleSpeedModTime)>=5 and (currentTime-ball.startPaddleSpeedModTime)<=6) and ball.paddleSpeedPower1==True:
                    paddle.paddleSpeed=4
                    paddleCoordinates=paddle.canvas.coords(paddle.position)
                    canvas.delete("paddleInitial")
                    paddle.position=paddle.canvas.create_rectangle(paddleCoordinates[0],paddleCoordinates[1],paddleCoordinates[2],paddleCoordinates[3],fill="black",tag="paddleInitial",width=2)
                    ball.paddleSpeedPower1=False

                if ((currentTime-ball.startPaddleModTime)>=5 and (currentTime-ball.startPaddleModTime)<=6)and ball.paddleSpeedPower==True:
                    ball.restorePaddle("black")
                    ball.paddleSpeedPower=False

            if allHitGround(balls) or allBricksGone(rectX):
                # Reset the game for a new round
                if allHitGround(balls):
                    reward = -1000

                global INIT_SCREEN
                score = 0
                penalty_score = 0
                canvas.itemconfig(score_id, text = "Current score: "+str(score))
                startPaddleModTime = 0
                restoreBricks()
                paddle.restorePaddle()
                for ball in balls:
                    ball.restoreBall()
                rectX, rectY = oldRect()
                done = True

                root.update_idletasks()
                root.update()
                time.sleep(0.5)

                INIT_SCREEN = getScreen()
                break

      # if reward != -1000:
      #     reward = penalty_score
      screen = getScreen()
      print("reward: " + str(reward))

      return done, screen, reward

# imgCounter = 1
def getScreen(x1=0, y1=0, w=500, h=535):
    geo = root.geometry()
    g = str(geo)
    first = g.find("+")
    second = g.find("+", first+1)
    xoffset = int(g[first+1: second])
    yoffset = int(g[second+1:])
    img = pyautogui.screenshot(region=(xoffset+10, yoffset, w, h))
    # global imgCounter
    # img.save(r"C:\Users\Lector\Desktop\test\testShot"+str(imgCounter)+".png")
    # imgCounter+=1
    resize = T.Compose([T.Resize(100), T.ToTensor()])
    img = resize(img).unsqueeze(0).to(torch.device("cpu"))
    return img


startPaddleModTime=0
playerReady=False
setupBricks()

paddle=Paddle(canvas,"black")
num_balls = 3
balls = []
colors = ['blue', 'yellow', 'green', 'red']
score = 0
penalty_score = 0
score_id = canvas.create_text(70,140,text="Current score: 0",fill="orange", tag="score",font=("Times",14))
#imgCounter = 1

for i in range(num_balls):
    color = colors[random.randint(0, len(colors)-1)]
    ball=Ball(canvas,paddle,color, "ball"+str(i))
    balls.append(ball)


for ball in balls:
  ball.draw(rectX,rectY)

paddle.draw()
root.update_idletasks()
root.update()
time.sleep(0.5)

INIT_SCREEN = getScreen()
# canvas.create_text(250,250,text="Hit Spacebar when ready",tag="pressed",font=("Times",12))
# canvas.bind_all("<space>",startGame)

# Action selection and Optimization
# Should be moved to Game.py...
# ------------------------------------------------------------
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 200000
TARGET_UPDATE = 10000
LEARNING_FREQ = 1
LEARNING_START = 5000
MEMORY_SIZE = 10000
OPTIM_LR = 0.0025

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
device = torch.device("cpu")

policy_dqn = dqn.DQN(3).to(device)
target_dqn = dqn.DQN(3).to(device)
target_dqn.load_state_dict(policy_dqn.state_dict())
#policy_dqn.load_state_dict(torch.load(r"C:\Users\Lector\Desktop\Breakout\policy.pt"))
#target_dqn.load_state_dict(torch.load(r"C:\Users\Lector\Desktop\Breakout\target.pt"))
target_dqn.eval()

steps_done = 0
optim_done = 0

optimizer = torch.optim.RMSprop(policy_dqn.parameters(), lr=OPTIM_LR)
memory = dqn.Memory(MEMORY_SIZE)


def getAction(state):
	# Get epsilon
	global steps_done
	if steps_done < EPS_DECAY:
		eps = EPS_START - (EPS_START - EPS_END) * steps_done / EPS_DECAY
	else:
		eps = EPS_END
	steps_done += 1
	print("Steps done: " + str(steps_done))
	# Decide and return action as a tensor (use [item()] to get int when in "game.py")
	sample = random.random()
	if sample > eps:
		with torch.no_grad():
			return policy_dqn(state).max(1)[1].view(1,1)
	else:
		return torch.tensor([[random.randrange(3)]], device = device, dtype=torch.long)


episode_scores = []

def plot_scores():
    plt.figure(2)
    plt.clf()
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy())
    # Take 100 episode averages and plot them too
    if len(scores_t) >= 100:
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimizeModel(memory):
	if memory.getLength() < BATCH_SIZE:
		return

	if steps_done < LEARNING_START:
		return

	global optim_done
	# Sample and transpose memory data in batches
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))

	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Generate the target values using [target_dqn]
	next_state_batch = torch.cat([s for s in batch.next_state
		if s is not None])
	next_state_mask = torch.tensor(tuple(map(lambda s: s is not None,
		batch.next_state)), device=device, dtype=torch.uint8)

	# Customize max Q values in final state
	next_state_qValues = torch.zeros(BATCH_SIZE, device=device)
	next_state_qValues[next_state_mask] = target_dqn(next_state_batch).max(1)[0].detach()

	target_values = (next_state_qValues * GAMMA) + reward_batch

	# Calculating the current values using [policy_dqn]
	current_values = policy_dqn(state_batch).gather(1, action_batch)

# -----------------------------------------------------
	## May need to change this part using other loss functions...
	# bellman_error = target_values - current_values.squeeze()
	# d_error = -1. * bellman_error.clamp(-1, 1)
	loss = F.smooth_l1_loss(current_values, target_values.unsqueeze(1))

	optimizer.zero_grad()
	#current_values.backward(d_error.data.unsqueeze(1))
	loss.backward()
    ## Manually clipping dqn parameters?
	for param in policy_dqn.parameters():
		param.grad.data.clamp_(-1, 1)
# -----------------------------------------------------

	optimizer.step()
	torch.save(policy_dqn.state_dict(), r"C:\Users\Lector\Desktop\breakout\weights\policy.pt")

	# Optimize target network occasionally
	if steps_done % TARGET_UPDATE == 0:
		target_dqn.load_state_dict(policy_dqn.state_dict())
		torch.save(target_dqn.state_dict(), r"C:\Users\Lector\Desktop\breakout\weights\policy.pt")

	optim_done += 1
	print("Optim steps done: "+str(optim_done))


# Main training body
# Need to integrate with [game.py] somehow
# -----------------------------------------------------
EPISODES = 5000

for episode_i in range(EPISODES):
  current_screen = INIT_SCREEN
  # _, next_screen, _ = play(0)
  current_state = current_screen
  done = False
  latest_score = 0

  while (not done):
    latest_score = score
    action = getAction(current_state)
    done, next_screen, reward = play(action)
    reward = torch.tensor([reward], device=device, dtype=torch.float)

    if not done:
      next_state = next_screen
    else:
      next_state = None

    memory.remember(current_state, action, next_state, reward)
    optimizeModel(memory)

    current_state = next_state

    if done:
        #episode_scores.append(latest_score-30)
        episode_scores.append(latest_score)
        plot_scores()

  print("Current Episode: "+str(episode_i))

plt.ioff()
plt.show()

root.mainloop()
