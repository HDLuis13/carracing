from custom_carracing import CustomCarRacing
env = CustomCarRacing()

env.reset()
for _ in range(1):
    env.render()
    env.step(3)
env.close()