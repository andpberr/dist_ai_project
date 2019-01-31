import spade

class MyAgent(spade.Agent.Agent):
    def _setup(self):
        print("MyAgent starting . . .")

if __name__ == '__main__':
    a = MyAgent('agent@127.0.0.1','secret')
    a.start()
    print(dir(a))
    print("Stopping!")
    a.stop()
    print("Stopped")
