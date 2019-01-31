import spade

#class MyAgent(spade.Agent.Agent):
#    class MyBehav(spade.Behaviour.PeriodicBehaviour):
#        def onStart(self):
#            print("Starting behavior . . .")
#            self.counter = 0
#
#        def _onTick(self):
#            print("Counter:{0}".format(self.counter))
#            self.counter += 1
#
#    def _setup(self):
#        print("MyAgent starting . . .")
#        b = self.MyBehav(1)
#        self.addBehaviour(b, None)


class MyAgent(spade.Agent.Agent):
    class MyBehav(spade.Behaviour.OneShotBehaviour):
        def onStart(self):
            print("Starting behavior . . .")

        def _process(self):
            print("Hello world from a OneShot")

        def onEnd(self):
            print("Ending behavior . . .")

    def _setup(self):
        print("MyAgent starting . . .")
        b = self.MyBehav()
        self.addBehaviour(b, None)


if __name__ == '__main__':
    a = MyAgent("agent@127.0.0.1","secret")
    a.start()
