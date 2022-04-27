from afccp.core.comprehensive_functions import *


class MoreCCPMethods:

    def new_method(self):
        print('This is a new method for the CadetCareerProblem class')
        print('I can access all of the stuff from the other one and create new methods')
        print('N:', self.parameters['N'])  # Pycharm isn't going to like this, but this will work once CCP is defined
