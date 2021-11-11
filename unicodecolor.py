class textformatter:
    """
    Contains variables that can be used in formatted strings to
    e.g. change the colour of the text. Also contains functions that
    use these variables to colour text.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def blue(self,string):
        return self.OKBLUE + string + self.ENDC

    def cyan(self,string):
        return self.OKCYAN + string + self.ENDC

    def green(self,string):
        return self.OKGREEN + string + self.ENDC

    def red(self,string):
        return self.FAIL + string + self.ENDC

    def yellow(self,string):
        return self.WARNING + string + self.ENDC

    def bold(self,string):
        return self.BOLD + string + self.ENDC

    def underline(self,string):
        return self.UNDERLINE + string + self.ENDC 
