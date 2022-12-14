import numpy as np
import time

# Progress bar
# Initialize printer before use
# Then update the printer using the print function in every iteration
# Finally just let it kill itself or call printer.finish() to release the print flush
class Printer:
    def __init__(self, total_fr, description = " ", print_every = 0.2, len_bar = 40):
        self.total_fr = total_fr
        self.print_every = print_every
        self.len_bar = len_bar
        self.lpl = 0
        self.__enabled = True
        self.description = description
        if description[-1] != " ":
            self.description += " "
        self.t = 0
        self.last_t = 0
        self.last_desc = ""
    
    def finish(self):
        self.print(self.total_fr, self.last_desc)
        self.__enabled = False
        self.lpl = 0
        print()
    
    def print(self, fr, description = "data"):
        if not self.__enabled:
            return
        
        if self.last_t == 0:
            self.last_t = time.time()
        
        if fr >= self.total_fr - 1 or self.t >= self.print_every:
            ratio = round((fr + 1)/self.total_fr * self.len_bar)
            st = self.description + description + ": [" + ratio * "=" + (self.len_bar - ratio) * " " + "]  " + str(fr) + "/" + str(self.total_fr)
            print("\b" * self.lpl + st, end = "", flush = True)
            self.t = 0
            self.lpl = len(st)
        else:
            t = time.time()
            self.t += t - self.last_t
            self.last_t = t
            self.last_desc = description

    def __del__(self):
        self.finish()

