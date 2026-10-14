class car:
    def __init__(self, mark, year, speed):
        self.mark = mark
        self.speed = speed
        self.year = year

    def set_speed(self, speed):
        if speed > 0:
            self.speed = speed    

    def __str__(self):
        return f'{self.speed}, {self.mark}, {self.year}'

    def describe(self):
        print(f'{self.speed}, {self.mark}, {self.year}')

f1 = car("bmw", "2005", 0)
print(f1)
    