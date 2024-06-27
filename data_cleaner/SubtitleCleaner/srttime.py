class SrtTime:
    def __init__(self, miliseconds=0, seconds=0, minutes=0, hours=0):
        self.miliseconds = miliseconds
        self.seconds = seconds
        self.minutes = minutes
        self.hours = hours

    @staticmethod
    def getStrTime(time_string):
        hms, miliseconds = time_string.split(',')
        h, m, s = hms.split(':')
        h, m, s, miliseconds = int(h), int(m), int(s), int(miliseconds)
        return SrtTime(hours=h, minutes=m, seconds=s, miliseconds=miliseconds)

    @property
    def miliseconds(self):
        return self._miliseconds

    @property
    def seconds(self):
        return self._seconds

    @property
    def minutes(self):
        return self._minutes

    @property
    def hours(self):
        return self._hours

    @miliseconds.setter
    def miliseconds(self, value):
        if value < 0: raise ValueError("Value cannot be less than 0")
        self._miliseconds = value

    @seconds.setter
    def seconds(self, value):
        if value < 0: raise ValueError("Value cannot be less than 0")
        self._seconds = value

    @minutes.setter
    def minutes(self, value):
        if value < 0: raise ValueError("Value cannot be less than 0")
        self._minutes = value

    @hours.setter
    def hours(self, value):
        if value < 0: raise ValueError("Value cannot be less than 0")
        self._hours = value

    def __compare(self, other):
        if self.hours > other.hours:
            return 1
        elif self.hours < other.hours:
            return -1
        else:
            if self.minutes > other.minutes:
                return 1
            elif self.minutes < other.minutes:
                return -1
            else:
                if self.seconds > other.seconds:
                    return 1
                elif self.seconds < other.seconds:
                    return -1
                else:
                    if self.miliseconds > other.miliseconds:
                        return 1
                    elif self.miliseconds < other.miliseconds:
                        return -1
                    else:
                        return 0

    def __eq__(self, other):
        return self.__compare(other) == 0

    def __lt__(self, other):
        return self.__compare(other) == -1

    def __gt__(self, other):
        return self.__compare(other) == 1

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __str__(self):
        return f"{self.hours:0>2}:{self.minutes:0>2}:{self.seconds:0>2},{self.miliseconds:0>3}"
