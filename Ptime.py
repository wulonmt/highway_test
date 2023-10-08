import time

#make time eazier to read
class Ptime():
    def __init__(self):
        self.saved_time = ""
        self.ptime = ""
        self.month_number_dict = {
        'Jan': '01',
        'Feb': '02',
        'Mar': '03',
        'Apr': '04',
        'May': '05',
        'Jun': '06',
        'Jul': '07',
        'Aug': '08',
        'Sep': '09',
        'Oct': '10',
        'Nov': '11',
        'Dec': '12'
        }
    def set_time_now(self):
        self.saved_time = str(time.ctime())
        self.saved_origin = int(time.time())
        
    def get_origin_time(self):
        return self.saved_origin
        
    def get_time(self):
        time_list = self.saved_time.split(' ')
        if '' in time_list:
            time_list.remove('')
        if(int(time_list[2]) < 10):
            time_list[2] = "0" + time_list[2]
        time_list[1] = self.month_number_dict[time_list[1]]
        time_list[3] = time_list[3].replace(":",".")
        mask = [4, 1, 2, 3]
        self.ptime = ""
        for i in mask:
            self.ptime += time_list[i]
        return self.ptime
        
if __name__ == "__main__":
    t = Ptime()
    t.set_time_now()
    print("time now: ", t.get_time())
    print("time origin: ", t.get_origin_time())
