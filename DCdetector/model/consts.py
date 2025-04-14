import time

str_time = str(time.time())

def log(string):
    with open('f{str_time}.log', 'a') as f:
        f.write(string + '\n')
    f.close()