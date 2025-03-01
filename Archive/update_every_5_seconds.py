import threading
import time
import random
global_variable = 0

names = ['Alan_Tam','Michael_Hui','Jackie_Cheung']
name = 'Alan_Tam'
name_index = 0
def update_global():
    global name
    global name_index
    #name = random.choice(names)
    name_index = (name_index+1) % len(names)
    name = names[name_index]
    print(f"Updated global_variable: {name}")
    threading.Timer(3, update_global).start()

update_global()
print(global_variable)
