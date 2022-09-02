from datetime import date
from datetime import timedelta
from tqdm import tqdm
from tqdm import trange


def issymmetry(day):
    j, i, flag = len(day) - 1, 0, 1
    while i < j:
        if day[i] != day[j]:
            flag = 0
            break
        else:
            i += 1
            j -= 1
    if flag: print(day)
    return flag

#print(today)
#print(today + delta)
#print(today.strftime("%y%m%d"))
delta = timedelta(days=1)
today = date.today()
for day in trange(365*100):
    delta1 = timedelta(days=day+1)
    flag = issymmetry((today + delta1).strftime("%Y%m%d"))
    if flag: break

for day in trange(365*100):
    delta2 = timedelta(days=-day-1)
    flag = issymmetry((today + delta2).strftime("%Y%m%d"))
    if flag: break

t1 = date(2030, 3, 2)
t2 = date.today()

print((t1-t2).days))
