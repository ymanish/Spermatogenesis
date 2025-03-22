import os
import re
path = "./results/"
unsucessful_file = open('run_again_param_file.txt', 'w')

for root,d_names,f_names in os.walk(path):
    count=1
    for i in f_names:
        print(i)
        if re.search("out$", i):
            f = open(path+i, 'r')
            try:
                if re.search("Finished ", f.readlines()[5]):
                    # print('ok')
                    f.close()

                else:

                    f = open(path+i, 'r')
                    lines = f.readlines()
                    param_id = lines[0].split('\n')[0]
                    ku = lines[1].split('\n')[0]
                    kw = lines[2].split('\n')[0]
                    ka = lines[3].split('\n')[0]
                    kd = lines[4].split('\n')[0]
                    print(param_id, ku, kw,ka, kd)
                    unsucessful_file.write(f'{param_id},{ku},{kw},{ka},{kd},\n')
                    f.close()
            except IndexError:
                f = open(path+i, 'r')
                lines = f.readlines()
                param_id = lines[0].split('\n')[0]
                ku = lines[1].split('\n')[0]
                kw = lines[2].split('\n')[0]
                ka = lines[3].split('\n')[0]
                kd = lines[4].split('\n')[0]
                print(param_id, ku, kw, ka, kd)
                unsucessful_file.write(f'{param_id},{ku},{kw},{ka},{kd},\n')
                f.close()
            #     count=count+1

