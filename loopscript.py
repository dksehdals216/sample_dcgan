import os


for i in range(10):
    print("at loop: " + str(i+1))
    print("Epoch size: " + str(100 * (i+1)))
    os.system("CUDA_VISIBLE_DEVCIES=2 python3 dcgan.py -ep " + str(100 * (i+1)))
