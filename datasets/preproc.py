import pyift.pyift as ift
import os
import sys

if len(sys.argv) != 4:
    print("python preproc <folder with images> <new width> <new height>")
    exit()


folder = sys.argv[1]
os.system("ls -v {}/*.png > temp.txt".format(folder))
width  = float(sys.argv[2])
height = float(sys.argv[3])

f = open("temp.txt","r")
for line in f:
    line = line.strip()
    orig = ift.ReadImageByExt(line)
    img  = ift.Interp2D(orig,width/orig.xsize,height/orig.ysize)
    ift.WriteImageByExt(img, line)

f.close()    
