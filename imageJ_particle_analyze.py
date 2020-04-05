# -*- coding: utf-8 -*-
# @Time : 2020/3/30 下午4:23
# @Author : LuoLu
# @FileName: imageJ_particle_analyze.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

from ij import IJ
from inra.ijpb.morphology import Strel
from inra.ijpb.morphology import Morphology

# Open Image
imp = IJ.openImage("/Users/pdubois/Desktop/cont1_01_crop1.tif");
IJ.run(imp, "8-bit", "");
IJ.run(imp, "Set Scale...", "distance=0.44 known=1 pixel=1 unit=um");
IJ.run(imp, "Despeckle", "");

# Create structuring element
strel = Strel.Shape.DISK.fromRadius(5);
# Apply filter
ip = Morphology.closing(imp.getProcessor(), strel);
# Set result to current image (imp)
imp.setProcessor(ip);

IJ.setAutoThreshold(imp, "Minimum dark");
IJ.setThreshold(imp, 157, 255);
IJ.run(imp, "Analyze Particles...", "display clear include summarize add")
IJ.selectWindow("Results")
IJ.saveAs("Results", "/Users/pdubois/Desktop/Results.xls");
IJ.run("Close");
IJ.selectWindow("ROI Manager")
IJ.run("Close");
IJ.saveAs(imp, "Jpeg", "/Users/pdubois/Desktop/cont1_01_crop1_segm.jpg");
IJ.run("Close");