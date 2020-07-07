# print("hi")
# import osmnx as ox
# ox.plot_graph(ox.graph_from_place('Modena, Italy'))
import numpy
# from pcraster import *
# from pcraster.numpy import *

# import osmnx as ox
# import matplotlib.pyplot as plt

# place_name = "Kamppi, Helsinki, Finland"
# graph = ox.graph_from_place(place_name)
# type(graph)

import gridmap
#from gridmap import OccupancyGridMap
import matplotlib.pyplot as plt

#gmap = OccupancyGridMap.from_png('NewYork_0_512.png', 1)
#print(gmap)

from PIL import Image
import numpy as np
import pickle
# Open the maze image and make greyscale, and get its dimensions
im = Image.open('NewYork_0_256.png').convert('L')
w, h = im.size
print(w)
print(h)
# Ensure all black pixels are 0 and all white pixels are 1
binary = im.point(lambda p: p > 128 and 1)

# Resize to half its height and width so we can fit on Stack Overflow, get new dimensions
binary = binary.resize((w // 2, h // 2), Image.NEAREST)
w, h = binary.size

# Convert to Numpy array - because that's how images are best stored and processed in Python
nim = np.array(binary)

'''# Print that puppy out
for r in range(h):
    for c in range(w):
        print(nim[r, c], end='')
    print()'''

im = Image.open('NewYork_1_1024.png').convert('L')
w, h = im.size
print(w)
print(h)
# Ensure all black pixels are 0 and all white pixels are 1
binary = im.point(lambda p: p > 128 and 1)

# Resize to half its height and width so we can fit on Stack Overflow, get new dimensions
binary = binary.resize(((w // 8)+1, (h // 8)+1), Image.NEAREST)
w, h = binary.size

# Convert to Numpy array - because that's how images are best stored and processed in Python
nim = np.array(binary)
#print(len(nim))


#np.where((nim==0)|(nim==1), nim^1, nim)
print(nim)
print(len(nim))
B = np.where(nim> 0.5, 0, 1)
# Print that puppy out
for r in range(h):
    for c in range(w):
        print(nim[r, c], end=',')
    print()

with open("mazes_citymap.pkl", "wb") as f:
    pickle.dump(B, f)