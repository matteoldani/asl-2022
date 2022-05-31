#General imports:

import sys
import os
import math
from csv import DictReader
from xml.etree.ElementInclude import include

from matplotlib import rc
#rc('text', usetex=True) # this is if you want to use latex to print text. If you do you can create strings that go on labels or titles like this for example (with an r in front): r"$n=$ " + str(int(n))
from numpy import *
from pylab import *
import random
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.lines as lns
from scipy import stats
from matplotlib.patches import Polygon
import matplotlib.font_manager as fm

#font = fm.FontProperties(
#        family = 'Gill Sans', fname = 'GillSans.ttc')

included = ['bs1', 'bs2', 'opt47', 'opt41', 'opt31']

background_color =(0.85,0.85,0.85) #'#C0C0C0'    
dark_grey_color = (0.298, 0.298, 0.298)
grid_color = 'white' #FAFAF7'
matplotlib.rc('axes', facecolor = background_color)
matplotlib.rc('axes', edgecolor = grid_color)
matplotlib.rc('axes', linewidth = 1.2)
matplotlib.rc('axes', grid = True )
matplotlib.rc('axes', axisbelow = True)
matplotlib.rc('grid',color = grid_color)
matplotlib.rc('grid',linestyle='-' )
matplotlib.rc('grid',linewidth=0.7 )
matplotlib.rc('xtick.major',size =0 )
matplotlib.rc('xtick.minor',size =0 )
matplotlib.rc('ytick.major',size =0 )
matplotlib.rc('ytick.minor',size =0 )

def addPerfLine(peakPerf, label):
	#Peak performance line and text
	ax.axhline(y=peakPerf, linewidth=0.75, color='black')
	yCoordinateTransformed = (log(peakPerf)-log(Y_MIN))/(log(Y_MAX/Y_MIN))
	ax.text(0.76,yCoordinateTransformed+0.01, label+" ("+str(peakPerf)+" F/C)", fontsize=8, transform=ax.transAxes)


def addBWLine(BW, label):
	x = np.linspace(X_MIN, X_MAX)
	y = x*BW
	ax.plot(x, y, linewidth=0.75, color='black')
	yCoordinateTransformed = (log(X_MIN*BW)-log(Y_MIN))/(log(Y_MAX/Y_MIN))
	ax.text(0.01,yCoordinateTransformed+0.0075*(len(str(BW))-1), label+' ('+str(BW)+' B/C)',fontsize=8, rotation=20, transform=ax.transAxes)

FREQ = 3500000000
GIGA = 1000000000

def gflops_to_flops_per_cycle(gflops):
	return (gflops * GIGA) / FREQ

X_MIN=0.04
X_MAX=10
Y_MIN=0.1
Y_MAX=200.0

PEAK_PERF=[28.0, 56] # 
PEAK_PERF_LABELS=['Vector Add', 'Vector FMA'] # 
PEAK_BW=[24.3, 60.0, 97, 297]
PEAK_BW_LABELS = ['DRAM', 'L3', 'L2', 'L1']

for i in range(len(PEAK_PERF)):
	PEAK_PERF[i] = round((PEAK_PERF[i] * GIGA) / FREQ, 2)

for i in range(len(PEAK_BW)):
	PEAK_BW[i] = round((PEAK_BW[i] * GIGA) / FREQ, 2)

INVERSE_GOLDEN_RATIO=0.618
OUTPUT_FILE= "../../outputs/rooflinePlotTotal.pdf"
TITLE="Roofline Plot"
X_LABEL="Operational Intensity [Flops/Byte]"
Y_LABEL="Performance [Flops/Cycle]"
ANNOTATE_POINTS=1
AXIS_ASPECT_RATIO=log10(X_MAX/X_MIN)/log10(Y_MAX/Y_MIN)

colors=[(0.2117, 0.467, 0.216), (0.258, 0.282, 0.725), (0.776,0.0196,0.07),(1,0,1)  ,'#FF9900', '#00CED1' ]
fig = plt.figure(figsize=(10, 12))
# Returns the Axes instance
ax = fig.add_subplot(111)

#Log scale - Roofline is always log-log plot, so remove the condition if LOG_X
ax.set_yscale('log')
ax.set_xscale('log')

#formatting:
#ax.set_title(TITLE,fontsize=14,fontweight='bold')
ax.set_xlabel(X_LABEL, fontsize=12)
ax.set_ylabel(Y_LABEL, fontsize=12)

#x-y range
ax.axis([X_MIN,X_MAX,Y_MIN,Y_MAX])
ax.set_aspect(INVERSE_GOLDEN_RATIO*AXIS_ASPECT_RATIO)

# Manually adjust xtick/ytick labels when log scale
locs, labels = xticks()
minloc =int(log10(X_MIN))
maxloc =int(log10(X_MAX) +1)

newlocs = []
newlabels = []

for i in range(minloc,maxloc):
    newlocs.append(10**i)
    if 10**i <= 100:
        newlabels.append(str(10**i))
    else:
        newlabels.append(r'$10^ %d$' %i)

xticks(newlocs, newlabels)

locs, labels = yticks()
minloc =int(log10(Y_MIN))
maxloc =int(log10(Y_MAX) +1)
newlocs = []
newlabels = []

for i in range(minloc,maxloc):
   newlocs.append(10**i)
   if 10**i <= 100:
       newlabels.append(str(10**i))
   else:
       newlabels.append(r'$10^ %d$' %i)
yticks(newlocs, newlabels)

# Load the data 

data = dict()

with open('/home/asl/asl-2022-vgsteiger/docs/plotting/roofline/roofline_data.csv', 'r') as input_data:
	csv_reader = DictReader(input_data)
	for row in csv_reader:
		if row['opt'] not in data:
			data[row['opt']] = dict()
		data[row['opt']][int(row['size'])] = (float(row['optin']), float(row['perf']))

pp = []
ss=[]
for serie in data.keys():

	if serie not in included:
		continue

	xData = []
	yData = []
	
	for size in sorted(data[serie]):
		xData.append(data[serie][size][0])
		yData.append(data[serie][size][1])

	x=[]
	xerr_low=[]
	xerr_high = []
	yerr_high = []
	y = []
	yerr_low = []

	for xDataItem in xData:
		x.append(stats.scoreatpercentile(xDataItem, 50))
		xerr_low.append(stats.scoreatpercentile(xDataItem, 25))
		xerr_high.append(stats.scoreatpercentile(xDataItem, 75))	
	
	for yDataItem in yData:
		yDataItem = gflops_to_flops_per_cycle(yDataItem)
		y.append(stats.scoreatpercentile(yDataItem, 50))
		yerr_low.append(stats.scoreatpercentile(yDataItem, 25))
		yerr_high.append(stats.scoreatpercentile(yDataItem, 75)) 

	xerr_low = [a - b for a, b in zip(x, xerr_low)] 
	xerr_high = [a - b for a, b in zip(xerr_high, x)]
	yerr_low = [a - b for a, b in zip(y, yerr_low)]
	yerr_high = [a - b for a, b in zip(yerr_high, y)]

	ax.scatter(x[0], y[0], s=4,zorder=12,  color=dark_grey_color)
	ax.scatter(x[len(x)-1], y[len(y)-1],s=4, zorder=12, color=dark_grey_color)

	p, = ax.plot(x, y, '-') # , color=colors[i]
	pp.append(p)
	ss.append(serie)

	if ANNOTATE_POINTS:
		ax.annotate(sorted(data[serie].keys())[0],
        xy=(x[0], y[0]), xycoords='data',
        xytext=(+3, +1), textcoords='offset points', fontsize=8)

		ax.annotate(sorted(data[serie].keys())[len(data[serie].keys())-1],
        xy=(x[len(x)-1],y[len(y)-1]), xycoords='data',
        xytext=(+3, +1), textcoords='offset points', fontsize=8)

	ax.annotate(serie,
        xy=(x[int(len(x) / 2)], y[int(len(y) / 2)]), xycoords='data',
        xytext=(0, -12), textcoords='offset points', fontsize=8)

#Peak performance line and text
for p,l in zip(PEAK_PERF, PEAK_PERF_LABELS):
	addPerfLine(p,l)

#BW line and text
for bw,l in zip(PEAK_BW, PEAK_BW_LABELS):
	addBWLine(bw,l)

matplotlib.pyplot.grid(True, which="both")

#save file
fig.savefig(OUTPUT_FILE, dpi=250,  bbox_inches='tight')