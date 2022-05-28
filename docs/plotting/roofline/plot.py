#General imports:

import sys
import os
import math
from matplotlib import rc
rc('text', usetex=True) # this is if you want to use latex to print text. If you do you can create strings that go on labels or titles like this for example (with an r in front): r"$n=$ " + str(int(n))
from numpy import *
from pylab import *
import random
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.lines as lns
from scipy import stats
from matplotlib.patches import Polygon
import matplotlib.font_manager as fm

font = fm.FontProperties(
        family = 'Gill Sans', fname = 'GillSans.ttc')


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
	x = np.linspace(X_MIN, X_MAX, X_MAX)
	y = x*BW
	ax.plot(x, y, linewidth=0.75, color='black')
	yCoordinateTransformed = (log(X_MIN*BW)-log(Y_MIN))/(log(Y_MAX/Y_MIN))+0.16 #0.16 is the offset of the lower axis
	ax.text(0.01,yCoordinateTransformed+0.05+0.0075*(len(str(BW))-1), label+' ('+str(BW)+' B/C)',fontsize=8, rotation=45, transform=ax.transAxes)

FREQ = 3500000000
GIGA = 1000000000

def gflops_to_flops_per_cycle(gflops):
	return (gflops * GIGA) / FREQ

#X_MIN=0.01
X_MIN=0.1
#X_MAX=100.0
X_MAX=100.0
#Y_MIN=0.1
Y_MIN=0.1
Y_MAX=200.0
#PEAK_PERF=8.0
#PEAK_BW=11.95

PEAK_PERF=[7.0, 28.0, 56]
PEAK_PERF_LABELS=['Scalar Add', 'Vector Add', 'Vector FMA']
PEAK_BW=[24.3, 60.0, 97, 297]
PEAK_BW_LABELS = ['DRAM', 'L3', 'L2', 'L1']

for i in range(len(PEAK_BW)):
	PEAK_BW[i] = (PEAK_BW[i] * GIGA) / FREQ

INVERSE_GOLDEN_RATIO=0.618
OUTPUT_FILE="rooflinePlotTotal.pdf"
TITLE="Roofline Plot"
X_LABEL="Operational Intensity [Flops/Byte]"
Y_LABEL="Performance [Flops/Cycle]"
ANNOTATE_POINTS=1
AXIS_ASPECT_RATIO=log10(X_MAX/X_MIN)/log10(Y_MAX/Y_MIN)

colors=[(0.2117, 0.467, 0.216), (0.258, 0.282, 0.725), (0.776,0.0196,0.07),(1,0,1)  ,'#FF9900', '#00CED1' ]
fig = plt.figure()
# Returns the Axes instance
ax = fig.add_subplot(111)

#Log scale - Roofline is always log-log plot, so remove the condition if LOG_X
ax.set_yscale('log')
ax.set_xscale('log')

#formatting:
ax.set_title(TITLE,fontsize=14,fontweight='bold')
ax.set_xlabel(X_LABEL, fontproperties = font, fontsize=12)
ax.set_ylabel(Y_LABEL, fontproperties = font, fontsize=12)

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

sizes = [80, 160, 320, 400, 640]
series = ['bs1', 'bs2', 'opt0', 'opt1', 'aopt1', 'aopt2', 'opt2', 'opt3', 'opt21', 'opt22', 'opt23', 'opt24', 'opt31', 'opt32', 'opt33', 'opt34']
dataTotal = [
	(),
	(),
	()
]

pp = []
ss=[]
for serie,i in zip(series,range(len(series))):

	xData = []
	yData = []
	
	for j in range(len(dataTotal[i])):
		xData.append(dataTotal[i][j][0])
		yData.append(dataTotal[i][j][1])

	x=[]
	xerr_low=[]
	xerr_high = []
	yerr_high = []
	y = []
	yerr_low = []

	for xDataItem in xData:
		xDataItem = gflops_to_flops_per_cycle(xDataItem)
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

	#print x
	#print xerr_low
	#print xerr_high
	#print y
	#print yerr_low
	#print yerr_high

	ax.scatter(x[0], y[0], s=4,zorder=12,  color=dark_grey_color)
	ax.scatter(x[len(x)-1], y[len(y)-1],s=4, zorder=12, color=dark_grey_color)

	p, =ax.plot(x, y, '-', color=colors[i],label=serie)
	pp.append(p)
	ss.append(serie)
	ax.errorbar(x, y, yerr=[yerr_low, yerr_high], xerr=[xerr_low, xerr_high], fmt='b.',elinewidth=0.4, ecolor = 'Black', capsize=0, color=colors[i])  

	if ANNOTATE_POINTS:
		ax.annotate(sizes[0],
        xy=(x[0], y[0]), xycoords='data',
        xytext=(+3, +1), textcoords='offset points', fontsize=8)

		ax.annotate(sizes[len(sizes)-1],
        xy=(x[len(x)-1],y[len(y)-1]), xycoords='data',
        xytext=(+3, +1), textcoords='offset points', fontsize=8)

# Work around to get rid of the problem with frameon=False and the extra space generated in the plot
ax.legend(pp,ss, numpoints=1, loc='best',fontsize =6).get_frame().set_visible(False)
#ax.legend(pp,ss, numpoints=1, loc='best',fontsize =6,frameon = False )





#Peak performance line and text
for p,l in zip(PEAK_PERF, PEAK_PERF_LABELS):
	addPerfLine(p,l)

#BW line and text
for bw,l in zip(PEAK_BW, PEAK_BW_LABELS):
	addBWLine(bw,l)

#save file
fig.savefig(OUTPUT_FILE, dpi=250,  bbox_inches='tight')