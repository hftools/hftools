=======
hftools
=======

Hftools is a python package containing useful tools for high frequency 
electrical engineering problems. Hftools is focused on generally useful tools
such as file readers, multiport classes, an improved array class (hfarray),
dataset class. More specialized tools will be broken out into seprate packages.
One such package is caltools which contains tools for doing Vector Network 
Analyzer (VNA) calibrations.


hfarray
=======
The hftools package is built around the hfarray object which is an extension
to the regular numpy ndarray object. The extension adds a axis descriptor 
objects that identify the purpose of each dimension of an array, e.g. frequency
axis, repeated measurements axis, voltage sweep axis, or matrix row/column axis.
The axis descriptor objects are used to align arrays for broadcasting. The axis
descriptor object can also contain information about units and preferred
formatting when converting to text.

Plotting
========
The hftools.plotting package adds new projections to matplotlib to simplify
plotting of complex numbers, e.g. db, real, imag, mag, groupdelay. Some
automation has also been included for the axis labels. If the first axis
in a plotted hfarray has the unit Hz then the x-axis will be labelled 
"Frequency [Hz]".

