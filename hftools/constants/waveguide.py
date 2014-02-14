# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
from collections import namedtuple
from hftools.utils import to_numeric

#TODO: skaffa referens f�r denna data

waveguides = \
    """EIA	RCSC	IEC	f0 [GHz]	f1 [GHz]	fcl [GHz]	fcu [GHz]	a [in]	b [in]
WR650	WG6	R14	1.15	1.72	0.908	1.816	6.500	3.250
WR510	WG7	R18	1.45	2.20	1.157	2.314	5.100	2.550
WR430	WG8	R22	1.72	2.60	1.372	2.745	4.300	2.150
WR340	WG9A	R26	2.20	3.30	1.736	3.471	3.400	1.700
WR284	WG10	R32	2.60	3.95	2.078	4.156	2.840	1.340
WR229	WG11A	R40	3.30	4.90	2.577	5.154	2.290	1.145
WR187	WG12	R48	3.95	5.85	3.153	6.305	1.872	0.872
WR159	WG13	R58	4.90	7.05	3.712	7.423	1.590	0.795
WR137	WG14	R70	5.85	8.20	4.301	8.603	1.372	0.622
WR112	WG15	R84	7.05	10.00	5.26	10.52	1.122	0.497
WR90	WG16	R100	8.20	12.40	6.557	13.114	0.900	0.400
WR75	WG17	R120	10.00	15.00	7.869	15.737	0.750	0.375
WR62	WG18	R140	12.40	18.00	9.488	18.976	0.622	0.311
WR51	WG19	R180	15.00	22.00	11.572	23.143	0.510	0.255
WR42	WG20	R220	18.00	26.50	14.051	28.102	0.420	0.170
WR34	WG21	R260	22.00	33.00	17.357	34.715	0.340	0.170
WR28	WG22	R320	26.50	40.00	21.077	42.154	0.280	0.140
WR22	WG23	R400	33.00	50.00	26.346	52.692	0.224	0.112
WR19	WG24	R500	40.00	60.00	31.391	62.782	0.188	0.094
WR15	WG25	R620	50.00	75.00	39.875	79.75	0.148	0.074
WR12	WG26	R740	60.00	90.00	48.373	96.746	0.122	0.061
WR10	WG27	R900	75.00	110.00	59.015	118.03	0.100	0.050
WR8	WG28	R1200	90.00	140.00	73.768	147.536	0.080	0.040
WR7	WG29	R1400	112.00	172.00	90.791	181.583	0.0650	0.0325
WR5	WG30	R1800	140.00	220.00	115.714	231.429	0.0510	0.0255
WR4	WG31	R2200	172.00	260.00	137.243	274.485	0.0430	0.0215
WR3	WG32	R2600	220.00	330.00	173.571	347.143	0.0340	0.0170"""

WaveGuide = namedtuple("WaveGuide", "EIA RCSC IEC f0 f1 fcl fcu a b")

WR = dict()

for rad in waveguides.strip().split("\n")[1:]:
    x = map(to_numeric, rad.split())
    x[-1] *= 25.4
    x[-2] *= 25.4
    x[3] *= 1e9
    x[4] *= 1e9
    x[5] *= 1e9
    x[6] *= 1e9
    wg = WaveGuide(*x)
    WR[wg.EIA] = wg
