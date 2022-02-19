from rocketpy import Environment, SolidMotor, Rocket, Flight, Function
from datetime import datetime
from time import process_time, perf_counter, time
import glob

import numpy as np
from numpy.random import normal, uniform, choice
from IPython.display import display


# ------------------------------------------------------------------------ Environment Setting
Env = Environment(
    railLength=12,
    latitude=-23.363611,
    longitude=-48.011389,
    elevation=668
)

# import datetime

# tomorrow = datetime.date.today() + datetime.timedelta(days=1)

# Env.setDate((tomorrow.year, tomorrow.month, tomorrow.day, 12)) # Hour given in UTC time

# Env.setAtmosphericModel(type='Forecast', file='GFS')
# Env.setAtmosphericModel(type='StandardAtmosphere')

# Env.info()


URL = 'https://rucsoundings.noaa.gov/get_raobs.cgi?data_source=RAOB&latest=latest&start_year=2019&start_month_name=Feb&start_mday=5&start_hour=12&start_min=0&n_hrs=1.0&fcst_len=shortest&airport=83779&text=Ascii%20text%20%28GSD%20format%29&hydrometeors=false&start=latest'

Env.setAtmosphericModel(type='NOAARucSounding', file=URL)

# --------------------------------------------------------------------------Motor Setting
Hypnos = SolidMotor(
    thrustSource="ThrustyBoi.eng",
    burnOut=14.3,#
    grainNumber=6,#
    grainSeparation=0.1/1000,#
    grainDensity=860,#
    grainOuterRadius=38.95/1000,#
    grainInitialInnerRadius=13.5/1000,#
    grainInitialHeight=140/1000,#
    nozzleRadius=19.85/1000,#
    throatRadius=8.8/1000,#
    interpolationMethod='linear'
)

# Hypnos.info()
#Motor Setting Pass
# ------------------------------------------------------------------------Initializing Rocket
SporadicImpulse = Rocket(
    motor=Hypnos,
    radius=156/2000,#
    mass=48.276,#
    inertiaI=17.081,#
    inertiaZ=0.00351,#
    distanceRocketNozzle=-2.03,#
    distanceRocketPropellant=-0.571,#
    powerOffDrag='powerOffDragCurve.csv',#
    powerOnDrag='powerOffDragCurve.csv'#
)

SporadicImpulse.setRailButtons([0.2, -0.5])
# help(Function)
# Rocket Pass

# -----------------------------------------------------------------------Aerodynamic Surfaces
NoseCone = SporadicImpulse.addNose(length=0.468, kind="vonKarman", distanceToCM=2.442)

FinSet = SporadicImpulse.addFins(4, span=0.15, rootChord=0.304, tipChord=0.152, distanceToCM=-0.906)

#Tail = SporadicImpulse.addTail(topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656)


# -----------------------------------------------------------------------Parachute Parameters 
def drogueTrigger(p, y):
    # p = pressure
    # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
    # activate drogue when vz < 0 m/s.
    return True if y[5] < 0 else False

def mainTrigger(p, y):
    # p = pressure
    # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
    # activate main when vz < 0 m/s and z < 800 m.  
    return True if y[5] < 0 and y[2] < 800 else False

Main = SporadicImpulse.addParachute('Main',
                            CdS=10.0,
                            trigger=mainTrigger,
                            samplingRate=105,
                            lag=1.5,
                            noise=(0, 8.3, 0.5))

Drogue = SporadicImpulse.addParachute('Drogue',
                              CdS=1.0,
                              trigger=drogueTrigger,
                              samplingRate=105,
                              lag=1.5,
                              noise=(0, 8.3, 0.5))

SporadicImpulse.parachutes.remove(Drogue)
SporadicImpulse.parachutes.remove(Main)


# ----------------------------------------------------------------------Flight Test
TestFlight = Flight(rocket=SporadicImpulse, environment=Env, inclination=85, heading=0)
TestFlight.allInfo()

# --------------------------------------------------------------------Test Monte Carlo


