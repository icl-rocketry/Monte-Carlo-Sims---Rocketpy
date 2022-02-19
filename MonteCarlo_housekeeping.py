from datetime import datetime
from time import process_time, perf_counter, time
import glob

from rocketpy import Environment, SolidMotor, Rocket, Flight, Function

import numpy as np
from numpy.random import normal, uniform, choice
from IPython.display import display


# %config InlineBackend.figure_formats = ['svg']
import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
mpl.rcParams['figure.figsize'] = [8, 5]
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.titlesize'] = 14


analysis_parameters = {
    # Mass Details
    "rocketMass": (48.276, 0.001), # Rocket's dry mass (kg) and its uncertainty (standard deviation)

    # Propulsion Details - run help(SolidMotor) for more information
    # Sporadic Impulse values of 12/02/22 (Everything)
    "impulse": (14070, 500),                         # Motor total impulse (N*s) +-500N
    "burnOut": (14.3, 1),                              # Motor burn out time (s)
    "nozzleRadius": (19.85/1000, 0.5/1000),            # Motor's nozzle radius (m) 
    "throatRadius": (8.8/1000, 0.5/1000) ,                # Motor's nozzle throat radius (m) 8.8mm 17.6 diam
    "grainSeparation": (0.1/1000, 1/1000),                # Motor's grain separation (axial distance between two grains) (m) 0.1mm
    "grainDensity": (860, 50),                         # Motor's grain density (kg/m^3) 860kgm-3
    "grainOuterRadius": (38.95/1000, 0.375/1000),        # Motor's grain outer radius (m) 77.9/2mm
    "grainInitialInnerRadius": (13.5/1000, 0.375/1000), # Motor's grain inner radius (m) 27/2mm
    "grainInitialHeight": (140/1000, 1/1000),           # Motor's grain height (m) 14cm

    # Aerodynamic Details - run help(Rocket) for more information
    # Sporadic Impulse values of 12/02/22 (Everything)
    "inertiaI": (17.018, 0.0001),                    # Rocket's inertia moment perpendicular to its axis (kg*m^2) (Solid Cylinder approximation)
    "inertiaZ": (0.14, 0.0001),                      # Rocket's inertia moment relative to its axis (kg*m^2) (Solid Cylinder approximation)
    "radius": (156/1000, 0.001),                     # Rocket's radius (kg*m^2)
    "distanceRocketNozzle": (-2.03,0.001),           # Distance between rocket's center of dry mass and nozzle exit plane (m) (negative)
    "distanceRocketPropellant": (-0.571,0.001),      # Distance between rocket's center of dry mass and and center of propellant mass (m) (negative)
    "powerOffDrag": (1, 0.033),                      # Multiplier for rocket's drag curve. Usually has a mean value of 1 and a uncertainty of 5% to 10% (unchanged for Sporadic Impulse)
    "powerOnDrag": (1, 0.033),                       # Multiplier for rocket's drag curve. Usually has a mean value of 1 and a uncertainty of 5% to 10% (unchanged for Sporadic Impulse)
    "noseLength": (0.468, 0.001),                    # Rocket's nose cone length (m)
    "noseDistanceToCM": (2.442, 0.001),              # Axial distance between rocket's center of dry mass and nearest point in its nose cone (m)
    "finSpan": (0.15, 0.0005),                       # Fin span (m) 150mm, 
    "finRootChord": (0.304, 0.0005),                 # Fin root chord (m) 304mm
    "finTipChord": (0.152, 0.0005),                  # Fin tip chord (m) 152mm
    "finDistanceToCM": (-0.906, 0.001),              # Axial distance between rocket's center of dry mass and nearest point in its fin (m)

    # Launch and Environment Details - run help(Environment) and help(Flight) for more information
    # Sporadic Impulse values of 12/02/22 (Everything)
    "inclination": (85, 1),                           # Launch rail inclination angle relative to the horizontal plane (degrees)
    "heading": (90, 2),                                 # Launch rail heading relative to north (degrees)
    "railLength": (12 , 0.0005),                       # Launch rail length (m)
    "ensembleMember": list(range(10)),                  # Members of the ensemble forecast to be used

    # Drogue Parachute Details - run help(Rocket) for more information
    # Sporadic Impulse values of 08/02/22 (CdSDrogue)
    "CdSDrogue": (0.9*0.656, 0.07),                     # Drag coefficient times reference area for the drogue chute (m^2)
    "lag_rec": (1 , 0.5),                               # Time delay between parachute ejection signal is detected and parachute is inflated (s)
    
    # Main Parachute Details - run help(Rocket) for more information
    # Sporadic Impulse values of 08/02/22 (CdSDrogue)
    "CdSMain": (0.9*0.656, 0.07),                     # Drag coefficient times reference area for the drogue chute (m^2)
    "lag_rec": (1 , 0.5),                               # Time delay between parachute ejection signal is detected and parachute is inflated (s)

    # Electronic Systems Details - run help(Rocket) for more information
    "lag_se": (0.73, 0.16)                              # Time delay between sensor signal is received and ejection signal is fired (s)
}






def flight_settings(analysis_parameters, total_number):
    i = 0
    while i < total_number:
        # Generate a flight setting
        flight_setting = {}
        for parameter_key, parameter_value in analysis_parameters.items():
            if type(parameter_value) is tuple:
                flight_setting[parameter_key] =  normal(*parameter_value)
            else:
                flight_setting[parameter_key] =  choice(parameter_value)

        # Skip if certain values are negative, which happens due to the normal curve but isnt realistic
        if flight_setting['lag_rec'] < 0 or  flight_setting['lag_se'] < 0: continue

        # Update counter
        i += 1
        # Yield a flight setting
        yield flight_setting


def export_flight_data(flight_setting, flight_data, exec_time):
    # Generate flight results
    flight_result = {"outOfRailTime": flight_data.outOfRailTime,
                 "outOfRailVelocity": flight_data.outOfRailVelocity,
                        "apogeeTime": flight_data.apogeeTime,
                    "apogeeAltitude": flight_data.apogee - Env.elevation,
                           "apogeeX": flight_data.apogeeX,
                           "apogeeY": flight_data.apogeeY,
                        "impactTime": flight_data.tFinal,
                           "impactX": flight_data.xImpact,
                           "impactY": flight_data.yImpact,
                    "impactVelocity": flight_data.impactVelocity,
               "initialStaticMargin": flight_data.rocket.staticMargin(0),
             "outOfRailStaticMargin": flight_data.rocket.staticMargin(TestFlight.outOfRailTime),
                 "finalStaticMargin": flight_data.rocket.staticMargin(TestFlight.rocket.motor.burnOutTime),
                    "numberOfEvents": len(flight_data.parachuteEvents),
                     "executionTime": exec_time}

    # Calculate maximum reached velocity
    sol = np.array(flight_data.solution)
    flight_data.vx = Function(sol[:, [0, 4]], 'Time (s)', 'Vx (m/s)', 'linear', extrapolation="natural")
    flight_data.vy = Function(sol[:, [0, 5]], 'Time (s)', 'Vy (m/s)', 'linear', extrapolation="natural")
    flight_data.vz = Function(sol[:, [0, 6]], 'Time (s)', 'Vz (m/s)', 'linear', extrapolation="natural")
    flight_data.v = (flight_data.vx**2 + flight_data.vy**2 + flight_data.vz**2)**0.5
    flight_data.maxVel = np.amax(flight_data.v.source[:, 1])
    flight_result['maxVelocity'] = flight_data.maxVel

    # Take care of parachute results
    if len(flight_data.parachuteEvents) > 0:
        flight_result['drogueTriggerTime'] = flight_data.parachuteEvents[0][0]
        flight_result['drogueInflatedTime'] = flight_data.parachuteEvents[0][0] + flight_data.parachuteEvents[0][1].lag
        flight_result['drogueInflatedVelocity'] = flight_data.v(flight_data.parachuteEvents[0][0] + flight_data.parachuteEvents[0][1].lag)
    else:
        flight_result['drogueTriggerTime'] = 0
        flight_result['drogueInflatedTime'] = 0
        flight_result['drogueInflatedVelocity'] = 0

    # Write flight setting and results to file
    dispersion_input_file.write(str(flight_setting) + '\n')
    dispersion_output_file.write(str(flight_result) + '\n')

def export_flight_error(flight_setting):
    dispersion_error_file.write(str(flight_setting) + '\n')





# Basic analysis info
filename = 'dispersion_analysis_outputs/SporadicImpulse_rocket_v0'
number_of_simulations = 5
# Create data files for inputs, outputs and error logging
dispersion_error_file = open(str(filename)+'.disp_errors.txt', 'w')
dispersion_input_file = open(str(filename)+'.disp_inputs.txt', 'w')
dispersion_output_file = open(str(filename)+'.disp_outputs.txt', 'w')

# Initialize counter and timer
i = 0

initial_wall_time = time()
initial_cpu_time = process_time()

# Define basic Environment object
Env = Environment(
    railLength=12,
    date=(2019, 8, 10, 21),
    latitude=-23.363611,
    longitude=-48.011389
)
Env.setElevation(668)
Env.maxExpectedHeight = 1500
Env.setAtmosphericModel(
    type='Ensemble',
    file='dispersion_analysis_inputs/LASC2019_reanalysis.nc',
    dictionary="ECMWF"
)

# Set up parachutes. This rocket, named Sporadic Impulse, only has a drogue chute.
def drogueTrigger(p, y):
    # Check if rocket is going down, i.e. if it has passed the apogee
    vertical_velocity = y[5]
    # Return true to activate parachute once the vertical velocity is negative
    return True if vertical_velocity < 0 else False

def mainTrigger(p, y):
    # Check if rocket is going down, i.e. if it has passed the apogee
    vertical_velocity = y[5]
    # Return true to activate parachute once the vertical velocity is negative
    return True if vertical_velocity < 0 and y[2] < 500 else False #Opening based on height

# Iterate over flight settings
out = display('Starting', display_id=True)
for setting in flight_settings(analysis_parameters, number_of_simulations):
    start_time = process_time()
    i += 1

    # Update environment object
    Env.selectEnsembleMember(setting['ensembleMember'])
    Env.railLength = setting['railLength']

    # Create motor
    Hypnos =  SolidMotor(
        thrustSource='dispersion_analysis_inputs/ThrustyBoi.csv',
        burnOut=setting['burnOut'],
        reshapeThrustCurve=(setting['burnOut'], setting['impulse']),
        nozzleRadius=setting['nozzleRadius'],
        throatRadius=setting['throatRadius'],
        grainNumber=6,
        grainSeparation=setting['grainSeparation'],
        grainDensity=setting['grainDensity'],
        grainOuterRadius=setting['grainOuterRadius'],
        grainInitialInnerRadius=setting['grainInitialInnerRadius'],
        grainInitialHeight=setting['grainInitialHeight'],
        interpolationMethod='linear'
    )

    # Create rocket
    SporadicImpulse = Rocket(
        motor=Hypnos,
        radius=setting['radius'],
        mass=setting['rocketMass'],
        inertiaI=setting['inertiaI'],
        inertiaZ=setting['inertiaZ'],
        distanceRocketNozzle=setting['distanceRocketNozzle'],
        distanceRocketPropellant=setting['distanceRocketPropellant'],
        powerOffDrag='dispersion_analysis_inputs/Cd_PowerOff.csv',
        powerOnDrag='dispersion_analysis_inputs/Cd_PowerOn.csv'
    )
    SporadicImpulse.setRailButtons([0.224, -0.93], 30)
    # Edit rocket drag
    SporadicImpulse.powerOffDrag *= setting["powerOffDrag"]
    SporadicImpulse.powerOnDrag *= setting["powerOnDrag"]
    # Add rocket nose, fins and tail
    NoseCone = SporadicImpulse.addNose(
        length=setting['noseLength'],
        kind='vonKarman',
        distanceToCM=setting['noseDistanceToCM']
    )
    FinSet = SporadicImpulse.addFins(
        n=3,
        rootChord=setting['finRootChord'],
        tipChord=setting['finTipChord'],
        span=setting['finSpan'],
        distanceToCM=setting['finDistanceToCM']
    )
    # Add parachute
    Drogue = SporadicImpulse.addParachute(
        'Drogue',
        CdS=setting['CdSDrogue'],
        trigger=drogueTrigger,
        samplingRate=105,
        lag=setting['lag_rec'] + setting['lag_se'],
        noise=(0, 8.3, 0.5)
    )
    
    # Add parachute
    Main = SporadicImpulse.addParachute(
        'Main',
       CdS=setting['CdSMain'],
       trigger=mainTrigger,
       samplingRate=105,
       lag=setting['lag_rec'] + setting['lag_se'],
       noise=(0, 8.3, 0.5)
    )

    # Run trajectory simulation
    try:
        TestFlight = Flight(
            rocket=SporadicImpulse,
            environment=Env,
            inclination=setting['inclination'],
            heading=setting['heading'],
            maxTime=600
        )
        export_flight_data(setting, TestFlight, process_time() - start_time)
    except Exception as E:
        print(E)
        export_flight_error(setting)

    # Register time
    out.update(f"Curent iteration: {i:06d} | Average Time per Iteration: {(process_time() - initial_cpu_time)/i:2.6f} s")

# Done

## Print and save total time
final_string = f"Completed {i} iterations successfully. Total CPU time: {process_time() - initial_cpu_time} s. Total wall time: {time() - initial_wall_time} s"
out.update(final_string)
dispersion_input_file.write(final_string + '\n')
dispersion_output_file.write(final_string + '\n')
dispersion_error_file.write(final_string + '\n')

## Close files
dispersion_input_file.close()
dispersion_output_file.close()
dispersion_error_file.close()




filename = 'dispersion_analysis_outputs/SporadicImpulse_rocket_v0'

# Initialize variable to store all results
dispersion_general_results = []

dispersion_results = {"outOfRailTime": [],
                  "outOfRailVelocity": [],
                         "apogeeTime": [],
                     "apogeeAltitude": [],
                            "apogeeX": [],
                            "apogeeY": [],
                         "impactTime": [],
                            "impactX": [],
                            "impactY": [],
                     "impactVelocity": [],
                "initialStaticMargin": [],
              "outOfRailStaticMargin": [],
                  "finalStaticMargin": [],
                     "numberOfEvents": [],
                        "maxVelocity": [],
                  "drogueTriggerTime": [],
                 "drogueInflatedTime": [],
             "drogueInflatedVelocity": [],
                      "executionTime": []}

# Get all dispersion results
# Get file
dispersion_output_file = open(str(filename)+'.disp_outputs.txt', 'r+')

# Read each line of the file and convert to dict
for line in dispersion_output_file:
    # Skip comments lines
    if line[0] != '{': continue
    # Eval results and store them
    flight_result = eval(line)
    dispersion_general_results.append(flight_result)
    for parameter_key, parameter_value in flight_result.items():
        dispersion_results[parameter_key].append(parameter_value)

# Close data file
dispersion_output_file.close()

# Print number of flights simulated
N = len(dispersion_general_results)
print('Number of simulations: ', N)

print(f'Out of Rail Time -         Mean Value: {np.mean(dispersion_results["outOfRailTime"]):0.3f} s')
print(f'Out of Rail Time - Standard Deviation: {np.std(dispersion_results["outOfRailTime"]):0.3f} s')

plt.figure()
plt.hist(dispersion_results["outOfRailTime"], bins=int(N**0.5))
plt.title('Out of Rail Time')
plt.xlabel('Time (s)')
plt.ylabel('Number of Occurences')
plt.show()

# You can also use Plotly instead of Matplotlib if you wish!
# import plotly.express as px
# fig1 = px.histogram(
#     x=dispersion_results["outOfRailTime"],
#     title='Out of Rail Time',
#     nbins=int(N**0.5)
# )
# fig1.update_layout(
#     xaxis_title_text='Time (s)',
#     yaxis_title_text='Number of occurences'
# )

#Out of rail velocity
print(f'Out of Rail Velocity -         Mean Value: {np.mean(dispersion_results["outOfRailVelocity"]):0.3f} m/s')
print(f'Out of Rail Velocity - Standard Deviation: {np.std(dispersion_results["outOfRailVelocity"]):0.3f} m/s')

plt.figure()
plt.hist(dispersion_results["outOfRailVelocity"], bins=int(N**0.5))
plt.title('Out of Rail Velocity')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Number of Occurences')
plt.show()

#Apogee Time
print(f'Apogee Time -         Mean Value: {np.mean(dispersion_results["apogeeTime"]):0.3f} s')
print(f'Apogee Time - Standard Deviation: {np.std(dispersion_results["apogeeTime"]):0.3f} s')

plt.figure()
plt.hist(dispersion_results["apogeeTime"], bins=int(N**0.5))
plt.title('Apogee Time')
plt.xlabel('Time (s)')
plt.ylabel('Number of Occurences')
plt.show()

#Apogee Altitude
print(f'Apogee Altitude -         Mean Value: {np.mean(dispersion_results["apogeeAltitude"]):0.3f} m')
print(f'Apogee Altitude - Standard Deviation: {np.std(dispersion_results["apogeeAltitude"]):0.3f} m')

plt.figure()
plt.hist(dispersion_results["apogeeAltitude"], bins=int(N**0.5))
plt.title('Apogee Altitude')
plt.xlabel('Altitude (m)')
plt.ylabel('Number of Occurences')
plt.show()

# Real measured apogee for SporadicImpulse = 860 m

#Apogee X Position
print(f'Apogee X Position -         Mean Value: {np.mean(dispersion_results["apogeeX"]):0.3f} m')
print(f'Apogee X Position - Standard Deviation: {np.std(dispersion_results["apogeeX"]):0.3f} m')

plt.figure()
plt.hist(dispersion_results["apogeeX"], bins=int(N**0.5))
plt.title('Apogee X Position')
plt.xlabel('Apogee X Position (m)')
plt.ylabel('Number of Occurences')
plt.show()

#Apogee Y Position
print(f'Apogee Y Position -         Mean Value: {np.mean(dispersion_results["apogeeY"]):0.3f} m')
print(f'Apogee Y Position - Standard Deviation: {np.std(dispersion_results["apogeeY"]):0.3f} m')

plt.figure()
plt.hist(dispersion_results["apogeeY"], bins=int(N**0.5))
plt.title('Apogee Y Position')
plt.xlabel('Apogee Y Position (m)')
plt.ylabel('Number of Occurences')
plt.show()

#Impact Time
print(f'Impact Time -         Mean Value: {np.mean(dispersion_results["impactTime"]):0.3f} s')
print(f'Impact Time - Standard Deviation: {np.std(dispersion_results["impactTime"]):0.3f} s')

plt.figure()
plt.hist(dispersion_results["impactTime"], bins=int(N**0.5))
plt.title('Impact Time')
plt.xlabel('Time (s)')
plt.ylabel('Number of Occurences')
plt.show()

#Impact X Position
print(f'Impact X Position -         Mean Value: {np.mean(dispersion_results["impactX"]):0.3f} m')
print(f'Impact X Position - Standard Deviation: {np.std(dispersion_results["impactX"]):0.3f} m')

plt.figure()
plt.hist(dispersion_results["impactX"], bins=int(N**0.5))
plt.title('Impact X Position')
plt.xlabel('Impact X Position (m)')
plt.ylabel('Number of Occurences')
plt.show()

#Impact Y Position
print(f'Impact Y Position -         Mean Value: {np.mean(dispersion_results["impactY"]):0.3f} m')
print(f'Impact Y Position - Standard Deviation: {np.std(dispersion_results["impactY"]):0.3f} m')

plt.figure()
plt.hist(dispersion_results["impactY"], bins=int(N**0.5))
plt.title('Impact Y Position')
plt.xlabel('Impact Y Position (m)')
plt.ylabel('Number of Occurences')
plt.show()

#Impact Velocity
print(f'Impact Velocity -         Mean Value: {np.mean(dispersion_results["impactVelocity"]):0.3f} m/s')
print(f'Impact Velocity - Standard Deviation: {np.std(dispersion_results["impactVelocity"]):0.3f} m/s')

plt.figure()
plt.hist(dispersion_results["impactVelocity"], bins=int(N**0.5))
plt.title('Impact Velocity')
# plt.grid()
plt.xlim(-35,0)
plt.xlabel('Velocity (m/s)')
plt.ylabel('Number of Occurences')
plt.show()

#Static Margin
print(f'Initial Static Margin -             Mean Value: {np.mean(dispersion_results["initialStaticMargin"]):0.3f} c')
print(f'Initial Static Margin -     Standard Deviation: {np.std(dispersion_results["initialStaticMargin"]):0.3f} c')

print(f'Out of Rail Static Margin -         Mean Value: {np.mean(dispersion_results["outOfRailStaticMargin"]):0.3f} c')
print(f'Out of Rail Static Margin - Standard Deviation: {np.std(dispersion_results["outOfRailStaticMargin"]):0.3f} c')

print(f'Final Static Margin -               Mean Value: {np.mean(dispersion_results["finalStaticMargin"]):0.3f} c')
print(f'Final Static Margin -       Standard Deviation: {np.std(dispersion_results["finalStaticMargin"]):0.3f} c')

plt.figure()
plt.hist(dispersion_results["initialStaticMargin"], label='Initial', bins=int(N**0.5))
plt.hist(dispersion_results["outOfRailStaticMargin"], label='Out of Rail', bins=int(N**0.5))
plt.hist(dispersion_results["finalStaticMargin"], label='Final', bins=int(N**0.5))
plt.legend()
plt.title('Static Margin')
plt.xlabel('Static Margin (c)')
plt.ylabel('Number of Occurences')
plt.show()

#Maximum Velocity
print(f'Maximum Velocity -         Mean Value: {np.mean(dispersion_results["maxVelocity"]):0.3f} m/s')
print(f'Maximum Velocity - Standard Deviation: {np.std(dispersion_results["maxVelocity"]):0.3f} m/s')

plt.figure()
plt.hist(dispersion_results["maxVelocity"], bins=int(N**0.5))
plt.title('Maximum Velocity')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Number of Occurences')
plt.show()

#NUmber of Parachute Events
plt.figure()
plt.hist(dispersion_results["numberOfEvents"])
plt.title('Parachute Events')
plt.xlabel('Number of Parachute Events')
plt.ylabel('Number of Occurences')
plt.show()

#Drogue Parachute Trigger TIme
print(f'Drogue Parachute Trigger Time -         Mean Value: {np.mean(dispersion_results["drogueTriggerTime"]):0.3f} s')
print(f'Drogue Parachute Trigger Time - Standard Deviation: {np.std(dispersion_results["drogueTriggerTime"]):0.3f} s')

plt.figure()
plt.hist(dispersion_results["drogueTriggerTime"], bins=int(N**0.5))
plt.title('Drogue Parachute Trigger Time')
plt.xlabel('Time (s)')
plt.ylabel('Number of Occurences')
plt.show()

#Drogue Parachute Fully Inflated Time
print(f'Drogue Parachute Fully Inflated Time -         Mean Value: {np.mean(dispersion_results["drogueInflatedTime"]):0.3f} s')
print(f'Drogue Parachute Fully Inflated Time - Standard Deviation: {np.std(dispersion_results["drogueInflatedTime"]):0.3f} s')

plt.figure()
plt.hist(dispersion_results["drogueInflatedTime"], bins=int(N**0.5))
plt.title('Drogue Parachute Fully Inflated Time')
plt.xlabel('Time (s)')
plt.ylabel('Number of Occurences')
plt.show()

#Error Ellipses
# Import libraries
from imageio import imread
from matplotlib.patches import Ellipse

# Import background map
img = imread("dispersion_analysis_inputs/SporadicImpulse_basemap_final_2.jpg")

# Retrieve dispersion data por apogee and impact XY position
apogeeX = np.array(dispersion_results['apogeeX'])
apogeeY = np.array(dispersion_results['apogeeY'])
impactX = np.array(dispersion_results['impactX'])
impactY = np.array(dispersion_results['impactY'])

# Define function to calculate eigen values
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

# Create plot figure
plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
ax = plt.subplot(111)

# Calculate error ellipses for impact
impactCov = np.cov(impactX, impactY)
impactVals, impactVecs = eigsorted(impactCov)
impactTheta = np.degrees(np.arctan2(*impactVecs[:,0][::-1]))
impactW, impactH = 2 * np.sqrt(impactVals)

# Draw error ellipses for impact
impact_ellipses = []
for j in [1, 2, 3]:
    impactEll = Ellipse(xy=(np.mean(impactX), np.mean(impactY)),
                  width=impactW*j, height=impactH*j,
                  angle=impactTheta, color='black')
    impactEll.set_facecolor((0, 0, 1, 0.2))
    impact_ellipses.append(impactEll)
    ax.add_artist(impactEll)

# Calculate error ellipses for apogee
apogeeCov = np.cov(apogeeX, apogeeY)
apogeeVals, apogeeVecs = eigsorted(apogeeCov)
apogeeTheta = np.degrees(np.arctan2(*apogeeVecs[:,0][::-1]))
apogeeW, apogeeH = 2 * np.sqrt(apogeeVals)

# Draw error ellipses for apogee
for j in [1, 2, 3]:
    apogeeEll = Ellipse(xy=(np.mean(apogeeX), np.mean(apogeeY)),
                  width=apogeeW*j, height=apogeeH*j,
                  angle=apogeeTheta, color='black')
    apogeeEll.set_facecolor((0, 1, 0, 0.2))
    ax.add_artist(apogeeEll)

# Draw launch point
plt.scatter(0, 0, s=30, marker='*', color='black', label='Launch Point')
# Draw apogee points
plt.scatter(apogeeX, apogeeY, s=5, marker='^', color='green', label='Simulated Apogee')
# Draw impact points
plt.scatter(impactX, impactY, s=5, marker='v', color='blue', label='Simulated Landing Point')
# Draw real landing point
plt.scatter(411.89, -61.07, s=20, marker='X', color='red', label='Measured Landing Point')

plt.legend()

# Add title and labels to plot
ax.set_title('1$\sigma$, 2$\sigma$ and 3$\sigma$ Dispersion Ellipses: Apogee and Lading Points')
ax.set_ylabel('North (m)')
ax.set_xlabel('East (m)')

# Add background image to plot
# You can translate the basemap by changing dx and dy (in meters)
dx = 0
dy = 0
plt.imshow(img,zorder=0, extent=[-1000-dx, 1000-dx, -1000-dy, 1000-dy])
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-1000, 1000)
plt.ylim(-1000, 1000)

# Save plot and show result
plt.savefig(str(filename)+ '.pdf', bbox_inches='tight', pad_inches=0)
plt.savefig(str(filename)+ '.svg', bbox_inches='tight', pad_inches=0)
plt.show()
