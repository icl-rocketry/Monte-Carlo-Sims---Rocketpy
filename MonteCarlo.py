import os
os.chdir('dispersion_analysis')

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
    "rocketMass": (8.257, 0.001), # Rocket's dry mass (kg) and its uncertainty (standard deviation)

    # Propulsion Details - run help(SolidMotor) for more information
    "impulse": (1415.15, 35.3),                         # Motor total impulse (N*s)
    "burnOut": (5.274, 1),                              # Motor burn out time (s)
    "nozzleRadius": (21.642/1000, 0.5/1000),            # Motor's nozzle radius (m)
    "throatRadius": (8/1000, 0.5/1000) ,                # Motor's nozzle throat radius (m)
    "grainSeparation": (6/1000, 1/1000),                # Motor's grain separation (axial distance between two grains) (m)
    "grainDensity": (1707, 50),                         # Motor's grain density (kg/m^3)
    "grainOuterRadius": (21.4/1000, 0.375/1000),        # Motor's grain outer radius (m)
    "grainInitialInnerRadius": (9.65/1000, 0.375/1000), # Motor's grain inner radius (m)
    "grainInitialHeight": (120/1000, 1/1000),           # Motor's grain height (m)

    # Aerodynamic Details - run help(Rocket) for more information
    "inertiaI": (3.675, 0.03675),                       # Rocket's inertia moment perpendicular to its axis (kg*m^2)
    "inertiaZ": (0.007, 0.00007),                       # Rocket's inertia moment relative to its axis (kg*m^2)
    "radius": (40.45/1000, 0.001),                      # Rocket's radius (kg*m^2)
    "distanceRocketNozzle": (-1.024,0.001),             # Distance between rocket's center of dry mass and nozzle exit plane (m) (negative)
    "distanceRocketPropellant": (-0.571,0.001),         # Distance between rocket's center of dry mass and and center of propellant mass (m) (negative)
    "powerOffDrag": (0.9081/1.05, 0.033),               # Multiplier for rocket's drag curve. Usually has a mean value of 1 and a uncertainty of 5% to 10%
    "powerOnDrag": (0.9081/1.05, 0.033),                # Multiplier for rocket's drag curve. Usually has a mean value of 1 and a uncertainty of 5% to 10%
    "noseLength": (0.274, 0.001),                       # Rocket's nose cone length (m)
    "noseDistanceToCM": (1.134, 0.001),                 # Axial distance between rocket's center of dry mass and nearest point in its nose cone (m)
    "finSpan": (0.077, 0.0005),                         # Fin span (m)
    "finRootChord": (0.058, 0.0005),                    # Fin root chord (m)
    "finTipChord": (0.018, 0.0005),                     # Fin tip chord (m)
    "finDistanceToCM": (-0.906, 0.001),                 # Axial distance between rocket's center of dry mass and nearest point in its fin (m)

    # Launch and Environment Details - run help(Environment) and help(Flight) for more information
    "inclination": (84.7, 1),                           # Launch rail inclination angle relative to the horizontal plane (degrees)
    "heading": (53, 2),                                 # Launch rail heading relative to north (degrees)
    "railLength": (5.7 , 0.0005),                       # Launch rail length (m)
    "ensembleMember": list(range(10)),                  # Members of the ensemble forecast to be used

    # Parachute Details - run help(Rocket) for more information
    "CdSDrogue": (0.349*1.3, 0.07),                     # Drag coefficient times reference area for the drogue chute (m^2)
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
filename = 'dispersion_analysis_outputs/valetudo_rocket_v0'
number_of_simulations = 20000

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
    railLength=5.7,
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

# Set up parachutes. This rocket, named Valetudo, only has a drogue chute.
def drogueTrigger(p, y):
    # Check if rocket is going down, i.e. if it has passed the apogee
    vertical_velocity = y[5]
    # Return true to activate parachute once the vertical velocity is negative
    return True if vertical_velocity < 0 else False

# Iterate over flight settings
out = display('Starting', display_id=True)
for setting in flight_settings(analysis_parameters, number_of_simulations):
    start_time = process_time()
    i += 1

    # Update environment object
    Env.selectEnsembleMember(setting['ensembleMember'])
    Env.railLength = setting['railLength']

    # Create motor
    Keron =  SolidMotor(
        thrustSource='dispersion_analysis_inputs/thrustCurve.csv',
        burnOut=5.274,
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
    Valetudo = Rocket(
        motor=Keron,
        radius=setting['radius'],
        mass=setting['rocketMass'],
        inertiaI=setting['inertiaI'],
        inertiaZ=setting['inertiaZ'],
        distanceRocketNozzle=setting['distanceRocketNozzle'],
        distanceRocketPropellant=setting['distanceRocketPropellant'],
        powerOffDrag='dispersion_analysis_inputs/Cd_PowerOff.csv',
        powerOnDrag='dispersion_analysis_inputs/Cd_PowerOn.csv'
    )
    Valetudo.setRailButtons([0.224, -0.93], 30)
    # Edit rocket drag
    Valetudo.powerOffDrag *= setting["powerOffDrag"]
    Valetudo.powerOnDrag *= setting["powerOnDrag"]
    # Add rocket nose, fins and tail
    NoseCone = Valetudo.addNose(
        length=setting['noseLength'],
        kind='vonKarman',
        distanceToCM=setting['noseDistanceToCM']
    )
    FinSet = Valetudo.addFins(
        n=3,
        rootChord=setting['finRootChord'],
        tipChord=setting['finTipChord'],
        span=setting['finSpan'],
        distanceToCM=setting['finDistanceToCM']
    )
    # Add parachute
    Drogue = Valetudo.addParachute(
        'Drogue',
        CdS=setting['CdSDrogue'],
        trigger=drogueTrigger,
        samplingRate=105,
        lag=setting['lag_rec'] + setting['lag_se'],
        noise=(0, 8.3, 0.5)
    )

    # Run trajectory simulation
    try:
        TestFlight = Flight(
            rocket=Valetudo,
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




filename = 'dispersion_analysis_outputs/valetudo_rocket_v0'

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