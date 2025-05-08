# https://scipython.com/blog/the-maxwellboltzmann-distribution-in-two-dimensions/
# https://stackoverflow.com/questions/35211114/2d-elastic-ball-collision-physics

from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from matplotlib.animation import PillowWriter
import scienceplots
plt.style.use(['science', 'notebook'])
import numpy as np
import random
from decimal import Decimal





def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur"""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]



def init_velocity_random():
    """"returns a list of random velocities (sum of all is specified) in random basis directions"""

    velocities = np.zeros((2, number_particles))

    for i in range(0, number_particles):
        if(i%2 == 0):
            velocities[np.random.choice([0,1])][i] = -constrained_sum_sample_pos(number_particles, number_particles*500)[i]
        else:
            velocities[np.random.choice([0,1])][i] = constrained_sum_sample_pos(number_particles, number_particles*500)[i]
    
    return velocities



def init_velocity():
    """"returns initializes velocities for particles"""

    velocities = np.zeros((2, number_particles))

    for i in range(0,number_particles):
        if(np.random.choice([0,1])):
            velocities[0][i] = 500
        else:
            velocities[0][i] = -500

    return velocities



def get_distance_pairs(ids_pairs, positions):
    dx = np.array(positions[0][ids_pairs[:,0]] - positions[0][ids_pairs[:,1]]).T
    dy = np.array(positions[1][ids_pairs[:,0]] - positions[1][ids_pairs[:,1]]).T

    distance_pairs = np.sqrt( dx**2 + dy**2 )
    return distance_pairs



def new_velocities(v1, v2, r1,r2):
    
    global count

    #different trys for solving the problem+++++++++++++++++++++++++++++++++++++++++++++++
    #-------------------------------------------------------------------------------------

    # Normal vector and tangent
    #n = r1 - r2
    #n_norm = np.linalg.norm(n, axis=0)
    #un = n / (n_norm + 1e-10)  # Avoid division by zero
    
    # Relative velocity
    #v_rel = v1 - v2
    
    # Velocity exchange in normal direction (for equal mass)
    #impulse = np.sum(v_rel * un, axis=0)
    #v1new = v1 - impulse * un
    #v2new = v2 + impulse * un

    #-------------------------------------------------------------------------------------

    # delta_r = r1 - r2
    # distance_sq = np.sum(delta_r**2, axis=0)
    
    # Avoid division by near-zero (use threshold)
    # mask = distance_sq > 1e-14  # Adjust threshold based on your scale
    # term = np.zeros_like(distance_sq)
    # term[mask] = np.sum((v1 - v2) * delta_r, axis=0)[mask] / distance_sq[mask]
    
    # v1new = v1 - term * delta_r
    # v2new = v2 + term * delta_r  # Symmetric due to Newton's 3rd law

    #-------------------------------------------------------------------------------------

    #v1new = v1 - (((v1 -v2) * (r1 - r2))/(abs(r1-r2)**2))*(r1-r2)
    #v2new = v2 - (((v2 -v1) * (r2 - r1))/(abs(r1-r2)**2))*(r2-r1)

    #-------------------------------------------------------------------------------------
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    #BUG
    """
    Der Fehler tritt wahrscheinlich hier auf
    In diesem Teil werden die Geschwindigkeiten der zwei kollidierten Teilchen angepasst (bei identischen Massen)
    Theoretisch müsste es zwischen dem arithmetischen Mittel der Geschwindkeiten keine Veränderung geben,
    allerdings ist dies nicht der Fall und dieses verändert sich um einen verhältnismäßig kleinen Betrag,
    daraus kann man schließen, dass Energie nicht erhalten wird
    -> zu dem kann bei vielen Iteration der Fehler akkumulieren

    Eine berechtigte Anmerkung wäre, dass mir ein Fehler bei der restlichen Implementation unterlaufen ist
    Aber da, selbst wenn beispielsweise Kollisionen falsch detektiert oder Geschwindigkeiten zu Zeitpunkten geändert werden, wo dies nicht passieren sollte,
    die Anpassung der Geschwindigkeit auf diese Gleichungen zurückzuführen ist, ändert dies nichts an der Tatsache, dass Energie erhalten bleiben sollte,
    auch wenn Teilchen sich unnatürlich bewegen

    Ich bin deshalb zum Schluss gekommen, dass Floating Point Arithmetic Fehler der Grund sind,
    zum Beispiel bei der Division oder der Berechnung der Skalarprodukte 
    """
    v1new = v1 - (np.sum(((v1-v2)*(r1-r2)), axis=0))/(np.sum((r1-r2)**2, axis=0)) * (r1-r2)
    v2new = v2 - (np.sum(((v1-v2)*(r1-r2)), axis=0))/(np.sum((r2-r1)**2, axis=0)) * (r2-r1)



    
    #for Debugging -------------------------------------------------------------------------------------------------------------------------------------------
    #shows, that there is some kind of problem with the equation (probably floating point arithmetic) as energy changes and therefore is not preserved
    temp = count
    
    if(np.linalg.norm(v1-v2) > np.linalg.norm(v1new-v2new)):
        count -= 1
    elif(np.linalg.norm(v1-v2) < np.linalg.norm(v1new-v2new)):
        count += 1

    #under the assumption that masses are equal and energy is preserved
    #the absolute value ofthe velocities summed before and after the collision should be equal
    #which they arent
    # if(temp!=count): 
    #     summevalt = (np.sum(abs(v1new))+np.sum(abs(v2new)))
    #     summevneu = (np.sum(abs(v1))+np.sum(abs(v2)))
    #     # print(str(summevalt) + "\n" + str(summevneu) + "\n")

    #     ratiovnewvold.append(summevalt/summevneu)

    #for plotting the ratio between the total sum of the new and ol velocities 
    sumvnew = (np.sum(abs(v1new))+np.sum(abs(v2new)))
    sumvold = (np.sum(abs(v1))+np.sum(abs(v2)))
    ratiovnewvold.append(sumvold/sumvnew)
    #---------------------------------------------------------------------------------------------------------------------------------------------------------

    return v1new, v2new



def motion(positions, velocities, ids_pairs, timesteps, distancescalar):
    boundary_particle = radius*2
    positions_inTime = np.zeros((timesteps, positions.shape[0], positions.shape[1]))
    velocities_inTime = np.zeros((timesteps, velocities.shape[0], velocities.shape[1]))

    #initialize (before time starts)
    positions_inTime[0] = positions.copy()
    velocities_inTime[0] = velocities.copy()

    for i in range(1,timesteps):
        distancepairs = get_distance_pairs(ids_pairs, positions) < boundary_particle
        index_collision = ids_pairs[distancepairs]
        #velocity and direction change if particles collide
        velocities[:,index_collision[:,0]], velocities[:,index_collision[:,1]] = new_velocities(velocities[:,index_collision[:,0]], velocities[:,index_collision[:,1]], positions[:,index_collision[:,0]], positions[:,index_collision[:,1]])

        #collision with box-borders
        velocities[0, positions[0] > 1] *= -1 #right border
        velocities[0, positions[0] < 0] *= -1 #left border
        velocities[1, positions[1] > 1] *= -1 #upper border
        velocities[1, positions[1] < 0] *= -1 #lower border
        
        # if i % 100 == 0:
        #     current_KE = 0.5 * np.sum(velocities**2)
        #     correction_factor = np.sqrt(initial_KE / current_KE)
        #     print(correction_factor)
        #     velocities *= correction_factor

        #TODO: to review
        #resolve overlaps----------------------------------------------------------------------------------
        r1 = positions[:, index_collision[:,0]]  # Position of first particle
        r2 = positions[:, index_collision[:,1]]  # Position of second particle
        distances = np.linalg.norm(r1-r2,axis=0)
        n = (r1 - r2) / (distances + 1e-10)  # Add epsilon to avoid division by zero
        positions[:, index_collision[:,0]] += 0.5 * (boundary_particle - distances) * n
        positions[:, index_collision[:,1]] -= 0.5 * (boundary_particle - distances) * n
        #--------------------------------------------------------------------------------------------------

        positions = positions + velocities * distancescalar
        positions_inTime[i] = positions.copy()
        velocities_inTime[i] = velocities.copy()
    
    return positions_inTime, velocities_inTime



def animate(i):
    [ax.clear() for ax in axes]
    ax = axes[0]
    x, y = positions_inTime[i][0], positions_inTime[i][1]
    circles = [plt.Circle((xi, yi), radius=radius, linewidth=0) for xi,yi in zip(x,y)]
    c = matplotlib.collections.PatchCollection(circles, facecolors='red')
    ax.add_collection(c)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax = axes[1]
    ax.hist(np.sqrt(np.sum(abs(velocities_inTime[i])**2, axis=0)), bins=bins, density=True)
    ax.plot(v,fv)
    ax.set_xlabel('Velocity [m/s]')
    ax.set_ylabel('# Particles')
    ax.set_xlim(0,1500)
    ax.set_ylim(0,0.006)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    fig.tight_layout()



def rendering():
    ani = animation.FuncAnimation(fig, animate, 500, interval=50)
    ani.save('boltzmann.gif',writer='pillow',fps=30,dpi=100)




count = 0
number_particles = 800
radius = 0.01
timesteps = 500

positions = np.random.random((2, number_particles)).astype(np.longdouble)
ids = np.arange(number_particles)

velocities = init_velocity().astype(np.longdouble)
ids_pairs = np.asarray(list(combinations(ids, 2)))



initial_KE = 0.5 * np.sum(velocities**2)

norm_velo = np.linalg.norm(velocities[0]-velocities[1])

startingmean = np.sum(abs(velocities))/number_particles

ratiovnewvold = []


positions_inTime, velocities_inTime = motion(positions, velocities, ids_pairs, timesteps, distancescalar=0.00008)





#for Debugging-----------------------------------------------------------------------------------------------------
print("mean velocity[beginning]: " + str(startingmean) + "\n")

#when is energy first no more preserved
for i in range(0, timesteps):
    if(np.sum(abs(velocities_inTime[i]))/number_particles != startingmean):
        print(f"the energy first changes at timestep {i} \n")
        break

print("mean velocity[middle]: " + str(np.sum(abs(velocities_inTime[int(timesteps/2)]))/number_particles)+ "\n")
print("mean velocity[end]: " + str(np.sum(abs(velocities_inTime[int(timesteps-1)]))/number_particles)+ "\n")


#displays ratio of the velocities
xaxis = np.array(range(0,len(ratiovnewvold)))
plt.title("ratio of velocity")
plt.xlabel("number of collisions")
plt.ylabel("ratio")
plt.plot(xaxis, ratiovnewvold, color="red", marker="o")
plt.legend()
plt.show()
#-> fluctuates around 1 but should be 1


totalEnergy = []
totalEnergy.append(startingmean)
for i in range(0, timesteps):
    totalEnergy.append(np.sum(abs(velocities_inTime[i]))/number_particles)

#shows the total energy change over time
xaxis = np.array(range(0,timesteps+1))
plt.title("total energy change")
plt.xlabel("timesteps")
plt.ylabel("energy")
plt.plot(xaxis, totalEnergy, color="red", marker="o")
plt.legend()
plt.show()
#------------------------------------------------------------------------------------------------------------------



#maxwell-boltzmann two-dimensional
v = np.linspace(0, 2000, 1000)
a = 2/startingmean**2
fv = a*v*np.exp(-a*v**2 / 2)

fig, axes = plt.subplots(1, 2, figsize=(20,10))
bins = np.linspace(0,1500,50)


#uncomment for animated gif of moving particles and dynamic Maxwell-Boltzmann Distribution
#rendering()
