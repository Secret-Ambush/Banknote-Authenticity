
import random, sys

################################################################################

# the x and y in our function (x - y + 7) (aka. dimensions)
number_of_variables = 3
# the minimum possible value a pose difference can have
min_values = [-10, -10, -3.14/2]
# the maximum possible value a pose difference can have
max_values = [10, 10, 3.14/2]
# the number of particles in the swarm
number_of_particles = 100
# number of times the algorithm moves each particle in the problem space
number_of_iterations = 2000

w = 0.729    # inertia
c1 = 1.49 # cognitive (particle)
c2 = 1.49 # social (swarm)

################################################################################

class Particle:

    # this is only done one time when we create each particle
    # value of pose, velocities, fitness, best fitness, and best pose
    # is going to be updated later in the code during each iteration
    def __init__(self, number_of_variables, min_values, max_values):

        # init pose difference values
        self.pose = [0.0 for v in range(number_of_variables)]
        # init velocities of x and y
        self.velocities = [0.0 for v in range(number_of_variables)]

        for v in range(number_of_variables):
            # Initialize pose particle params uniformly
            self.pose[v] = ((max_values[v] - min_values[v]) * random.random()
                                + min_values[v])
            # Initialize pose particle velocities uniformly
            self.velocities[v] = ((max_values[v] - min_values[v]) * random.random()
                                + min_values[v])

        # current fitness after updating the x and y values
        self.fitness = Fitness(self.pose)
        # the current particle pose as the best fitness found yet
        self.best_particle_pose = list(self.pose)
        # the current particle fitness as the best fitness found yet
        self.best_particle_fitness = self.fitness

################################################################################

# This is the objective function by which we examine the quality of the solution
# remember that we want to find the values of the pose difference such that we minimize the
# value of the whole function. The best solution would be f(10,10, pi/6)=7, the lowest possible value of the function.
def Fitness(pose):
    import numpy as np
    # Dummy optimization function
    return ((pose[0]-10)**2) + ((pose[1]-10)**2) + np.cos(pose[2]-3.14/6)**2 * (pose[2]-3.14/6)**2 + 7

def Map_matching_score(observations, pose, map):
    value = [] # TODO: Rasmus calculation routine
    return value

################################################################################

# calculate a new velocity for one variable
def calculate_new_velocity_value(particle, v):

    # generate random numbers
    r1 = random.random()
    r2 = random.random()

    # the learning rate part
    part_1 = (w * particle.velocities[v])
    # the cognitive part - learning from itself
    part_2 = (c1 * r1 * (particle.best_particle_pose[v] - particle.pose[v]))
    # the social part - learning from others
    part_3 = (c2 * r2 * (best_swarm_pose[v] - particle.pose[v]))

    new_velocity = part_1 + part_2 + part_3

    return new_velocity

################################################################################

# create the swarm
swarm = [Particle(number_of_variables, min_values, max_values)
                    for __x in range(number_of_particles)]

######################################### best particle error and pose ####

best_swarm_pose = [0.0 for v in range(number_of_variables)]
best_swarm_fitness = sys.float_info.max

for particle in swarm: # check each particle
    if particle.fitness < best_swarm_fitness:
        best_swarm_fitness = particle.fitness
        best_swarm_pose = list(particle.pose)

################################################################################

for __x in range(number_of_iterations):
    for particle in swarm:
        # start moving/updating particles to calculate new fitness

        # compute new velocities for each particle
        for v in range(number_of_variables):

            particle.velocities[v] = calculate_new_velocity_value(particle, v)

            if particle.velocities[v] < min_values[v]:
                particle.velocities[v] = min_values[v]
            elif particle.velocities[v] > max_values[v]:
                particle.velocities[v] = max_values[v]

        # compute new pose using the new velocities
        for v in range(number_of_variables):
            particle.pose[v] += particle.velocities[v]

            if particle.pose[v] < min_values[v]:
                particle.pose[v] = min_values[v]
            elif particle.pose[v] > max_values[v]:
                particle.pose[v] = max_values[v]

        # compute the fitness of the new pose
        particle.fitness = Fitness(particle.pose)

        # are the new pose a new best for the particle?
        if particle.fitness < particle.best_particle_fitness:
            particle.best_particle_fitness = particle.fitness
            particle.best_particle_pose = list(particle.pose)

        # are the new pose a new best overall?
        if particle.fitness < best_swarm_fitness:
            best_swarm_fitness = particle.fitness
            best_swarm_pose = list(particle.pose)


################################################################################

print(best_swarm_pose)