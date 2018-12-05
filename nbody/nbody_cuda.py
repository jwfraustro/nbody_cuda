import math
from random import uniform

import numpy
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from numba import *
from numba import cuda

lastPosX = 0
lastPosY = 0
zoomScale = 1.0
dataL = 0
xRot = 0
yRot = 0
zRot = 0

# softening distance
MIN_DIST = 0.1

# gravity constant
GRAVITY = 0.1

# timesteps
DELTA_T = 1

# particle count
PARTICLES = 50000
# maximum radius
RADIUS = 1000

PAUSE = False


# Freeman surface brightness distribution for galactic disk
def FreemanDistribution(r, r_d=0.35):
	# r_d: characteristic scale length (% of radius where brightness = 50%
	p = math.exp(-r / r_d)

	return p


@cuda.jit
# cuda implementation of euler integration
def velocityCalculation(A, V, B):
	row, col = cuda.grid(2)
	# make sure we don't go out of cuda memory
	if row < A.shape[0]:
		if col < A.shape[0]:
			# make sure we're not on the same particle
			if row == col:
				return
			# calculate distance between particles
			r_x = (A[col, 0] - A[row, 0])
			r_y = (A[col, 1] - A[row, 1])
			r_z = (A[col, 2] - A[row, 2])

			r = (r_x ** 2 + r_y ** 2 + r_z ** 2) ** 0.5

			# a softening factor and making sure we don't divide by zero
			r = max(r, MIN_DIST)

			# updating velocities
			V[row, 0] += ((GRAVITY * .01 * DELTA_T) / r) * r_x / r
			V[row, 1] += ((GRAVITY * .01 * DELTA_T) / r) * r_y / r
			V[row, 2] += ((GRAVITY * .01 * DELTA_T) / r) * r_z / r

		# distance to gravity well / force
		r_x = (B[0] - A[row, 0])
		r_y = (B[1] - A[row, 1])
		r_z = (B[2] - A[row, 2])

		r = (r_x ** 2 + r_y ** 2 + r_z ** 2) ** 0.5

		r = max(r, MIN_DIST)

		# multiply by 3 -- arbitrary scaling factor
		V[row, 0] += ((GRAVITY * 3 * DELTA_T) / r) * r_x / r
		V[row, 1] += ((GRAVITY * 3 * DELTA_T) / r) * r_y / r
		V[row, 2] += ((GRAVITY * 3 * DELTA_T) / r) * r_z / r

		# update positions in parallel
		A[row, 0] += V[row, 0]
		A[row, 1] += V[row, 1]
		A[row, 2] += V[row, 2]


def initializeParticles(number, distance):
	# initialize arrays
	vertices = numpy.zeros((number, 3), numpy.float)
	velocities = numpy.zeros((number, 3), numpy.float)

	for i in range(number):
		# generate random position, then sample freeman distribution to see if it is valid
		while True:
			rand = uniform(0, 1)
			r = uniform(0, 1)
			if rand <= FreemanDistribution(r):
				r = r * distance
				break
			else:
				continue

		# random angle in the circle
		theta = uniform(0, 2 * math.pi)

		x = r * math.cos(theta)
		y = r * math.sin(theta)
		# TODO -- brightness distribution for disk galaxies
		z = uniform(-5, 5)

		vertices[i] = [x, y, z]

		# initial radial velocity based on Keplerian model
		# TODO -- use Kuzmin-style model
		rad_vel = (GRAVITY / r ** 2) ** 0.5

		vx = -r * rad_vel * math.sin(theta)
		vy = r * rad_vel * math.cos(theta)
		# random z wiggle
		vz = uniform(-0.0005, 0.0005)

		velocities[i] = [vx, vy, vz]

	return vertices, velocities, numpy.array([0, 0, 0], numpy.float)


def draw(vertices):
	glVertexPointerd(vertices)
	glEnable(GL_VERTEX_ARRAY)
	glDrawArrays(GL_POINTS, 0, len(vertices))


def main():
	global lastPosX, lastPosY, zoomScale, xRot, yRot, zRot, PAUSE, GRAVITY

	# initialize particles, velocities and gravity well
	A, V, B = initializeParticles(PARTICLES, RADIUS)

	# CUDA memory initialization
	threadsperblock = (16, 16)
	blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
	blockspergrid_y = int(math.ceil(A.shape[1] / threadsperblock[1]))
	blockspergrid = (blockspergrid_x, blockspergrid_y)

	# display setup
	display = (600, 600)
	pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
	gluPerspective(45, (display[0] / display[1]), 0.1, RADIUS * 2)
	glTranslatef(0.0, 0.0, -RADIUS)
	glRotatef(0, 0, 20, 0.5)

	clock = pygame.time.Clock()

	while True:
		# maximum refresh rate
		clock.tick(90)
		pygame.display.set_caption("%s  FPS: %.2f" % ('N-Body Particles - Single Gravity Well', clock.get_fps()))

		# event handling
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit(0)
			if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
				glScaled(1.05, 1.05, 1.05)
			elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
				glScaled(0.95, 0.95, 0.95)
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_KP6:
					B[0] += 5
				elif event.key == pygame.K_KP4:
					B[0] -= 5
				elif event.key == pygame.K_KP8:
					B[1] += 5
				elif event.key == pygame.K_KP2:
					B[1] -= 5
				elif event.key == pygame.K_KP9:
					B[2] += 5
				elif event.key == pygame.K_KP3:
					B[2] -= 5
				elif event.key == pygame.K_SPACE:
					PAUSE = not PAUSE
				elif event.key == pygame.K_r:
					A, V, B = initializeParticles(PARTICLES, RADIUS)

				elif event.key == pygame.K_s:
					pygame.image.save()

			if event.type == pygame.MOUSEMOTION:
				x, y = event.pos
				dx = x - lastPosX
				dy = y - lastPosY

				mouseState = pygame.mouse.get_pressed()
				if mouseState[0]:
					modelView = (GLfloat * 16)()
					mvm = glGetFloatv(GL_MODELVIEW_MATRIX, modelView)

					# To combine x-axis and y-axis rotation
					temp = (GLfloat * 3)()
					temp[0] = modelView[0] * dy + modelView[1] * dx
					temp[1] = modelView[4] * dy + modelView[5] * dx
					temp[2] = modelView[8] * dy + modelView[9] * dx
					norm_xy = math.sqrt(temp[0] * temp[0] + temp[1] * temp[1] + temp[2] * temp[2])
					glRotatef(math.sqrt(dx * dx + dy * dy), temp[0] / norm_xy, temp[1] / norm_xy, temp[2] / norm_xy)

				lastPosX = x
				lastPosY = y

		# calculate and redraw
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		if not PAUSE:
			velocityCalculation[blockspergrid, threadsperblock](A, V, B)
		draw(A)
		pygame.display.flip()


if __name__ == '__main__':
	main()
