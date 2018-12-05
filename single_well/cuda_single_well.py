from numba import cuda, njit
import numpy, math, sys
from random import uniform, random
from timeit import default_timer as timer

import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from numba import *
from pygame.locals import *

lastPosX = 0
lastPosY = 0
zoomScale = 1.0
dataL = 0
xRot = 0
yRot = 0
zRot = 0
FRICTION = 0.01
INIT_VEL = 0.6
MIN_DIST = 0.01
GRAVITY = 0.3
DELTA_T = 1
PARTICLES = 250000
RADIUS = 500

def initializeArrays(number, distance):

	vertices = numpy.zeros((number, 3), numpy.float)
	velocities = numpy.zeros((number,3), numpy.float)

	for i in range(number):

		x = uniform(-distance, distance)
		y = uniform(-distance, distance)
		z = uniform(-20, 20)

		vertices[i] = [x,y,z]

		r = math.sqrt(x ** 2 + y ** 2)

		rad_vel = (GRAVITY / r ** 2) ** 0.5

		theta = math.atan2(y, x)

		vx = -r * rad_vel * math.sin(theta)
		vy = r * rad_vel * math.cos(theta)
		vz = uniform(-0.02, 0.02)

		velocities[i] = [vx,vy,vz]

	return vertices, velocities, numpy.array([0,0,0], numpy.float)


@cuda.jit
def velocityCalculation(A, V, B):

	row = cuda.grid(1)
	if row < A.shape[0]:
		r_x = (B[0] - A[row, 0])
		r_y = (B[1] - A[row, 1])
		r_z = (B[2] - A[row, 2])

		r = (r_x ** 2 + r_y ** 2 + r_z ** 2) ** 0.5

		r = max(r, MIN_DIST)

		V[row, 0] += ((GRAVITY * DELTA_T) / r) * r_x / r
		V[row, 1] += ((GRAVITY * DELTA_T) / r) * r_y / r
		V[row, 2] += ((GRAVITY * DELTA_T) / r) * r_z / r

		A[row, 0] += V[row, 0]
		A[row, 1] += V[row, 1]
		A[row, 2] += V[row, 2]

def draw2(vertices):
	glVertexPointerd(vertices)
	glEnable(GL_VERTEX_ARRAY)
	glDrawArrays(GL_POINTS, 0, len(vertices))

def main():
	global lastPosX, lastPosY, zoomScale, xRot, yRot, zRot

	A, V, B = initializeArrays(PARTICLES,RADIUS)

	threadsperblock = 128
	blockspergrid = int(math.ceil(A.shape[0] / threadsperblock))

	display = (1280, 720)
	pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
	gluPerspective(45, (display[0] / display[1]), 0.1, 500.0)
	glTranslatef(0.0, 0.0, -350)
	glRotatef(75, -1, 0.25, 0.5)

	clock = pygame.time.Clock()

	while True:
		clock.tick(90)
		pygame.display.set_caption("%s  FPS: %.2f" % ('Particles - Single Gravity Well', clock.get_fps()))

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit(0)
			if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
				glScaled(1.05, 1.05, 1.05)
			elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
				glScaled(0.95, 0.95, 0.95)
			# if event.type == pygame.KEYDOWN:
			# 	if event.key == pygame.K_KP6:
			# 		gwell_1[0] += 5
			# 	elif event.key == pygame.K_KP4:
			# 		gwell_1[0] -= 5
			# 	elif event.key == pygame.K_KP8:
			# 		gwell_1[1] += 5
			# 	elif event.key == pygame.K_KP2:
			# 		gwell_1[1] -= 5
			# 	elif event.key == pygame.K_KP9:
			# 		gwell_1[2] += 5
			# 	elif event.key == pygame.K_KP3:
			# 		gwell_1[2] -= 5
			# 	print(gwell_1)
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

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		velocityCalculation[blockspergrid, threadsperblock](A,V,B)

		draw2(A)

		pygame.display.flip()

if __name__ == '__main__':
    main()


