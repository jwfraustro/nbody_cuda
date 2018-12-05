import math
from random import uniform, random

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
INIT_VEL = 0.3
MIN_DIST = 0.001
GRAVITY = 0.3
DELTA_T = 5
PARTICLES = 5000
RADIUS = 400


def bright(r):
	p = math.exp(-r / 0.35)

	return p

class Body:
	def __init__(self):
		self.x = 0
		self.y = 0
		self.z = 0

		self.vx = 0
		self.vy = 0
		self.vz = 0

		self.mass = 0

		self.color = (0, 0, 0)


def initializeVertices(number, distance):
	vertices = []

	for i in range(number):
		particle = Body()

		while True:
			rand = uniform(0, 1)
			r = uniform(0, 1)
			if rand <= bright(r):
				r = r * distance
				break
			else:
				continue

		particle.z = uniform(-10, 10)
		theta = uniform(0, 2*math.pi)

		particle.x = r * math.cos(theta)
		particle.y =  r * math.sin(theta)


		rad_vel = (GRAVITY / r**2)**0.5

		theta = math.atan2(particle.y, particle.x)

		particle.vx = -r * rad_vel * math.sin(theta)
		particle.vy = r * rad_vel * math.cos(theta)
		particle.vz = uniform(-0.02, 0.02)

		vertices.append(particle)

	return vertices

def drawVertices(vertices):
	glBegin(GL_POINTS)
	for v in vertices:
		glVertex3f(v.x, v.y, v.z)
	glEnd()


def drawCenter(x, y, z):
	loop1 = [
		[x+1, y,z],
		[x,y+1,z],
		[x-1,y,z],
		[x,y-1,z],

	]

	loop2 = [
		[x + 1, y, z],
		[x, y, z - 1],
		[x - 1, y, z],
		[x, y, z + 1]
	]

	loop3 = [
		[x,y+1,z],
		[x,y,z+1],
		[x,y-1,z],
		[x,y,z-1]
	]

	glBegin(GL_LINE_LOOP)
	for v in loop1:
		glVertex3f(v[0], v[1], v[2])
	glEnd()
	glBegin(GL_LINE_LOOP)
	for v in loop2:
		glVertex3f(v[0], v[1], v[2])
	glEnd()
	glBegin(GL_LINE_LOOP)
	for v in loop3:
		glVertex3f(v[0], v[1], v[2])
	glEnd()

@njit(fastmath=True)
def step(x1, y1, z1, vx, vy, vz, x2, y2, z2):

	r_x = (x2 - x1)
	r_y = (y2 - y1)
	r_z = (z2 - z1)

	r = (r_x ** 2 + r_y ** 2 + r_z ** 2)**0.5

	r = max(r, MIN_DIST)

	vx += ((GRAVITY * DELTA_T) / r) * r_x / r
	vy += ((GRAVITY * DELTA_T) / r) * r_y / r
	vz += ((GRAVITY * DELTA_T) / r) * r_z / r

	# vx -= vx*FRICTION/r**3
	# vy -= vy*FRICTION/r**3
	# vz -= vz*FRICTION/r**3

	x1 += vx * DELTA_T
	y1 += vy * DELTA_T
	z1 += vz * DELTA_T

	return x1, y1, z1, vx, vy, vz,r

def run(vertices, gravity_well):
	for v in vertices:
		v.x, v.y, v.z, v.vx, v.vy, v.vz,r = step(v.x, v.y, v.z, v.vx, v.vy, v.vz, *gravity_well)
		if r<10:
			vertices.remove(v)
			vertices.append(*initializeVertices(1,RADIUS))
	drawVertices(vertices)
	#drawCenter(*gravity_well)

def main():
	global lastPosX, lastPosY, zoomScale, xRot, yRot, zRot
	pygame.init()

	gwell_1 = [0, 0, 0]
	gwell_2 = [10,0,0]

	vertices = initializeVertices(PARTICLES, RADIUS)

	display = (1280, 720)

	pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
	gluPerspective(45, (display[0] / display[1]), 0.1, RADIUS*3)
	glTranslatef(0.0, 50, -RADIUS*2)
	glRotatef(75, -1, 0.25, 0.5)

	clock = pygame.time.Clock()

	while True:
		clock.tick(31)
		pygame.display.set_caption("%s  FPS: %.2f" % ('Particles - Single Gravity Well', clock.get_fps()))
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
					gwell_1[0] += 5
				elif event.key == pygame.K_KP4:
					gwell_1[0] -= 5
				elif event.key == pygame.K_KP8:
					gwell_1[1] += 5
				elif event.key == pygame.K_KP2:
					gwell_1[1] -= 5
				elif event.key == pygame.K_KP9:
					gwell_1[2] += 5
				elif event.key == pygame.K_KP3:
					gwell_1[2] -= 5
				print(gwell_1)
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
		run(vertices,gwell_1)
		pygame.display.flip()


if __name__ == '__main__':
	main()