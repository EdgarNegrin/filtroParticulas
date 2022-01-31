#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Rob�tica Computacional -
# Grado en Ingenier�a Inform�tica (Cuarto)
# Pr�ctica: Filtros de particulas.

from math import *

from cv2 import ml_LogisticRegression
from robot import *
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import select
from datetime import datetime
# ******************************************************************************
# Declaraci�n de funciones

def distancia(a,b):
  # Distancia entre dos puntos (admite poses)
  return np.linalg.norm(np.subtract(a[:2],b[:2]))

def angulo_rel(pose,p):
  # Diferencia angular entre una pose y un punto objetivo 'p'
  w = atan2(p[1]-pose[1],p[0]-pose[0])-pose[2]
  while w >  pi: w -= 2*pi
  while w < -pi: w += 2*pi
  return w

def pinta(secuencia,args):
  # Dibujar una secuencia de puntos
  t = np.array(secuencia).T.tolist()
  plt.plot(t[0],t[1],args)

def mostrar(objetivos,trayectoria,trayectreal,filtro):
  # Mostrar mapa y trayectoria
  plt.ion() # modo interactivo
  plt.clf()
  plt.axis('equal')
  # Fijar los bordes del gr�fico
  objT   = np.array(objetivos).T.tolist()
  bordes = [min(objT[0]),max(objT[0]),min(objT[1]),max(objT[1])]
  centro = [(bordes[0]+bordes[1])/2.,(bordes[2]+bordes[3])/2.]
  radio  = max(bordes[1]-bordes[0],bordes[3]-bordes[2])
  plt.xlim(centro[0]-radio,centro[0]+radio)
  plt.ylim(centro[1]-radio,centro[1]+radio)
  # Representar mapa
  for p in filtro:
    dx = cos(p.orientation)*.05
    dy = sin(p.orientation)*.05
    plt.arrow(p.x,p.y,dx,dy,head_width=.05,head_length=.05,color='k')
  pinta(trayectoria,'--g')
  pinta(trayectreal,'-r')
  pinta(objetivos,'-.ob')
  p = hipotesis(filtro)
  dx = cos(p[2])*.05
  dy = sin(p[2])*.05
  plt.arrow(p[0],p[1],dx,dy,head_width=.075,head_length=.075,color='m')
  # Mostrar y comprobar pulsaciones de teclado:
  plt.draw()
#  if sys.stdin in select.select([sys.stdin],[],[],.01)[0]:
#    line = sys.stdin.readline()
  input()

def genera_filtro(num_particulas, balizas, real, centro=[2,2], radio=3):
  # Inicializaci�n de un filtro de tama�o 'num_particulas', cuyas part�culas
  # imitan a la muestra dada y se distribuyen aleatoriamente sobre un �rea dada.
  # devuelve lista de particulas
  particulas = []
  for i in range(num_particulas):
    new = real.copy()
    # Añadimos la posicion
    newX = random.uniform(centro[0]-radio, centro[0]+radio)
    newY = random.uniform(centro[1]-radio, centro[1]+radio)
    newHorientation = random.uniform(-pi, pi)
    new.set(newX, newY, newHorientation)
    # Calculo del peso
    new.weight = real.measurement_prob(new.sense(balizas), balizas)
    particulas.append(new)
  return particulas

def dispersion(filtro):
  # Dispersion espacial del filtro de particulas
  minX = min([p.pose()[0] for p in filtro])
  maxX = max([p.pose()[0] for p in filtro])
  minY = min([p.pose()[1] for p in filtro])
  maxY = max([p.pose()[1] for p in filtro])
  
  return [minX, maxX, minY, maxY]


def peso_medio(filtro):
  # Peso medio normalizado del filtro de particulas
  return 0

# ******************************************************************************

random.seed(0)

# Definici�n del robot:
P_INICIAL = [0.,4.,0.] # Pose inicial (posici�n y orientacion)
V_LINEAL  = .7         # Velocidad lineal    (m/s)
V_ANGULAR = 140.       # Velocidad angular   (�/s)
FPS       = 10.        # Resoluci�n temporal (fps)
HOLONOMICO = 0         # Robot holon�mico
GIROPARADO = 0         # Si tiene que tener vel. lineal 0 para girar
LONGITUD   = .1        # Longitud del robot

N_PARTIC  = 50         # Tama�o del filtro de part�culas
N_INICIAL = 2000       # Tama�o inicial del filtro

# Definici�n de trayectorias:
trayectorias = [
    [[0,2],[4,2]],
    [[2,4],[4,0],[0,0]],
    [[2,4],[2,0],[0,2],[4,2]],
    [[2+2*sin(.4*pi*i),2+2*cos(.4*pi*i)] for i in range(5)],
    [[2+2*sin(.8*pi*i),2+2*cos(.8*pi*i)] for i in range(5)],
    [[2+2*sin(1.2*pi*i),2+2*cos(1.2*pi*i)] for i in range(5)],
    [[2*(i+1),4*(1+cos(pi*i))] for i in range(6)],
    [[2+.2*(22-i)*sin(.1*pi*i),2+.2*(22-i)*cos(.1*pi*i)] for i in range(20)],
    [[2+(22-i)/5*sin(.1*pi*i),2+(22-i)/5*cos(.1*pi*i)] for i in range(20)]
    ]

# Definici�n de los puntos objetivo:
if len(sys.argv)<2 or int(sys.argv[1])<0 or int(sys.argv[1])>=len(trayectorias):
  sys.exit(sys.argv[0]+" <�ndice entre 0 y "+str(len(trayectorias)-1)+">")
objetivos = trayectorias[int(sys.argv[1])]

# Definici�n de constantes:
EPSILON = .1                # Umbral de distancia
V = V_LINEAL/FPS            # Metros por fotograma
W = V_ANGULAR*pi/(180*FPS)  # Radianes por fotograma

real = robot()
real.set_noise(.01,.01,.01) # Ruido lineal / radial / de sensado
real.set(*P_INICIAL)

#inicializaci�n del filtro de part�culas y de la trayectoria
filtro = genera_filtro(10000, objetivos, real)
pose = hipotesis(filtro)# particula con mayor peso(funcion hipotesis) esta es la ideal
trayectoria = [pose]

trayectreal = [real.pose()]
mostrar(objetivos,trayectoria,trayectreal,filtro)

tiempo  = 0.
espacio = 0.
for punto in objetivos:
  while distancia(trayectoria[-1],punto) > EPSILON and len(trayectoria) <= 1000:
    
    w = angulo_rel(pose,punto)
    if w > W:  w =  W
    if w < -W: w = -W
    v = distancia(pose,punto)
    if (v > V): v = V
    if (v < 0): v = 0
    if HOLONOMICO:
      if GIROPARADO and abs(w) > .01:v = 0
      real.move(w,v)
      for i in filtro:
        i.move(w,v)
        # recalcular el messurementpro (porque se ponen a 1 los pesos)
        i.weight = real.measurement_prob(i.sense(objetivos), objetivos)
    else:
      real.move_triciclo(w,v,LONGITUD)
      for i in filtro:
        i.move_triciclo(w,v,LONGITUD)
        # recalcular el messurementpro (porque se ponen a 1 los pesos)
        i.weight = real.measurement_prob(i.sense(objetivos), objetivos)

  
    # remuestreo
    filtro = resample(filtro, N_PARTIC)
    # Seleccionar hip�tesis de localizaci�n y actualizar la trayectoria
    trayectreal.append(real.pose())
    trayectoria.append(pose)
    mostrar(objetivos,trayectoria,trayectreal,filtro)
    pose = hipotesis(filtro)
    

    espacio += v
    tiempo  += 1

if len(trayectoria) > 1000:
  print ("<< ! >> Puede que no se haya alcanzado la posici�n final.")
print ("Recorrido: "+str(round(espacio,3))+"m / "+str(tiempo/FPS)+"s")
print ("Error medio de la trayectoria: "+str(round(sum(\
    [distancia(trayectoria[i],trayectreal[i])\
    for i in range(len(trayectoria))])/tiempo,3))+"m")
input()

