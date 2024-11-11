#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import gym
import pybullet as p
import pybullet_data
import math
import math
import random
import sympy as sym




class robot_env(gym.Env):
    def __init__(self):
        
        #print('Inicializa entorno')
        
        #Espacio de acciones
        self.action_space=gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,6), dtype=np.float32) #Momentos o torques (N*m) low=-150.0, high=150.0   

        #Espacio de observaciones
        self.observation_space=gym.spaces.Box(low=-2, high=2, shape=(1,3), dtype=np.float32) #Posición absoluta efector final
                                        
        #Cliente Pybullet
        self.physicsClient = p.connect(p.GUI)
      
        #posición base
        self.base_ini_pos = np.array([0.0, 0.0, 1.0])

        self.nb_episodes = 0 #number of performed episodes
        self.sampled_transitions = None

        #Diccionario de articulaciones
        self.dicc_joints = {'jointIndex':[], 'jointName':[], 'jointType':[], 'qIndex':[], 'uIndex':[], 'flags':[],
       'jointDamping':[], 'jointFriction':[], 'jointLowerLimit':[], 'jointUpperLimit':[],
       'jointMaxForce':[], 'jointMaxVelocity':[], 'linkName':[], 'jointAxis':[],
       'parentFramePos':[], 'parentFrameOrn':[], 'parentIndex':[]}

        self.dicc_manipulator_parameters = {'link':[i for i in range(1, 7)], 'a_i':[0, 0.425, 0.39225, 0, 0, 0],
        'alpha_i':[math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0],
        'd_i':[0.089159, 0, 0, 0.10915, 0.09465, 0.0823], 
        'theta_i':[sym.Symbol('theta_1'), sym.Symbol('theta_2'), sym.Symbol('theta_3'), 
                   sym.Symbol('theta_4'), sym.Symbol('theta_5'), sym.Symbol('theta_6')],
        'd_theta_i':[math.pi/10, math.pi/10, math.pi/10, math.pi/10, math.pi/10, math.pi/10]}
        
        self.time_step = 1/240 #periodo de simulación. Default frequency: 240 Hz 
        self.sigma = 0.05 #Sigma inicial
        self.total_reward = 0
        self.total_steps = 0 #total steps performed
    
        self.reset()



    def step(self, action):
        #print('\nHa entrado en step()')
        
        old_state = self.obs

        #Comandar acción
        #print('\naccion: ', action)
        #Convierte a escala [-150, 150]
        self.command_arm(action*self.action_space.high[0])

        p.stepSimulation() 
        
        #Base cambia de posición, siguiendo trayectoria senoidal
        #self.trayectoria()
        
        #Nueva observación:
        link_state = p.getLinkState(self.robot, 6) #Información sobre el efector final
        self.obs = np.array(link_state[4]).reshape(1,3)
        quaternion = np.array(link_state[5]).reshape(1,4)
        
        #Velocidades y posiciones angulares de las articulaciones:
        joint_states = p.getJointStates(self.robot, range(1, 7))
        ang_vel_array = np.array([state[1] for state in joint_states])
        ang_pos_array = np.array([state[0] for state in joint_states])

        #Velocidades lineales de los enlaces:
        vel_array, _ = self.direct_kinematics_and_link_velocities(ang_pos_array, ang_vel_array) #return v_link_vector, w_link_vector
        vel_array = np.transpose(vel_array)

        #Nuevo goal:
        #if self.sampled_transitions is not None:
            #self.goal = self.goals[self.steps] #Cambia de goal cada step
            #print('\nGoals: ', self.goals)


        #CREAR ESFERA EN GOAL
        # Crear solo la forma visual de la esfera
        sph_radius = 0.05  # Radio de la esfera
        sph_visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=sph_radius)

        # Crear el cuerpo de la esfera sin colisión (collision shape = -1)
        # La masa sigue siendo 0 para que no se mueva
        sphere_id = p.createMultiBody(0, -1, sph_visual_shape_id, self.goal[0])

        # Cambiar el color de la esfera
        p.changeVisualShape(sphere_id, -1, rgbaColor=[1, 0, 0, 1])  # Rojo



        #Calcular distancia posición
        euclid_pos = self.euclid(self.obs[0], self.goal[0])
        
        
        
        # Convertir la posición del link al sistema de la base
        pos_ef_base = p.multiplyTransforms(self.base_pos,
                                                  p.invertTransform(self.base_pos, self.base_orn)[1],
                                                  self.obs[0, 0:3],
                                                  [0, 0, 0, 1]
                                                  )[0]

        
        #FLAG REGIÓN PROHIBIDA

        #Singularidades:
            #Singularidades de muñeca:
            #theta_5=0 ó theta_5=+-pi

            #Singularidades de codo:
            #theta_3=0 ó theta_3=+-pi
            
        #Definir región de trabajo:
            #Radio máximo: 850 mm
            #Plano superior paralelo a la horizontal 
            #Plano inferior paralelo a la horizontal en la base del manipulador
            #Plano vertical 
            #Amplitud máxima de 240 grados [-120, 120]
            
        #Amplitud máxima de 240 grados [-120, 120]:
         
        flag={'valor': False, 'indice': np.nan, 'causa': ''}
        
        #Flag debajo de la base:
        for joint_index in range(1, 7):
            link_state = p.getLinkState(self.robot, joint_index)
            if link_state[4][2] <= self.base_pos[2]:
                flag={'valor': True, 'indice': joint_index, 'causa': "Debajo_base"}

        
        
        #Ángulo 3a articulación (theta_3):
        theta_3 = p.getJointState(self.robot, 3)[0]

        if abs(theta_3) == np.pi or abs(theta_3) == 0.0:
            flag={'valor': True, 'indice': 3, 'causa': "singularidad articulación 3. theta_3="+str(theta_3)}
        

        #Ángulo 5a articulación (theta_5):
        theta_5 = p.getJointState(self.robot, 5)[0]     

        if abs(theta_5) == np.pi or abs(theta_5) == 0.0:
            flag={'valor': True, 'indice': 5, 'causa': "singularidad articulación 5. theta_5="+str(theta_5)}

        

        #RECOMPENSA:
        self.reward, dc_reward_info = self.reward_function(self.goal[0], self.obs[0], euclid_pos, quaternion, flag, ang_vel_array)
        #print('\nRecompensa: ', self.reward)
        self.steps+=1
        self.total_steps+=1
        #Calcula sigma:
        self.calcula_sigma(sigma_min=0.01, reduction_rate=0.5, reward_threshold=4000, check_interval=250)
        
        #Finalización por quedarse sin intentos
        if self.steps>=250 or flag['valor'] == True:
            self.done=True
            self.nb_episodes+=1

            
        info={'quaternion': quaternion}
        
        
            
        return old_state, self.reward, self.obs, self.done, self.goal, flag, ang_vel_array, info 



    def reset(self):
        #print('\nha entrado en reset()')
        p.resetSimulation(self.physicsClient)
        

        #Cargar plano y modelo del brazo
        self.planeId = p.loadURDF("C:/Users/diego/anaconda3/Lib/site-packages/pybullet_data/plane.urdf")
        
        self.base_pos = self.base_ini_pos
        self.base_orn = p.getQuaternionFromEuler([0,0,0])
        
        self.robot = p.loadURDF(fileName="C:/Users/diego/Documents/MIAR/TFM/models/x100_with_ur5/urdf/ur5e.urdf", 
                    basePosition=self.base_pos, baseOrientation=self.base_orn)

        if self.nb_episodes == 0:
            self.num_joints = 6

            #Diccionario articulaciones
            for joint in range (1, self.num_joints+1):
                #p.setJointMotorControl2(robot,joint, p.VELOCITY_CONTROL,force=0)
                JointInfo = p.getJointInfo(self.robot, joint)
                for key, item in zip(self.dicc_joints.keys(), JointInfo):
                    self.dicc_joints[key].append(item)

            #DataFrame articulaciones
            self.df_joints=pd.DataFrame(self.dicc_joints)
            self.df_joints.set_index(keys="jointIndex", inplace=True, drop=True)

            #DataFrame manipulator parameters
            self.df_manipulator_parameters = pd.DataFrame(self.dicc_manipulator_parameters)
            self.df_manipulator_parameters.set_index(keys="link", inplace=True, drop=True)
        
        p.setGravity(0, 0, -9.81)
        
        for joint in range (0, self.num_joints):
            p.setJointMotorControl2(self.robot,joint, p.VELOCITY_CONTROL,force=0)
        
        target_ef_pos = np.array([[0.4, 0.0, 1.6]]) #Posición target del efector final con respecto al sistema global (m)

        #Goal
        self.goal = target_ef_pos
        
        self.steps=0 #Ejecuciones por episodio
        self.done=False #Indica si el episodio ha finalizado

        #Observación inicial:
        link_state = p.getLinkState(self.robot, 6) #Información sobre el efector final
        self.obs = np.array(link_state[4]).reshape(1,3)
        quaternion = np.array(link_state[5]).reshape(1,4)
 
        #print(self.obs)
        info={'quaternion': quaternion}
        
        
        return self.obs



    def command_arm(self, action):
        for joint_index, pos in zip(range(1, 7), action):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pos)
            



    def reward_function(self, goal, obs, euclid_pos, quaternion, flag, ang_vel_array):

        if flag['valor'] == True:
            if flag['indice'] == 6:
                reward = -150
            else:
                reward = -100

            msg = "Región prohibida. Causa: " + flag['causa']

            return reward, {'msg': msg}

             

        #Función de recompensa

        #Escalado distancia euclídea
        Dpt = euclid_pos
        Dpt_esc = Dpt/0.9 #[0, 1]

        
        #Cálculo de R_position
        beta = 0.05  #Ajustable

        if Dpt >= beta:
            J_reach = 0
        elif Dpt < beta:
            J_reach = 1

        R_position = J_reach - Dpt
            
        #Cálculo de R_move

        R_move = self.time_step*(ang_vel_array@np.transpose(ang_vel_array))/self.num_joints


        #Cálculo de R_stride

        R_stride = R_position - R_move

        #Cálculo de R_direction
        #Coordenadas target:

        Tx = goal[0]
        Ty = goal[1]
        Tz = goal[2]
                    
        #Coordenadas efector final:
        Px = obs[0]
        Py = obs[1]
        Pz = obs[2]
                
        #Cuaternio efector final:
        #print('quaternion: ', quaternion)
        Pq_x = quaternion[0][0]
        Pq_y = quaternion[0][1]
        Pq_z = quaternion[0][2]
        Pq_w = quaternion[0][3]



        #Expected direction:
        VPT = np.array([Tx-Px, Ty-Py, Tz-Pz])
        #VPT = np.array([0, 0, -1])
        if np.linalg.norm(VPT) != 0:
            VPT_norm = VPT/np.linalg.norm(VPT)
        else:
            VPT_norm = VPT
        #print('VPT: ', VPT)

        temp = np.sin(np.arccos(Pq_w))

        #eje de rotación dado el cuaternio:
        VPPq = np.array([Pq_x/temp, Pq_y/temp, Pq_z/temp])
        VPPq_norm = VPPq/np.linalg.norm(VPPq)
        #print('VPPq: ', VPPq)

        #Angulo de desviacion: 
        gamma = abs(np.arccos(np.dot(VPT_norm, VPPq_norm)))
        assert gamma <= np.pi
        assert gamma >= 0


        if math.isnan(gamma) == True:
            print('\ngamma: ', gamma)
            print('\nobs: ', obs)
            print('\ngoal: ', goal)
            print('\nVPT_norm: ', VPT_norm)
            print('\nVPPq_norm: ', VPPq_norm)

       

        if gamma < np.pi/2:
            R_direction = gamma/2*np.pi

        elif gamma >= np.pi/2:
            R_direction = (np.pi - gamma)/2*np.pi


        #Cálculo de R_posture
        R_posture = R_position - R_direction

        #Cálculo de R_sar
        alpha_1 = 1 - Dpt_esc
        alpha_2 = Dpt_esc


        if euclid_pos < 0.05: #if euclid_pos < 0.05 and delta_theta_deg < 5:
            msg = "Se ha alcanzado el objetivo!!"
            return 200, {'msg': msg, 'delta_theta_deg': delta_theta_deg, 'euclid_pos': euclid_pos, 'VPT_norm': VPT_norm, 'VPPq_norm': VPPq_norm}

        reward = alpha_1*R_stride + alpha_2*R_posture
        msg = "Aun no se ha alcanzado el objetivo"

        #print('\nangle: ', gamma)
        #print('\neuclid_pos: ', euclid_pos)
        #print('\nreward: ', reward)
        #print('\nR_stride: ', R_stride)
        #print('\nR_posture: ', R_posture)
        #print('\nR_postion: ', R_position)
        #print('\nR_move: ', R_move)
        #print('\nR_direction: ', R_direction)

        return reward, {'msg': msg, 'delta_theta_deg': delta_theta_deg, 'euclid_pos': euclid_pos, 'VPT_norm': VPT_norm, 'VPPq_norm': VPPq_norm}



    def euclid(self, pos, target):
        return np.linalg.norm(np.array(pos) - np.array(target))


    #Función para la reducción de la varianza Ornstein-Uhlenbeck
    def calcula_sigma(self, sigma_min, reduction_rate, reward_threshold, check_interval):
        self.total_reward += self.reward
    
        # Cada check_interval pasos, calcula el promedio de recompensa
        if self.total_steps % check_interval == 0 and self.total_steps > 0:
            average_reward = self.total_reward / check_interval
        
            # Si la recompensa promedio es alta, reduce sigma
            if average_reward >= reward_threshold:
                self.sigma = max(self.sigma * reduction_rate, sigma_min)
                print(f"Reduciendo sigma a {self.sigma} debido a recompensa promedio alta")
        
            # Reinicia el total de recompensas para el siguiente intervalo
            self.total_reward = 0

    def close(self):
        p.disconnect(self.physicsClient)

