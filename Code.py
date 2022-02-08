import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats as stats

###############################################################################
'''Ball class makes the ball objects that have no potential energy in between,
and collide elastically. This allows the ball to mimic the real balls movement '''
############################################################################### 

class Ball():
    counter = 0
    time = 0
    def __init__(self, mass, radius, position, velocity, clr = 'r'):
        Ball.counter += 1
        self.__mass = mass
        self.__radius = float(radius)
        self.__position = np.array(position, dtype = 'float')
        self.__velocity = np.array(velocity, dtype = 'float')
        self.__patch = plt.Circle(self.__position, self.__radius, fc =clr)
        
        
    def pos(self):
        return self.__position

    def vel(self):
        return self.__velocity
    
    def mass(self):
        return self.__mass
    
    def rad(self):
        return self.__radius
    
    def set_rad(self, new_rad):
        self.__radius = new_rad
        
    def KE(self):
        m = self.mass()
        v = self.vel()
        T = 0.5 * m * np.dot(v, v)
        return T
    
   
    def move(self, dt):
        self.__position += self.__velocity * dt
        self.__patch.center = self.pos()
        Ball.time += dt
        
    def time_to_collision(self, other):
        m1 = self.mass()
        m2 = other.mass()
        r = self.pos()-other.pos()
        v = self.vel()-other.vel()
        R1 = self.rad()
        R2 = other.rad()
        #a dt^2 +b dt +c =0
        a = np.dot(v,v)
        b = 2 * np.dot(r, v)
        
        if m1 > 0 and m2 > 0:
            c = np.dot(r,r) - (R1+R2)**2
            D = b**2 - 4*a*c
            if D > 0:
                dt = (-b-np.sqrt(D))/(2*a)
                if dt > 0:
                    return dt
            else:
                return 1000
            
        if m1 == 0 or m2 == 0:
            c = np.dot(r,r) - (R1-R2)**2
            D = b**2 - 4*a*c
            if D > 0:
                dt = (-b+np.sqrt(D))/(2*a)
                if dt > 0:
                    return dt
            else:
                return 1000
           
        else:
            return 1000
        #to differentiate ball to ball time from ball to container time
        #1000(large value) was returned if the ball pair doesn't collide
    
    def collide(self, other):
        r1, r2 = self.pos(), other.pos()
        v1, v2 = self.vel(), other.vel()
        m1, m2 = self.mass(), other.mass()
        r = r1-r2
        v = v1-v2
        R = np.linalg.norm(r)
    
        v1f = v1-((2*m2)/(m1+m2))*((np.dot(v,r))/R**2)*r
        v2f = v2+((2*m1)/(m1+m2))*((np.dot(v,r))/R**2)*r
        #for ball to ball collision with finite mass
        
        rhat = r/R
        vca = -v -2*(np.dot(-v,rhat))*rhat
        #for ball to container collision
        
        if m1 > 0 and m2 > 0:
            self.__velocity = v1f
            other.__velocity = v2f
            
        
        if m1 == 0 or m2 == 0:
            other.__velocity = vca
            return vca-(-v)
        #later be used for momentum calculation
          
    def get_patch(self):
        return self.__patch 
    
   
       
class Container(Ball):
    def __init__(self, radius):
        mass = 0
        radius = radius
        position = (0,0)
        velocity = (0,0)
        self.__patch1 = plt.Circle(position, radius, fc='b', fill = False, ls = 'solid')
        Ball.__init__(self, mass, radius, position, velocity, clr='b')
    
    def get_patch(self):
        return self.__patch1
###############################################################################
'''class Gas takes a list of balls as its argument and performs tasks: 
init_figure, animate: sets up the animation (both smooth and 'jumping' to 
collision version)

time: initiate the collision process by calculating time to next collision, 
the collision pair involved, and the time, which functions as a clock, from 
the initial settings of the balls

jump(small, pair, time): small: time to next collision, pair: collision pair, 
time: clock. This method moves the time forward by the equivalent amount to 
time to next collision. This method is critical, as it allows us to ignore
calculations in between the collisions, making more resources available. This 
returns useful quantities like change in momentum and velocity, which will be 
used later


animate(small, pair): this method does similar thing as the jump method, but 
instead of 'jumping' between the collision, it moves the time forward by 
small amount(time step), allowing us to present smoother animation

next_frame(i): this is used for animation purpose. It prepares the next frame of
the animation. 
'''
############################################################################### 
class Gas():
    def __init__(self, balls):
        #fisrt item in balls must be the container
        self.__balls = balls
        self.__text0 = None
        self.__time = 0
        #Time elapsed in animation
        self.__clock = 0
        #Time elapsed
        
    def init_figure(self):
        Cont = plt.Circle((self.__balls[0].pos()), self.__balls[0].rad(), ec = 'b', fill = False, ls = 'solid')
        ax.add_artist(Cont)
        self.__text0 = ax.text(-9.9,9,"f={:4d}".format(0,fontsize=12))
        patches = [self.__text0]
        
        for ball in self.__balls:
            pch = ball.get_patch()
            ax.add_patch(pch)
            patches.append(pch)

        return patches

    def time(self):
        time_to_collision = 100
        time = 0
        for i in range(len(self.__balls)):
            for j in range(i+1, len(self.__balls)):
                t = self.__balls[i].time_to_collision(self.__balls[j])
                if t < time_to_collision:
                    time_to_collision = t
                    pair = [i, j]
        return time_to_collision, pair, time
    
    def jump(self, time_to_collision, pair, time):
        time += time_to_collision
        
        velocity = []
        
        for i in range(1, len(self.__balls)):
            vel = self.__balls[i].vel()
            vel = np.linalg.norm(vel)
            velocity.append(vel)
        
        for ball in self.__balls:
            ball.move(time_to_collision)
            
        change_momentum = 0
        
        delta_v = self.__balls[pair[0]].collide(self.__balls[pair[1]])
        
        if pair[0] == 0:
            change_momentum += np.linalg.norm(delta_v)*self.__balls[pair[1]].mass()
        
        #recalculates the time and pair
        time_to_collision = 100
        for i in range(len(self.__balls)):
                for j in range(i+1, len(self.__balls)):
                    t = self.__balls[i].time_to_collision(self.__balls[j])
                    if t < time_to_collision:
                       time_to_collision = t
                       pair = [i, j]
        
        return time_to_collision, pair, time, change_momentum, velocity
    
    def animate(self, time_to_collision, pair):
        dt = 0.1
        self.__time += dt
        self.__clock += dt
        
        if abs(self.__time - time_to_collision)< dt:
            self.__time = 0
            self.__balls[pair[0]].collide(self.__balls[pair[1]])
            time_to_collision = 100
            for i in range(len(self.__balls)):
                for j in range(i+1, len(self.__balls)):
                    t = self.__balls[i].time_to_collision(self.__balls[j])
                    if t < time_to_collision:
                       time_to_collision = t
                       pair = [i, j]
            
        for ball in self.__balls:
            ball.move(dt)
  
        return time_to_collision, pair
         
       
            
        
    def next_frame(self, i):
        self.__text0.set_text("f={:4d}".format(i))
        patches = [self.__text0]
        
        global x
        #x = self.jump(x[0], x[1], x[2])
        x = self.animate(x[0], x[1])
        for ball in self.__balls:
            patches.append(ball.get_patch())   
        return patches
        
###############################################################################
'''ball_list_creator takes the arguements and create a list of particles without
overlapping and with average velocity = 0 (Used recursion)   
'''
###############################################################################    
def ball_list_creator(number_of_particles, mass, radius, cont_radius):
    
    N = number_of_particles
    m = mass
    r = radius
    balls =[Container(cont_radius)]
    
    #effective radius of container
    R = balls[0].rad()- r
    
    #To make a random velocity with the average velocity 0 
    a,b = np.random.random(int(N/2)), np.random.random(int(N/2))
    a /= a.sum()
    b /= -b.sum()
    #normalize
    vx = np.concatenate([a,b])
    vx *= N/2
    #multiply by a factor (would not change the average velocity)
    np.random.shuffle(vx)
    c,d = np.random.random(int(N/2)), np.random.random(int(N/2))
    c /= c.sum()
    d /= -d.sum()
    vy = np.concatenate([c,d])
    vy *= N/2
    np.random.shuffle(vy)
    
    
    for i in range(N):
        balls.append(Ball(m,r,getNewXy(balls,R,r),(vx[i],vy[i])))
    
    return balls

def getNewXy(balls, effective_radius, ball_rad):
    er = effective_radius
    brad = ball_rad
    x = np.random.uniform(-er,er)
    y = np.random.uniform(-er,er)
    distance = np.sqrt((x*x)+(y*y))
    
    if distance > er:
        return getNewXy(balls, er, brad)
    else:
        need_change = 0
        #identifier for the overlap between the balls (0= false, 1= true)
        for n in range(1,len(balls)):
            if np.linalg.norm(balls[n].pos()-(x,y))<brad*2:
                need_change = 1
                
        if need_change == 1:
            return getNewXy(balls,er,brad)
        
        else:
            return (x,y)
                



###############################################################################
'''Class Simulation performs various calculation(experimentation), from 
    arguments balls and desired number of collision of particles
    
    1. PT: relationship between pressure and temperature. This actually does not
    have any outputs; however, it is essential for other methods to work (i.e.
    PT_VN, Ideal)
    2. Maxwell_Boltzmann: Calculation of variance of velocities and parameters 
    in Maxwell-Boltzmann distribution. Histogram of 'experimental' velocities is 
    compared with the 'analytic' Maxwell-Boltzmann distribution, derived from
    the temperature, kinetic energy and Boltzmann constant.
    3. PT_VN: Relationship between pressure and volume or the number of particles, 
    or between temperature and volume or the number of particles. 
    4. Ideal
    5. van
    '''
###############################################################################

class Simulation():
    def __init__(self, number_of_balls, mass, radius, cont_radius, number_of_collision):
        self.__balls = ball_list_creator(number_of_balls, mass, radius, cont_radius)   
        self.__movie = Gas(self.__balls)
        self.n = number_of_collision
        self.N = len(self.__balls)-1
        #minus one for the container
        self.kb = 1.38E-23
        
    def PT(self):
        delta_p = 0
        KE = 0
        
        for ball in self.__balls:
            KE += ball.KE()
            
        x = self.__movie.time()    
        for i in range(self.n):
            x = self.__movie.jump(x[0],x[1], x[2])
            delta_p += x[3]
            #append changes in momentum every jumping step (only considering ball to container collision)
        
        
        time = x[2]
        #total time elapsed
        
        Force= delta_p/time
        Pressure = Force/(2*np.pi*self.__balls[0].rad())
        Temperature = KE/(self.N*self.kb)
        
        print('Time=', time, 'Pressure=',Pressure, 'Temperature=',Temperature)
        return Pressure, Temperature
    
      
    def Maxwell_Boltzmann(self, bins):
        x = self.__movie.time()
        velocity =[]
        velocity_squared=[]
        
        for i in range(self.n):
            x = self.__movie.jump(x[0],x[1],x[2])
            velocity.extend(x[4])
        #brings the velocity list for every iteration
        KE = 0
        for ball in self.__balls:
            KE += ball.KE()
        
        Temperature = KE/(self.N*self.kb)
        alpha = self.__balls[1].mass()/(2*self.kb*Temperature)
        A = np.sqrt(alpha/np.pi)
        
        Var = ((np.pi*A*A)/(alpha*alpha))*(1-((np.pi*np.pi*A*A)/(4*alpha)))
        
        for v in velocity:
            velocity_squared.append(v**2)
        
        mean_of_squares = np.mean(velocity_squared)
        rms = np.sqrt(mean_of_squares)
        print ('Experimental Variance is', np.var(velocity))
        print ('Analytical Variance is', Var)
        print ('Experimental RMS is', rms)
        print ('Analytical RMS is', 1/np.sqrt(alpha))
        
        vmax = max(velocity)
        v = np.linspace(0, vmax, 1000)
        prob_density = (A**2) * 2* np.pi * v * np.exp(- alpha * v**2)
        plt.figure()
        ax = plt.axes()
        ax.set_xlabel('Speed (m/s)', fontsize=14)
        ax.set_ylabel('Probability Density', fontsize=14)
        plt.hist(velocity, bins=bins, density = True, stacked=True)
        plt.plot(v, prob_density)
        plt.axis([0, vmax, 0, 0.7])
        plt.show()
 
    def PT_VN(self, independent_variable, iteration):
        if independent_variable =='volume':
            Pressure = []
            Temperature = []
            Volume = []
            init_cont_radius = self.__balls[0].rad()
            for i in range(iteration):
               self.__balls[0].set_rad(init_cont_radius+i)
               Volume.append(self.__balls[0].rad()*self.__balls[0].rad()*np.pi)
               op = self.PT()
               Pressure.append(op[0])
               Temperature.append(op[1])
               
            
            print('Pressure (Nm^-2)=',Pressure, 'Temperature(K)=',Temperature, 'Volume(m^2)=',Volume)
            
            fig = plt.figure(1)
            ax = plt.axes()
            ax.set_xlabel('V (m^2)')
            ax.set_ylabel('P (Nm^-2)')
            plt.plot(Volume, Pressure)
            
            fig = plt.figure(2)
            ax = plt.axes()
            ax.set_xlabel('V (m^2)')
            ax.set_ylabel('T (K)')
            plt.plot(Volume, Temperature)
            plt.show()
            
            return Pressure, Temperature, Volume
        
        elif independent_variable =='number':
            Pressure = []
            Temperature = []
            Number = []
            for i in range(iteration):
                self.__balls = ball_list_creator(100+i, 1, 1, 50)
                op = self.PT()
                Pressure.append(op[0])
                Temperature.append(op[1])
                Number.append(self.N)
                
            print('Pressure (Nm^-2)=',Pressure, 'Temperature(K)=',Temperature, 'Number=',Number)
            fig = plt.figure(1)
            ax = plt.axes()
            ax.set_xlabel('N')
            ax.set_ylabel('P (Nm^-2)')
            plt.plot(Number, Pressure)
            
            fig= plt.figure(2)
            ax.set_xlabel('N')
            ax.set_ylabel('T (K)')
            plt.plot(Number, Temperature)
            plt.show()
            return Pressure, Temperature, Number
        
             
    def Ideal(self, iteration):
        op = self.PT_VN('volume', iteration)
        Pressure = op[0]
        Temperature = op[1]
        Volume = op[2]
        constant = []
        Boltzmann =[]
        for i in range(iteration):
            constant.append(Pressure[i]*Volume[i]/Temperature[i])
            
        for i in range(iteration):
            Boltzmann.append(constant[i]/self.N)
            
        print('N times k must be',constant)
        print('Boltzmann constant is', Boltzmann)
        
        

    def van(self, iteration):
        op = self.PT_VN('volume', iteration)
        Pressure = op[0]
        Reciprocal_Pressure = []
        for p in Pressure:
            Reciprocal_Pressure.append(1/p)
        Temperature = op[1]
        Volume = op[2]
        
        
        
        slope, intercept, r_value, p_value, std_err= stats.linregress(Volume,Reciprocal_Pressure)
        
        print('slope=1/NkT=', slope)
        print('intercept=-b/NkT=', intercept)
        print('r-squared value=', r_value*r_value)
        print('std=', std_err)        
        print('y=mx+c, m=',1/(self.N*self.kb*Temperature[1]), 'c=-b/NkT=',(-np.pi*self.__balls[1].rad()*self.__balls[1].rad())/(self.kb*Temperature[1]))
        
        fig = plt.figure(3)
        plt.plot(Volume,Reciprocal_Pressure,'o', label='original data')
       
        plt.legend()
        plt.show()
        
        
        
        
 
        
        
        


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anim", default=0, type=int)
    parser.add_argument("--num", default=10, type=int, help="number of balls must be even number")
    parser.add_argument("--mass", default=1, type=int)
    parser.add_argument("--rad", default=0.5, type=float)
    parser.add_argument("--cont_rad", default=10, type=int)
    parser.add_argument("--exp", default="mb", type=str, help="options: mb, van, ptvn, ideal")
    parser.add_argument("--num_col", default=2000, type=int)

    args = parser.parse_args()
    
    
    ANIMATE = bool(args.anim)

    if ANIMATE == False:
        #N, M, R, ContR, Number of collision
        S = Simulation(args.num, args.mass, args.rad, args.cont_rad, args.num_col)
        if args.exp == "mb":
            S.Maxwell_Boltzmann(50)
        if args.exp == "van":
            S.van(10)
        if args.exp == "ptvn":
            S.PT_VN('volume', 20) # or  S.PT_VN('volume', 20)
        if args.exp == "ideal":
            S.Ideal(5)
        
    if ANIMATE == True:
        A = ball_list_creator(args.num, args.mass, args.rad, args.cont_rad)
        fig = plt.figure()
        ax = plt.axes(xlim=(-A[0].rad(), A[0].rad()), ylim=(-A[0].rad(), A[0].rad()))
        ax.axes.set_aspect('equal')  
    
    
          
        movie = Gas(A)
        global x
        x = movie.time()
    
        anim = animation.FuncAnimation( fig, 
                                    movie.next_frame, 
                                    init_func = movie.init_figure, 
                                    #frames = 1000, 
                                    interval = 50,
                                    blit = True)

        plt.show()
      



