import bruce, numpy as np


t = np.linspace(2461207.9023358147-0.1,2461208.0888338615+0.1, 100)
t_zero_PB_5 : 2461208.0888338615
t_zero_PC_2 : 2461207.9023358147
radius_1_PB : 0.01941760725285102
radius_1_PC : 0.025465135681673523
radius_1_PD : 0.03348806438619062
k_PB : 0.09212737829034653
k_PC : 0.11692356708629292
k_PD : 0.06374976913950997
b_PB : 0.1235499999339763
b_PC : 0.2252024534692364
b_PD : 0.2167302478971137
qlp_light_3 : 0.24213850164201606


planet_params_b = {'t_zero' : 2461208.0888338615, 'period' : 30,
                 'radius_1' :  0.01941760725285102, 'k' : 0.09212737829034653,
                 'incl' : np.arccos(0.01941760725285102*0.1235499999339763), 'e' : 0, 'w' : np.pi/2}

planet_params_c = {'t_zero' : 2461207.9023358147, 'period' : 30,
                 'radius_1' :0.025465135681673523, 'k' : 0.11692356708629292,
                 'incl' : np.arccos(0.025465135681673523* 0.2252024534692364), 'e' : 0., 'w' : np.pi/2}
bruce.visual.animate_orbit(t=t, planets_params=[planet_params_b,planet_params_c], zoom_primary=True,save_gif=False, gif_path='orbit.gif')