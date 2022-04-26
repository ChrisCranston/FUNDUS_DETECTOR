# ---------------------------------------------
# classify.py - 
# File for controlling the fuzzy logic used in the recommendation system
# variables in: haem_in [int] the number of haemorrhages detected by the system, 
#               exud_in [bool] the number of exudates detected by the system
#
# @Author - Chris Cranston W18018468
# ---------------------------------------------

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def fuzzy_classifier(haem_in, exud_in):
   # create antecedents and consequent
   x_haemhorrage = ctrl.Antecedent(np.arange(0, 50, 1), 'haemorrhage_risk')
   x_exudate = ctrl.Antecedent(np.arange(0, 50, 1), 'exudate_risk')
   x_NDESP = ctrl.Consequent(np.arange(0, 100, 1), 'NDESP_rating')

   # create fuzzy membership functions
   x_haemhorrage['lo_risk'] = fuzz.trimf(x_haemhorrage.universe, [0, 0, 2])
   x_haemhorrage['md_risk'] = fuzz.trimf(x_haemhorrage.universe, [1, 4, 7])
   x_haemhorrage['hi_risk'] = fuzz.trapmf(x_haemhorrage.universe, [3, 7,50,50])
   x_exudate['lo_risk'] = fuzz.trimf(x_exudate.universe, [0, 0, 1])
   x_exudate['md_risk'] = fuzz.trimf(x_exudate.universe, [1, 2, 3])
   x_exudate['hi_risk'] = fuzz.trapmf(x_exudate.universe, [2, 3,50,50])
   x_NDESP['R0'] = fuzz.trapmf(x_NDESP.universe, [0,0, 10, 40])
   x_NDESP['R1'] = fuzz.trimf(x_NDESP.universe, [30, 50, 70])
   x_NDESP['R2'] = fuzz.trapmf(x_NDESP.universe, [60, 90, 100,100])

   #    # RULES:
   #    #    1) if exudates or haemorrhage is high then r2
   #    #    2) if one is high and the other is medium then r2 
   #    #    3) if both are medium then r1 
   #    #    4) if one is low and one is medium r1 
   #    #    5) if both are low then r0 

   rule1 = ctrl.Rule(x_haemhorrage['hi_risk'] | x_exudate['hi_risk'], x_NDESP['R2'])
   rule2 = ctrl.Rule(x_haemhorrage['md_risk'] & x_exudate['md_risk'] , x_NDESP['R1'])
   rule3 = ctrl.Rule(x_haemhorrage['md_risk'] | x_exudate['md_risk'] , x_NDESP['R1'])
   rule4 = ctrl.Rule(x_haemhorrage['lo_risk'] & x_exudate['lo_risk'] , x_NDESP['R0'])
   rule5 = ctrl.Rule(x_haemhorrage['lo_risk'] | x_exudate['lo_risk'] , x_NDESP['R0'])

   # Create control system with defined rules
   ndesp_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
   ndesp = ctrl.ControlSystemSimulation(ndesp_ctrl)

   # parse input from object detection system
   ndesp.input['haemorrhage_risk'] = haem_in
   ndesp.input['exudate_risk'] = exud_in
   
   # compute result
   ndesp.compute()

   # return rating
   return ndesp.output['NDESP_rating']
   