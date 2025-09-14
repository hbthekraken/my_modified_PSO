import numpy as np
from SpecializedTeams import SpecializedTeams, norm, normalize
from time import perf_counter_ns as timer


class OptimizationModel:
  def __init__(self, dim):
    self.objective_function = None
    self.E = None
    self.E_RHS = None
    self.G = None
    self.G_RHS = None
    self.H = None
    self.H_RHS = None
    self.current_results = None
    self.dim = dim
    self.list_of_equality_constraints = []
    self.list_of_inequality_constraints = []
      

  def AddConstraint(self, f, RHS, type_):
##    if not isinstance(RHS, float):
##      raise ValueError(f'A float was expected, but got {RHS}')
    
    try:
      f(np.zeros(self.dim))
    except:
      print(f'f is not a proper input, a callable function accepting arrays of size ({self.dim},) is expected.')
      raise ValueError(f"Given constraint function output f(0)={f(np.zeros(self.dim))} is not appropriate.")

    if type_ == "=" or type_ == 0:
       self.list_of_equality_constraints.append((f, RHS))
      
    elif type_ == "<=" or type_ == 1:
      self.list_of_inequality_constraints.append((f, RHS, True))
    
    elif type_ == ">=" or type_ == 2:
      self.list_of_inequality_constraints.append((f, RHS, False))
    
    else:
      raise ValueError(f'Unexpected type {type_} is given.')
  
  def BoundVariables(self, bound, index): # bound = (a, b) where a and b are non-nan floats including inf, 
    # index is the index of the components of variable x that the bounds applies

    a, b = bound
    if b == a:
      raise ValueError(f"Given bounds define a single point, use equality constraints instead.")

    elif b < a:
      temp_var = b
      b = a
      a = temp_var
    
    elif a < b:
      pass
    
    else:
      raise ValueError(f"Elements of {bound} could not be sorted.")

    if isinstance(index, (int, np.int64, slice)):
      pass
    
    else:
      raise ValueError(f'{index} with type {type(index)} could not be identified.')
    
    if np.isnan(np.dot(bound, bound)):
      raise ValueError(f"Given bound tuple {bound} containts NaNs.")
    
    if np.any(np.isinf(bound)):
      if np.all(np.isinf(bound)):
        raise ValueError(f"{bound} defines no meaningful constraint.")

      elif np.isinf(a):
        self.Addconstraint(lambda x : x[index], b, "<=")

      elif np.isinf(b):
        self.AddConstraint(lambda x : x[index], a, ">=")

      else:
        raise Exception("What the hell moment at bounding with infs.")

    else:
      # a < x < b
      self.AddConstraint(lambda x : x[index], a, ">=")
      self.AddConstraint(lambda x : x[index], b, "<=")
  
  def SetObjective(self, z, objective="min"):
    try:
      z(np.zeros(self.dim))
    except:
      print(f'z is not a proper input, a callable function accepting arrays of size ({self.dim},) is expected.')

    if objective == "min":
      self.objective_function = z
    elif objective == "max":
      self.objective_function = lambda x : -z(x)
    else:
      raise ValueError(f"Unexpected objective type {objective} is given.")
  
  def SetConstraints(self):
    if len(self.list_of_equality_constraints) + len(self.list_of_inequality_constraints) == 0:
      print("No constraints to set are found.")

    else:
      eq_f = []
      less_than_f = []
      greater_than_f = []
      if len(self.list_of_equality_constraints) > 0:        
        eq_RHS = []
        k = len(self.list_of_equality_constraints)
        for eq_const in self.list_of_equality_constraints:
          eq_f.append(eq_const[0])
          eq_RHS.append(eq_const[1])
        
        self.E_RHS = np.hstack(eq_RHS)
        self.E = lambda x : np.hstack([f(x) for f in eq_f])
      
      if len(self.list_of_inequality_constraints) > 0:       
        less_than_RHS = []
        greater_than_RHS  =[]
        d = len(self.list_of_inequality_constraints)
        for ineq_const in self.list_of_inequality_constraints:
          if ineq_const[2]:
            less_than_f.append(ineq_const[0])
            less_than_RHS.append(ineq_const[1])
          
          else:
            greater_than_f.append(ineq_const[0])
            greater_than_RHS.append(ineq_const[1])
        
        if less_than_RHS:
          self.G = lambda x : np.hstack([ltf(x) for ltf in less_than_f])
          self.G_RHS = np.hstack(less_than_RHS)
        
        if greater_than_RHS:
          self.H = lambda x : np.hstack([gtf(x) for gtf in greater_than_f])
          self.H_RHS = np.hstack(greater_than_RHS)


  def StartOptimization(self, search_space_size, team_size=30, team_number=10, social_coefficient=0.95, cognitive_coefficient=0.95,
                        inertia=0.75, use_merge_and_exploit=True, keep_exploitation_short=True):
    start = timer()
    self.ModelOptimizer = SpecializedTeams(self.dim, team_size, team_number, social_coefficient, cognitive_coefficient, inertia, search_space_size, final_inertia=0.4)
    self.ModelOptimizer.feasibility_tolerance = 0.000001
    proc1 = timer()
    print(f'Teams are formed in {1e-6 * (proc1 - start):.5f} ms.')
    proc1 = timer()
    self.ModelOptimizer.initialize_the_teams(self.objective_function, equality_constraints=self.E, equality_RHS=self.E_RHS, less_than_constraints=self.G, less_than_RHS=self.G_RHS, 
                                             greater_than_constraints=self.H, greater_than_RHS=self.H_RHS)
    proc2 = timer()
    print(f'Teams are initialized in {1e-6 * (proc2 - proc1):.5f} ms.')
    proc2 = timer()
    self.ModelOptimizer.optimize_objective(criteria=0.00001, criteria_type="accumulation")
    proc3 = timer()
    print(f'Optimization is finished in {1e-6 * (proc3 - proc2):.5f} ms.')

    self.current_results = {"Best Minimum Value with penalty" : self.ModelOptimizer.global_best_value,
                            "Best Minimum Value position" : self.ModelOptimizer.global_best_position,
                            "Minimum Value of Objective without penalty" : self.ModelOptimizer.true_objective(self.ModelOptimizer.global_best_position),
                            "Is the best feasible" : self.ModelOptimizer.is_feasible(self.ModelOptimizer.global_best_position),
                            "Feasibility Scores of the best position" : self.ModelOptimizer.feasibility_scores(self.ModelOptimizer.global_best_position)}

    if use_merge_and_exploit:
      print(f'\nResults before applying Merge&Exploit: \n')
      for key in self.current_results.keys():
        print(f'{key} : {self.current_results[key]}')
      proc4 = timer()
      print()
      self.ModelOptimizer.chaotic_exploration(150, 0.005)
      self.ModelOptimizer.merge_and_exploit(500, min(0.4, self.ModelOptimizer.teamparams[-1]), keep_it_short=keep_exploitation_short)
      proc5 = timer()
      print(f"\nMerge&Exploit procedure is completed in {1e-6 * (proc5 - proc4):.5f} ms.")
      print(f'\nExtra optimization procedure is completely done. Final, possibly updated, results follow:')

      self.current_results = {"Best Minimum Value with penalty" : self.ModelOptimizer.global_best_value,
                            "Best Minimum Value position" : self.ModelOptimizer.global_best_position,
                            "Minimum Value of Objective without penalty" : self.ModelOptimizer.true_objective(self.ModelOptimizer.global_best_position),
                            "Is the best feasible" : self.ModelOptimizer.is_feasible(self.ModelOptimizer.global_best_position),
                            "Feasibility Scores of the best position" : self.ModelOptimizer.feasibility_scores(self.ModelOptimizer.global_best_position)}

      for key in self.current_results.keys():
        print(f'{key} : {self.current_results[key]}')
      print()

    else:
      print(f'\nOptimization procedure is completely done. Final results follow:\n')
      for key in self.current_results.keys():
        print(f'{key} : {self.current_results[key]}')
      print()
