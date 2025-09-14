import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter_ns as timer


def normalize(vector):
  norm = (vector * vector).sum() ** 0.5
  if norm < 1e-14:
    raise Exception(f"Norm of the vector is too small or zero: {norm} so it is non-normalizable.")
  else:
    return vector / norm


def norm(vector):
  return (vector * vector).sum() ** 0.5


class SpecializedTeams:
  def __init__(self, dim, team_size, team_number, social, cognitive, inertia, search_space_size, constriction=0.975,
               decreasing_inertia=True, final_inertia=0.3, search_space_center=None):
    self.dim = dim
    self.team_size = team_size
    self.team_number = team_number
    self.box_size = search_space_size
    self.teamparams = [social, cognitive, inertia, constriction, decreasing_inertia, final_inertia]
    self.team_particle_positions = {team_index : [float("nan")] * self.team_size  for team_index in range(self.team_number)}
    self.team_particle_velocities = {team_index : [float("nan")] * self.team_size for team_index in range(self.team_number)}
    self.team_particle_best_positions = {team_index : [float("nan")] * self.team_size for team_index in range(self.team_number)}
    self.team_particle_best_values = {team_index : [float("nan")] * self.team_size for team_index in range(self.team_number)}
    self.team_best_positions = [float("nan")] * self.team_number
    self.team_best_values = [float("inf")] * self.team_number
    self.global_best_value = float("inf")
    self.global_best_position = None
    self.team_cover_radius = None
    self.objective = None
    self.true_objective = None
    self.feasibility_scores = None
    self.is_feasible = None
    self.slack_variables = None

    if search_space_center is not None:
      SSC = np.array(search_space_center)
      if SSC.shape != (dim,):
        raise ValueError(f'Given center {SSC} does not match the expected dimension/shape ({dim},) with {SSC.shape}')
      self.search_space_center = SSC
    else:
      self.search_space_center = np.zeros(self.dim)
    
    self.feasibility_tolerance = 0.01
    self.E = None # E(x) = d_k for k-many equality constraints
    self.d_k = None
    self.G = None # G(x) <= b_d for d-many inequality constraints
    self.b_d = None
    self.LTC_count = 0
    self.GTC_count = 0
    self.EC_count = 0

  
  def initialize_the_teams(self, objective, equality_constraints=None, equality_RHS=None, less_than_constraints=None, less_than_RHS=None, greater_than_constraints=None, greater_than_RHS=None):
    generator = np.random.default_rng()

    if ((equality_constraints is None) ^ (equality_RHS is None)) or ((less_than_constraints is None)  ^ (less_than_RHS is None)) or ((greater_than_constraints is None) ^ (greater_than_RHS is None)):
      raise ValueError("One or more of the constraints' information is not given.")

    self.true_objective = objective
    if equality_constraints is not None:
      self.EC_count = len(equality_RHS)

      self.E = equality_constraints
      self.d_k = equality_RHS
   
    if (less_than_constraints is not None) and (greater_than_constraints is not None):
      self.LTC_count = len(less_than_RHS)
      self.GTC_count = len(greater_than_RHS)

      self.G = lambda x : np.hstack((less_than_constraints(x), -greater_than_constraints(x),))
      self.b_d = np.hstack((less_than_RHS, -greater_than_RHS,))
    
    elif less_than_constraints is not None:
      self.LTC_count = len(less_than_RHS)

      self.G = less_than_constraints
      self.b_d = less_than_RHS
    
    elif greater_than_constraints is not None:
      self.GTC_count = len(greater_than_RHS)

      self.G = lambda x : -greater_than_constraints(x)
      self.b_d = -greater_than_RHS
    
    if self.E is not None and self.G is not None:
      def z(x):
        S_i = self.b_d - self.G(x)
        S_e = self.d_k - self.E(x)
        return self.true_objective(x) - 99999.9 * min(S_i.min(), 0.0) + 99999.9 * np.abs(S_e).sum()
      
      def Feasibility_Score(x):
        S_i = self.b_d - self.G(x)
        S_e = self.d_k - self.E(x)
        InEqScore = 1.0 + min(S_i.min(), 0.0)
        EqScore = 1.0 - np.abs(S_e).max()
        return (InEqScore, EqScore,)
      
      self.feasibility_scores = Feasibility_Score

      def IsFeasible(x):
        score = self.feasibility_scores(x)
        is_feasible = True if score[0] >= 1.0 - self.feasibility_tolerance and score[1] >= 1.0 - self.feasibility_tolerance else False
        return is_feasible

      self.is_feasible = IsFeasible      
      self.objective = z
    
    elif self.E is not None:
      def z(x):
        S_e = self.d_k - self.E(x)
        return self.true_objective(x) + 999.9 * np.abs(S_e).sum()
      
      def Feasibility_Score(x):
        S_e = self.d_k - self.E(x)
        InEqScore = 1.0
        EqScore = 1.0 - np.abs(S_e).max()
        return (1.0, EqScore,)
      
      self.feasibility_scores = Feasibility_Score
      
      def IsFeasible(x):
        score = self.feasibility_scores(x)
        is_feasible = True if score[1] >= 1.0 - self.feasibility_tolerance else False
        return is_feasible

      self.is_feasible = IsFeasible  
      self.objective = z

    elif self.G is not None:
      def z(x):
        S_i = self.b_d - self.G(x)
        return self.true_objective(x) - 999.9 * min(S_i.min(), 0.0)
      
      def Feasibility_Score(x):
        S_i = self.b_d - self.G(x)
        InEqScore = 1.0 + min(S_i.min(), 0.0)
        return (InEqScore, 1.0,)
      
      self.feasibility_scores = Feasibility_Score

      def IsFeasible(x):
        score = self.feasibility_scores(x)
        is_feasible = True if score[0] >= 1.0 - self.feasibility_tolerance else False
        return is_feasible

      self.is_feasible = IsFeasible  
      self.objective = z

    else:
      self.feasibility_scores = lambda x : (1.0, 1.0,)
      self.is_feasible = lambda x : True
      self.objective = self.true_objective

    team_centers = []
    center_coverage_radius = self.box_size * 0.5 * ((1.0 / (self.team_number)) ** (1./self.dim))
    inner_hypercube_size = 1.1 * (self.box_size - center_coverage_radius / np.sqrt(self.dim))

    while len(team_centers) < self.team_number:
      candidate_center = inner_hypercube_size * (generator.uniform(size=self.dim) - 0.5)
      AddThis = True
      for other_center in team_centers:
        d = candidate_center - other_center
        if (d * d).sum() < 1.5 *  center_coverage_radius:
          AddThis = False
        else:
          pass
      if AddThis:
        team_centers.append(self.search_space_center + candidate_center)
      else:
        continue
    self.team_cover_radius = center_coverage_radius
    for team_index in range(self.team_number):
      team_center = team_centers[team_index]
      team_particle_positions = [team_center + 0.5 * center_coverage_radius * generator.normal(size=self.dim) for _ in range(self.team_size)]
      team_particle_velocities = [generator.uniform() * center_coverage_radius * normalize(generator.uniform(size=self.dim)) for _ in range(self.team_size)]
      team_particle_best_values = [self.objective(x) for x in team_particle_positions]
      for particle_index in range(self.team_size):        
        if team_particle_best_values[particle_index] < self.team_best_values[team_index]:
          self.team_best_positions[team_index] = team_particle_positions[particle_index]
          self.team_best_values[team_index] = team_particle_best_values[particle_index]
        else:
          continue
      self.team_particle_positions[team_index] = team_particle_positions
      self.team_particle_velocities[team_index] = team_particle_velocities
      self.team_particle_best_values[team_index] = team_particle_best_values
      self.team_particle_best_positions[team_index] = team_particle_positions
      if self.team_best_values[team_index] < self.global_best_value:
        self.global_best_positions = self.team_best_positions[team_index]
        self.global_best_values = self.team_best_values[team_index]
 
  
  def chaotic_exploration(self, iteration_number, chaotic_std_proportion):
    generator = np.random.default_rng()
    print("\nInitializing the chaotic exploration session...")
    p_best_updated_number = 0
    t_best_updated_number = 0
    g_best_updated_number = 0
    for _ in range(iteration_number):
      if _ % int(0.2 * iteration_number) == 0:
        print(f"Iteration number {_} has started.")

      for team_index in range(self.team_number):
        tm_p_pos = self.team_particle_positions[team_index]
        tm_p_vel = self.team_particle_velocities[team_index]
        tm_p_best_vals = self.team_particle_best_values[team_index]
        tm_best_val = self.team_best_values[team_index]
        for particle_index in range(self.team_size):
          cur_pos = tm_p_pos[particle_index]
          cur_vel = tm_p_vel[particle_index]
          cur_best_val = tm_p_best_vals[particle_index]
          next_vel = cur_vel + generator.normal(size=self.dim) * chaotic_std_proportion * self.team_cover_radius
          next_pos = cur_pos + next_vel
          next_val = self.objective(next_pos)
          tm_p_pos[particle_index] = next_pos
          tm_p_vel[particle_index] = next_vel
          if next_val <= cur_best_val:
            p_best_updated_number += 1
            tm_p_best_vals[particle_index] = next_val
            self.team_particle_best_positions[team_index][particle_index] = next_pos
            if next_val <= tm_best_val:
              t_best_updated_number += 1
              self.team_best_values[team_index] = next_val
              self.team_best_positions[team_index] = next_pos
              if next_val < self.global_best_value:
                g_best_updated_number += 1
                self.global_best_position = next_pos
                self.global_best_value = next_val
    print(f'During the entire chaotic exporation session:\nParticle, team, and global best position were updated {p_best_updated_number, t_best_updated_number} and {g_best_updated_number} many times, respectively.\n')
  
  def how_many_feasible(self, particles):
    return sum([self.is_feasible(x) for x in particles])


  def feasible_region_search(self, min_ratio_converging_teams=0.4, MaxIter=1000):
    # min_ratio_converging_teams: For a successful search stopping criteria, the ratio of converging teams to all
    # a team is converging when its ratio of complete particles is > completion_ratio_per_teams=0.4 
    # completion_ratio_per_team : a particle in a team is said to be complete when it is a feasible point 
    # hence this ratio represents the ratio of the complete particles to entire team for each team
    if self.E is None and self.G is None:
      print("Entire space is feasible as there are no given constraints. Use other procedures instead.")
      return None
    
    generator = np.random.default_rng()
    print("\nInitializing the feasible region exploration session...")

    generate_feasibility_value = lambda x : 2.0 - sum(self.feasibility_scores(x))

    soc, cog, w_i, constriction, dec_w, w_f = self.teamparams
    inertia_diff = w_i - w_f
    inertia_inflection_point = 0.05 * MaxIter
    scaler = (1.0 + np.exp(-0.5 * inertia_inflection_point)) * inertia_diff

    progression_level = int(0.2 * MaxIter)

    non_converging_teams = list(range(self.team_number))

    ratio_converging_teams = np.isnan(non_converging_teams).sum() / self.team_number

    completion_ratio_per_team = 0.4

    iter_index = 0
    subtract_iter = 3
    while iter_index < MaxIter and ratio_converging_teams <= min_ratio_converging_teams:
      if iter_index % progression_level == 0:
        print(f"Iteration is at {iter_index} with converging teams ratio {ratio_converging_teams:.2f} and global min {self.global_best_value}")
      iter_index += 1
      w = w_f + scaler * (1.0 / (1.0 + np.exp(0.5 * (iter_index - inertia_inflection_point)))) if dec_w else w_i
      
      for team_index in range(self.team_number):
        if non_converging_teams[team_index] == team_index:
          w = w_i
        tm_p_pos = self.team_particle_positions[team_index]
        tm_p_vel = self.team_particle_velocities[team_index]
        tm_p_best_vals = self.team_particle_best_values[team_index]
        tm_p_best_pos = self.team_particle_best_positions[team_index]
        tm_best_val = self.team_best_values[team_index]
        tm_best_pos = self.team_best_positions[team_index]
        for particle_index in range(self.team_size):
          cur_pos = tm_p_pos[particle_index]
          cur_vel = tm_p_vel[particle_index]
          cur_best_val = tm_p_best_vals[particle_index]
          cur_best_pos = tm_p_best_pos[particle_index]
          r1, r2 = generator.uniform(size=2)
          next_vel = cur_vel * w + r1 * soc * (tm_best_pos - cur_pos) + r2 * cog * (cur_best_pos - cur_pos)
          next_pos = cur_pos + next_vel * constriction
          next_val = generate_feasibility_value(next_pos)
          tm_p_pos[particle_index] = next_pos
          tm_p_vel[particle_index] = next_vel
          if next_val <= cur_best_val:
            tm_p_best_vals[particle_index] = next_val
            self.team_particle_best_positions[team_index][particle_index] = next_pos
            if next_val <= tm_best_val:
              self.team_best_values[team_index] = next_val
              self.team_best_positions[team_index] = next_pos
              if next_val < self.global_best_value:
                self.global_best_position = next_pos
                self.global_best_value = next_val
        
        if non_converging_teams[team_index] != float("nan"):
          completion_count = self.how_many_feasible(tm_p_pos)
          if completion_count > int(self.team_size * completion_ratio_per_team):
            non_converging_teams[team_index] = float("nan")
      
      ratio_converging_teams = np.isnan(non_converging_teams).sum() / self.team_number

    if ratio_converging_teams < min_ratio_converging_teams:
      print(f'\n Non converging teams : {non_converging_teams}')
      print(f'Feasible Region Search failed with min complete ratio {ratio_converging_teams:.3f} at iteration {iter_index}.')
      return False
    
    else:
      print(f'\n Non coınverging teams : {non_converging_teams}')
      print(f'Feasible Region Search successed and reached the min completion ratio {completion_ratio_per_team:.3f} with actual min {ratio_converging_teams:.3f} at iteration {iter_index}.')
      return True
  
  def merge_and_exploit(self, step_number, inertia, keep_it_short):
    soc, cog, w, constriction, decreasing_inertia, final_inertia = self.teamparams
    w = inertia
    generator = np.random.default_rng()
    print(f"Starting the Merge&Exploit procedure for {step_number} many iterations.")
    progression_level = int(step_number * 0.1)
    starting_best = self.true_objective(self.global_best_position)
    print(f"Current best true minimum found {starting_best:.5f} with feasibility score {self.feasibility_scores(self.global_best_position)}.")
    iter_index = 0
    stagnating_levels = 0
    DidItChange = False
    while iter_index < step_number:
      if iter_index % progression_level == 0:
        print(f"Iteration is at {iter_index} with current minimum {self.global_best_value}")
        if not keep_it_short:
          pass
        else:
          if DidItChange:
            DidItChange = False
          else:
            stagnating_levels += 1
          
          if stagnating_levels > 2:
            print(f'\nMerge&Exploit procedure is cut at iteration {iter_index} with best minimum found {self.global_best_value} as keep_it_short={keep_it_short}\n')
            break            

      iter_index += 1
      for team_index in range(self.team_number):
        tm_p_pos = self.team_particle_positions[team_index]
        tm_p_vel = self.team_particle_velocities[team_index]
        tm_p_best_vals = self.team_particle_best_values[team_index]
        tm_p_best_pos = self.team_particle_best_positions[team_index]
        for particle_index in range(self.team_size):
          cur_pos = tm_p_pos[particle_index]
          cur_vel = tm_p_vel[particle_index]
          cur_best_val = tm_p_best_vals[particle_index]
          cur_best_pos = tm_p_best_pos[particle_index]
          r1, r2 = generator.uniform(size=2)
          next_vel = cur_vel * w + r1 * soc * (self.global_best_position - cur_pos) + r2 * cog * (cur_best_pos - cur_pos)
          next_pos = cur_pos + next_vel * constriction
          next_val = self.objective(next_pos)
          tm_p_pos[particle_index] = next_pos
          tm_p_vel[particle_index] = next_vel
          if next_val <= cur_best_val and self.is_feasible(next_pos):
            tm_p_best_vals[particle_index] = next_val
            self.team_particle_best_positions[team_index][particle_index] = next_pos

            if next_val < self.global_best_value:
              value_change = self.global_best_value - next_val
              DidItChange = True if value_change > 1e-10 else DidItChange
              self.global_best_position = next_pos
              self.global_best_value = next_val
    
    current_true_global_best = self.true_objective(self.global_best_position)
    if current_true_global_best < starting_best:
      print(f"Best true minimum is improved to {current_true_global_best:.5f} with feasibility score {self.feasibility_scores(self.global_best_position)}.")
    else:
      print(f"This Merge&Exploit procedure did not improve the true minimum.")


  def get_slack_variables(self):
    if self.global_best_position is None:
      print(f"The model is not set yet, please set the model and run it to get the parameters.")
      return ([], [],)
    
    if (self.E is None) and (self.G is None):
      print("No constraints are found for the given model.")
      return ([], [],)

    a = []
    b = []
    x = self.global_best_position
    if self.G is not None:
      S_i = self.b_d - self.G(x)
      a = [float(_) for _ in S_i]
    
    if self.E is not None:
      S_e = self.d_k - self.E(x)
      b = [float(_) for _ in S_e]
    
    print(f'''Less-Than Constraints Number: {self.LTC_count}
Greater-Than Constraints Number: {self.GTC_count}
Equality Constraints Number: {self.EC_count}
Slack variables for Less-Than Constraints: {a[:self.LTC_count]}
Slack variables for Greater-Than Constraints: {a[self.LTC_count:]}
Slack variables for Equality Constraints: {b}''')
    
    return (a, b,)


    
  def optimize_objective(self, criteria=0.01, criteria_type="accumulation", optimization_max_iter=10000, complete_feasible_ratio=0.5,
                         FR_search_max_iter=1000, chaotic_sessions=3, chaotic_coefficient=0.1):
    if self.objective is None:  raise Exception("The swarm is not initialized properly.")
    int_criteria_type = None
    if criteria_type == "accumulation": int_criteria_type = 0
    elif criteria_type == "target":  int_criteria_type = 1
    else: raise ValueError(f"Given criteria type -{criteria_type}- is undefined.")

    progression_level = int(0.05 * optimization_max_iter)
    IsToStop = False

    soc, cog, w, constriction, decreasing_inertia, final_inertia = self.teamparams
    inertia_diff = w - final_inertia
    inertia_inflection_point = 0.05 * optimization_max_iter
    scaler = (1.0 + np.exp(-0.5 * inertia_inflection_point)) * inertia_diff

    FSSearch_iter = 0
    Opt_iter = 0

    if self.global_best_position is None:
      self.global_best_position = np.zeros(self.dim)
    
    result_FSearch = self.feasible_region_search(min_ratio_converging_teams=complete_feasible_ratio, MaxIter=FR_search_max_iter)

    if result_FSearch is None:
      print("Proceeding with the procedure without needing Feasible Region Search.")
    elif result_FSearch is True:
      print("Proceeding with the procedure with the successful completion of Feasible Region Search.")
    else:
      print("Proceeding to the optimization without enough feasibility of the particles.")
    
    generator = np.random.default_rng()

    soc, cog, w_i, constriction, dec_w, w_f = self.teamparams
    inertia_diff = w_i - w_f
    inertia_inflection_point = 0.05 * optimization_max_iter
    scaler = (1.0 + np.exp(-0.5 * inertia_inflection_point)) * inertia_diff

    progression_level = int(0.05 * optimization_max_iter)

    self.global_best_value = float("inf")
    self.team_particle_best_values = {team_index : [float("inf")] * self.team_size for team_index in range(self.team_number)}
    self.team_best_values = [float("inf")] * self.team_number

    IterationCompleteTeams = []
    max_speed_by_team = []
    current_max_speed = -1.0
    iter_index = 0
    while iter_index < optimization_max_iter:

      if iter_index % progression_level == 0:
        print(f"Iteration is at {iter_index} with current minimum {self.global_best_value}")

      iter_index += 1
      w = w_f + scaler * (1.0 / (1.0 + np.exp(0.5 * (iter_index - inertia_inflection_point)))) if dec_w else w_i

      for team_index in range(self.team_number):
        tm_p_pos = self.team_particle_positions[team_index]
        tm_p_vel = self.team_particle_velocities[team_index]
        tm_p_best_vals = self.team_particle_best_values[team_index]
        tm_p_best_pos = self.team_particle_best_positions[team_index]
        tm_best_val = self.team_best_values[team_index]
        tm_best_pos = self.team_best_positions[team_index]
        for particle_index in range(self.team_size):
          cur_pos = tm_p_pos[particle_index]
          cur_vel = tm_p_vel[particle_index]
          cur_best_val = tm_p_best_vals[particle_index]
          cur_best_pos = tm_p_best_pos[particle_index]
          r1, r2 = generator.uniform(size=2)
          next_vel = cur_vel * w + r1 * soc * (tm_best_pos - cur_pos) + r2 * cog * (cur_best_pos - cur_pos)
          next_pos = cur_pos + next_vel * constriction
          next_val = self.objective(next_pos)
          tm_p_pos[particle_index] = next_pos
          tm_p_vel[particle_index] = next_vel
          if next_val <= cur_best_val:
            tm_p_best_vals[particle_index] = next_val
            self.team_particle_best_positions[team_index][particle_index] = next_pos
            if next_val <= tm_best_val:
              self.team_best_values[team_index] = next_val
              self.team_best_positions[team_index] = next_pos
              if next_val < self.global_best_value:
                self.global_best_position = next_pos
                self.global_best_value = next_val
      
      current_max_speed = -1.0
      for team_index in range(self.team_number):
        for particle_index in range(self.team_size):
          this_speed = norm(self.team_particle_velocities[team_index][particle_index])
          if this_speed > current_max_speed:
            current_max_speed = this_speed
          else:
            pass
      
      if not int_criteria_type:        
        if current_max_speed < criteria:
          if chaotic_sessions > 0:
            chaotic_sessions -= 1
            self.chaotic_exploration(int(0.01) * optimization_max_iter, chaotic_coefficient)
          else:
            IsToStop = True
        else:
          continue

      elif int_criteria_type == 1:
        if self.global_best_value < criteria:
          if chaotic_sessions > 0:
            chaotic_sessions -= 1
            self.chaotic_exploration(int(0.01) * optimization_max_iter, chaotic_coefficient)
          else:
            IsToStop = True
        else:
          continue
      else:
        print(f"This criteria type (in integer) {int_criteria_type} is not recognized.")

      if IsToStop:
        if not int_criteria_type:
          print(f"The optimization is finished with best minimum {self.global_best_value:.4f} at iteration {iter_index} with max speed {current_max_speed:.4f} < {criteria:.4f}")
        elif int_criteria_type == 1:
          print(f"The optimization is finished with best minimum {self.global_best_value:.4f} at iteration {iter_index} with <target={criteria:.4f}")
        break
