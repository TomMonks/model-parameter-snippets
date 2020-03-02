from dataclasses import dataclass, FrozenInstanceError
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

@dataclass(frozen=True)
class ScenarioParameters:
    '''
    Dataclass for SimpleMonteCarloED
    
    There is a fair bit more you can do with
    dataclasses but this is the bare minimum and is 
    still very useful.  
    
    Note the @dataclass decorator.  This takes
    a parameter call frozen which means the class
    is immutable.  That's nice for parameters!
    '''
    name: str 
    #triage and assessment
    mean_process_time: float = 150.0
    #decision to admit
    mean_dta: float = 120.0
    #probability admit
    p_admit: float = 0.4

    


class SimpleMonteCarloED(object):
    '''
    SimpleMonteCarloED.

    A very simple monte carlo ED
    
    For a given no. of patients simulate
    1. Triage assessment and treatment process time
    2. Admit or not admit
    3. For admitted patients only - delay in admission.
    '''
    def __init__(self, params, random_state=None):
        '''
        Constructor for SimpleMonteCarlosED
        
        Parameters:
        -------
        params - ScenarioParmaeters, dataclass for sim model
        
        random_state - int, random seed. Allows for common random
        numbers to be used across multiple versions of the same model and
        reduces the noise in comparisons. (detault=None.)
        '''
        self._params = params
        self._rs = RandomState(random_state)

    def simulate(self, n_patients):
        '''Performa a single replications/run of the simuation model

        Params:
        -------
        n_patients - int, no. of patients to simulate.
        '''
        process_times = self._simulate_ed_process_times(n_patients)
        admissions = self._simulate_admission(n_patients)
        admit_delays = self._simulate_dta_times(n_patients)

        #total time in ED for admitted patients
        ed_times_admit = process_times[admissions == 1] \
            + admit_delays[admissions == 1]
            
        #total time in ED for non-admitted patients    
        ed_times_not_admit = process_times[admissions == 0] 

        #distribution of ed times.
        return np.append(ed_times_admit, ed_times_not_admit)

    def _simulate_ed_process_times(self, n_patients):
        '''
        simulate ed process times for n patients
        '''
        return self._rs.exponential(self._params.mean_process_time, 
                                     size=n_patients)

    def _simulate_admission(self, n_patients):
        '''simulate admission Y/N for n patients
        '''
        return self._rs.binomial(n=1,
                                p=self._params.p_admit, 
                                size=n_patients)

    def _simulate_dta_times(self, n_patients):
        '''simulate admission delays for n patients
        '''
        return self._rs.exponential(self._params.mean_dta, 
                                    size=n_patients)


    
if __name__ == '__main__':
    
    
    #dataclass automatically create __init__ for you.
    #create the different param sets for each scenrio.
    scenarios = []
    scenarios.append(ScenarioParameters('baseline'))
    scenarios.append(ScenarioParameters('flu_pandemic', 160.0, 180.0, 0.7))
    

    #access them just like you would access public attributes of a class
    print(scenarios[0].mean_process_time)
    print(scenarios[1].mean_process_time)
    
    #the frozen parameter in the decorator makes it immutable
    try:
        #try to change mean process time to some arbitracy no.
        scenarios[0].mean_process_time = 10_000
    except FrozenInstanceError:
        print('cannot change an immutable dataclass!')


    #example using them with a v.simple sim model.
    N_PATIENTS = 1000
    SEED = 909
    
    #collect results from single run of each scenario
    results = []
    for scenario in scenarios:
        model = SimpleMonteCarloED(scenario, random_state=909)
        total_times_in_dept = model.simulate(N_PATIENTS)
        results.append(total_times_in_dept)
    
    #plot results
    N_BINS = 20
    fig, axes = plt.subplots(len(results), 1, sharex=True, sharey=True,
                             tight_layout=True)

    for result, scenario, ax in zip(results, scenarios, axes):
        ax.hist(result, bins=N_BINS)
        ax.set(title=scenario.name)

    plt.show()

    

    

    
    
