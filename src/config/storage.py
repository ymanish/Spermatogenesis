import os
import json
import numpy as np
import hashlib
import pandas as pd
from pathlib import Path
import shutil

class SimulationStorage:
    def __init__(self, base_dir: Path, hash_length=8):
        self.base_dir = base_dir
        self.hash_length = hash_length
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = base_dir / "simulation_index.csv"
        self._initialize_index()
    
    def _initialize_index(self):
        """Create or load the simulation index"""
        if self.index_file.exists():
            self.index = pd.read_csv(self.index_file)
        else:
            self.index = pd.DataFrame(columns=[
                'param_hash', 'k_wrap', 'inf_protamine', 
                't_max', 't_steps', 'k_unbind', 'k_bind',
                'p_conc', 'cooperativity', 'binding_sites'
            ])
    
    def _update_index(self, params, param_hash):
        """Add new entry to index"""
        index_params = params.copy()
        
        new_entry = {
            'param_hash': param_hash,
            'k_wrap': index_params.get('k_wrap'),
            'inf_protamine': index_params.get('inf_protamine'),
            't_max': index_params.get('t_max'),
            't_steps': index_params.get('t_steps'),
            'k_unbind': index_params['prot_params'].get('k_unbind'),
            'k_bind': index_params['prot_params'].get('k_bind'),
            'p_conc': index_params['prot_params'].get('p_conc'),
            'cooperativity': index_params['prot_params'].get('cooperativity'),
            'binding_sites': index_params.get('binding_sites'),
        }
        
        new_df = pd.DataFrame([new_entry])
        self.index = pd.concat([self.index, new_df], ignore_index=True)
        self.index.to_csv(self.index_file, index=False)
    
    def _parameter_signature(self, params):
        # Create a stable representation of parameters
        param_repr = {
            'k_wrap': params.get('k_wrap'),
            'inf_protamine': params.get('inf_protamine'),
            't_max': params.get('t_max'),
            't_steps': params.get('t_steps'),
            'prot_params': params.get('prot_params'),
            'binding_sites': params.get('binding_sites'),
        }
        return hashlib.sha256(json.dumps(param_repr, sort_keys=True).encode()).hexdigest()[:self.hash_length]
    
    
    def get_simulation_directory(self, params) -> Path:
        """Get directory path for a parameter set"""
        param_hash = self._parameter_signature(params)
        return self.base_dir / param_hash
    
    def ensure_directory_structure(self, params):
        """Create directory structure for a parameter set"""
        sim_dir = self.get_simulation_directory(params)
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (sim_dir / "trajectories").mkdir(exist_ok=True)
        (sim_dir / "summaries").mkdir(exist_ok=True)
        
        # Save parameters if not already saved
        params_file = sim_dir / "parameters.json"
        if not params_file.exists():
            # Create a copy to avoid modifying original
            params_to_save = params.copy()

            # Convert numpy array to list for JSON serialization
            if 't_max' in params_to_save and 't_steps' in params_to_save:
                t_stop = params_to_save['t_max']      
                t_num = params_to_save.get('t_steps')
                
                params_to_save['t_points'] = np.linspace(0, t_stop, t_num).tolist()
            else:
                raise ValueError("t_max and t_steps must be provided in params")

            with open(params_file, 'w') as f:
                json.dump(params_to_save, f, indent=2)
            
            # Update index
            self._update_index(params, sim_dir.name)
        
        return sim_dir

    def get_output_paths(self, params, file_id: str) -> dict:
        """
        Get output file paths for a specific input file ID
        :param file_id: The identifier from the input file (e.g., "001")
        """
        sim_dir = self.ensure_directory_structure(params)
        return {
            'trajectory': sim_dir / "trajectories" / f"{file_id}.parquet",
            'summary': sim_dir / "summaries" / f"{file_id}.tsv"
        }

    # Query methods
    def find_simulations(self, **query_params):
        """
        Find simulations matching given parameters
        Example: storage.find_simulations(k_wrap=0.25, inf_protamine=False)
        """
        if 'prot_params' in query_params:
            prot = query_params.pop('prot_params')
            for key, value in prot.items():
                query_params[key] = value

        matches = self.index.copy()
        for param, value in query_params.items():
            if param in matches.columns:
                matches = matches[matches[param] == value]
        
        # Add full paths to results
        matches['directory'] = matches['param_hash'].apply(
            lambda h: self.base_dir / h
        )
        return matches
    
    def load_simulation(self, param_hash, file_id: str):
        """
        Load simulation data for a specific parameter hash and file ID
        Returns dictionary with paths
        """
        sim_dir = self.base_dir / param_hash
        if not sim_dir.exists():
            raise FileNotFoundError(f"No simulation found for hash: {param_hash}")
        
        return {
            'trajectory': sim_dir / "trajectories" / f"{file_id}.parquet",
            'summary': sim_dir / "summaries" / f"{file_id}.tsv",
            'parameters': sim_dir / "parameters.json"
        }
    
    def load_by_params(self, file_id: str, **params):
        """
        Find and load simulation by exact parameters and file ID
        Example: storage.load_by_params(file_id="001", k_wrap=0.25, inf_protamine=False)
        """
        # Find matching simulations
        matches = self.find_simulations(**params)
        
        if len(matches) == 0:
            raise ValueError("No matching simulations found")
        if len(matches) > 1:
            print(f"Warning: {len(matches)} matches found. Loading first")
        
        param_hash = matches.iloc[0]['param_hash']
        return self.load_simulation(param_hash, file_id)
    

if __name__ == "__main__":
    from src.config.path import RESULTS_DIR
    storage = SimulationStorage(base_dir=RESULTS_DIR)
    print(f"Simulation storage initialized at: {storage.base_dir}")

    params = {
        'k_wrap': 22.0,
        'prot_params': {
            'k_unbind': 0.23,
            'k_bind': 2113,
            'p_conc': 0.1,
            'cooperativity': 0.0
        },
        'binding_sites': 14,
        't_max': 10.0,
        't_steps': 100,
        'inf_protamine': True
    }
    
    # Get output paths for this file ID
    output_paths = storage.get_output_paths(params, file_id="001")
    traj_outfile = output_paths['trajectory']
    tsv_outfile = output_paths['summary']

    print(f"Trajectory output path: {traj_outfile}")
    print(f"TSV output path: {tsv_outfile}")

    storage = SimulationStorage(RESULTS_DIR)

    # Find parameter set
    matches = storage.find_simulations(**params)

    if not matches.empty:
        param_hash = matches.iloc[0]['param_hash']
        
        # Load results for specific file
        results = storage.load_simulation(param_hash, "001")
        
        # Access files
        traj_path = results['trajectory']
        summary_path = results['summary']
        print(f"Loaded trajectory path: {traj_path}")
        print(f"Loaded summary path: {summary_path}")

        # Load parameters
        with open(results['parameters']) as f:
            params = json.load(f)
        print(f"Loaded parameters: {params}")