import hashlib
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

class SimulationStorage:
    """
    Manages simulation output with hybrid naming:
    - Human-readable prefix
    - Short hash suffix for collision prevention
    - Optional CSV index for querying
    """
    
    def __init__(self, base_dir: Path, use_index: bool = True, hash_length: int = 6):
        """
        Args:
            base_dir: Base directory for all simulations
            use_index: Whether to maintain CSV index (useful for querying)
            hash_length: Length of hash suffix (default: 6 chars)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.use_index = use_index
        self.hash_length = hash_length
        
        if use_index:
            self.index_file = base_dir / "simulation_index.csv"
            self._initialize_index()
    
    def _initialize_index(self):
        """Create or load the simulation index."""
        if self.index_file.exists():
            self.index = pd.read_csv(self.index_file)
        else:
            self.index = pd.DataFrame(columns=[
                'directory', 'k_wrap', 'inf_protamine', 
                'tau_max', 'tau_steps', 'k_unbind', 'k_bind',
                'p_conc', 'cooperativity', 'binding_sites', 'replicates'
            ])
    
    def _get_param_hash(self, params: dict) -> str:
        """Generate short hash from parameters."""
        param_repr = {
            'k_wrap': params.get('k_wrap'),
            'inf_protamine': params.get('inf_protamine'),
            'tau_max': params.get('tau_max'),
            'tau_steps': params.get('tau_steps'),
            'prot_params': params.get('prot_params'),
            'binding_sites': params.get('binding_sites'),
        }
        full_hash = hashlib.sha256(
            json.dumps(param_repr, sort_keys=True).encode()
        ).hexdigest()
        return full_hash[:self.hash_length]
    
    def _get_human_readable_name(self, params: dict) -> str:
        """
        Generate human-readable directory name.
        
        Example: 'k1.0_p100.0_c4.5_rep20_inf'
        """
        prot = params.get('prot_params', {})
        parts = [
            f"k{params.get('k_wrap', 1.0):.1f}",
            f"p{prot.get('p_conc', 0.0):.1f}",
            f"c{prot.get('cooperativity', 0.0):.1f}",
        ]
        
        # Add optional flags
        if params.get('inf_protamine', True):
            parts.append("inf")
        
        return "_".join(parts)
    
    def get_directory_name(self, params: dict) -> str:
        """
        Generate hybrid directory name: human-readable + hash.
        
        Example: 'k1.0_p100_c4.5_inf__a3f7e9'
        """
        readable = self._get_human_readable_name(params)
        hash_suffix = self._get_param_hash(params)
        return f"{readable}__{hash_suffix}"
    
    def ensure_directory_structure(self, params: dict) -> Path:
        """Create directory structure for parameters."""
        dirname = self.get_directory_name(params)
        sim_dir = self.base_dir / dirname
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (sim_dir / "trajectories").mkdir(exist_ok=True)
        (sim_dir / "summaries").mkdir(exist_ok=True)
        (sim_dir / "survival").mkdir(exist_ok=True)
        
        # Save parameters
        params_file = sim_dir / "parameters.json"
        if not params_file.exists():
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=2)
        
        # Update index if enabled
        if self.use_index and not self._is_indexed(dirname):
            self._update_index(params, dirname)
        
        return sim_dir
    
    def _is_indexed(self, dirname: str) -> bool:
        """Check if directory is already in index."""
        return dirname in self.index['directory'].values
    
    def _update_index(self, params: dict, dirname: str):
        """Add entry to index."""
        prot = params.get('prot_params', {})
        
        new_entry = {
            'directory': dirname,
            'k_wrap': params.get('k_wrap'),
            'inf_protamine': params.get('inf_protamine'),
            'tau_max': params.get('tau_max'),
            'tau_steps': params.get('tau_steps'),
            'k_unbind': prot.get('k_unbind'),
            'k_bind': prot.get('k_bind'),
            'p_conc': prot.get('p_conc'),
            'cooperativity': prot.get('cooperativity'),
            'binding_sites': params.get('binding_sites'),
            'replicates': params.get('replicates', 20),
        }
        
        self.index = pd.concat([self.index, pd.DataFrame([new_entry])], ignore_index=True)
        self.index.to_csv(self.index_file, index=False)
    
    def get_output_paths(self, params: dict, file_id: str) -> dict:
        """
        Get output file paths for a specific file ID.

        Args:
            params: Simulation parameters
            file_id: File identifier (e.g., '001', 'RET_001')

        Returns:
            Dictionary with 'trajectory', 'summary', 'survival', 'param_dir' paths
        """
        sim_dir = self.ensure_directory_structure(params)
        return {
            'trajectory': sim_dir / "trajectories" / f"{file_id}.parquet",
            'summary':    sim_dir / "summaries"    / f"{file_id}.tsv",
            'survival':   sim_dir / "survival"     / f"{file_id}.parquet",
            'param_dir':  sim_dir,
        }
    
    def find_simulations(self, **query_params) -> pd.DataFrame:
        """
        Find simulations matching parameters.
        
        Example:
            storage.find_simulations(k_wrap=1.0, p_conc=100.0)
        """
        if not self.use_index:
            raise ValueError("Index not enabled. Set use_index=True in __init__")
        
        # Flatten prot_params if provided
        if 'prot_params' in query_params:
            prot = query_params.pop('prot_params')
            query_params.update(prot)
        
        matches = self.index.copy()
        for param, value in query_params.items():
            if param in matches.columns:
                matches = matches[matches[param] == value]
        
        # Add full paths
        matches['full_path'] = matches['directory'].apply(
            lambda d: self.base_dir / d
        )
        
        return matches
    
    def list_all_simulations(self) -> pd.DataFrame:
        """List all indexed simulations."""
        if not self.use_index:
            # Fallback: scan directories
            dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name != "temps"]
            return pd.DataFrame({'directory': [d.name for d in dirs]})
        return self.index.copy()
    
    def rebuild_index(self) -> pd.DataFrame:
        """
        Rebuild the index by scanning all directories in base_dir.
        
        Scans all subdirectories in base_dir, reads their parameters.json files,
        and rebuilds the simulation_index.csv.
        Run the rebuild_index.py script in the src/scripts directory when needed.

        Why rebuild?
        --------------
        Over time, simulations may be added or removed manually, or the index
        file may become corrupted. Rebuilding ensures the index accurately reflects
        the current state of the simulation directories. Especially in the cluster where 
        all the array jobs are writing to the same index file (simulation_index.csv),
        rebuilding the index after all jobs are done ensures consistency. 
        This is reason also I use index=False when writing the data during the simulations.

        Returns:
            DataFrame with rebuilt index
            
        Raises:
            ValueError: If index is not enabled
        """
        if not self.use_index:
            raise ValueError("Index not enabled. Set use_index=True in __init__")
        
        print(f"Scanning directories in: {self.base_dir}")
        
        self.index = pd.DataFrame(columns=self.index.columns)
        # Scan all subdirectories
        scanned = 0
        indexed = 0
        
        for sim_dir in self.base_dir.iterdir():
            if not sim_dir.is_dir():
                continue
            if sim_dir.name in ['temps', '.git', '__pycache__']:  # Skip special directories
                continue
            
            scanned += 1
            params_file = sim_dir / "parameters.json"
            
            if params_file.exists():
                try:
                    with open(params_file, 'r') as f:
                        params = json.load(f)
                    ## Update index
                    self._update_index(params, sim_dir.name)
                    indexed += 1
                except Exception as e:
                    print(f"Warning: Could not read parameters from {sim_dir.name}: {e}")
            else:
                print(f"Warning: No parameters.json found in {sim_dir.name}")

        print(f"\nScanned {scanned} directories, indexed {indexed} simulations")

        # Save index
        self.index.to_csv(self.index_file, index=False)
        print(f"✓ Index saved to: {self.index_file}")
        
        return self.index