"""
Storage Module for Markov Execution
====================================

Manages Markov solver output with hybrid naming:
- Human-readable prefix
- Short hash suffix for collision prevention
- Optional CSV index for querying

Author: MY
Date: 2025-12-11
"""

import hashlib
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


class MarkovStorage:
    """
    Manages Markov solver output with hybrid naming scheme.
    
    Directory structure:
        base_dir/
            k1.0_p10.0_c0.0__a3f7e9/
                parameters.json
                survivals/
                    001.parquet
                    002.parquet
                summaries/
                    001.tsv
                    002.tsv
            markov_index.csv
    
    Example:
        >>> storage = MarkovStorage(base_dir=Path("output/markov"))
        >>> 
        >>> params = {
        ...     'k_wrap': 1.0,
        ...     'prot_params': {'p_conc': 10.0, 'cooperativity': 0.0, ...},
        ...     'tau_max': 1000.0,
        ...     'tau_steps': 500,
        ...     'method': 'expm'
        ... }
        >>> 
        >>> paths = storage.get_output_paths(params, file_id='001')
        >>> # paths['summary'] = .../summaries/001.tsv
        >>> # paths['survivals'] = .../survivals/001.parquet
    """
    
    def __init__(self, base_dir: Path, use_index: bool = True, hash_length: int = 6):
        """
        Initialize Markov storage manager.
        
        Args:
            base_dir: Base directory for all Markov results
            use_index: Whether to maintain CSV index (useful for querying)
            hash_length: Length of hash suffix (default: 6 chars)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.use_index = use_index
        self.hash_length = hash_length
        
        if use_index:
            self.index_file = base_dir / "markov_index.csv"
            self._initialize_index()
    
    def _initialize_index(self):
        """Create or load the Markov index."""
        if self.index_file.exists():
            self.index = pd.read_csv(self.index_file)
        else:
            self.index = pd.DataFrame(columns=[
                'directory', 'k_wrap', 'tau_max', 'tau_steps', 'method',
                'k_bind', 'k_unbind', 'p_conc', 'cooperativity',
                'binding_sites', 'sparse', 'dimensionless', 'eads_delta',
                'eads_weight_mode', 'eads_apply', 'max_nucs', 'max_nucs_seed',
                'tnp2_enabled', 'tnp2_eps_cpg', 'tnp2_mu_t0',
            ])

    @staticmethod
    def _tnp2_storage_repr(params: dict) -> dict:
        """Produce a canonical TNP2 sub-dict for hashing and indexing.

        When the layer is disabled, all TNP2 specifics collapse to a single
        ``{'tnp2_enabled': False}`` representation so different sweep points
        with the layer off dedupe to one storage directory.
        """
        if not bool(params.get('tnp2_enabled', False)):
            return {'tnp2_enabled': False}
        return {
            'tnp2_enabled': True,
            'tnp2_eps_cpg': float(params.get('tnp2_eps_cpg', 1.0)),
            'tnp2_mu_t0': float(params.get('tnp2_mu_t0', -8.0)),
        }
    
    def _get_param_hash(self, params: dict) -> str:
        """
        Generate short hash from parameters.
        
        Args:
            params: Parameter dictionary
        
        Returns:
            Short hash string (e.g., 'a3f7e9')
        """
        param_repr = {
            'k_wrap': params.get('k_wrap'),
            'tau_max': params.get('tau_max'),
            'tau_steps': params.get('tau_steps'),
            'method': params.get('method'),
            'prot_params': params.get('prot_params'),
            'binding_sites': params.get('binding_sites'),
            'sparse': params.get('sparse'),
            'dimensionless': params.get('dimensionless'),
            'eads_delta': params.get('eads_delta', 0.0),
            'eads_weight_mode': params.get('eads_weight_mode', 'none'),
            'eads_apply': params.get('eads_apply', False),
            'max_nucs': params.get('max_nucs'),
            'max_nucs_seed': params.get('max_nucs_seed', 0),
            'tnp2': self._tnp2_storage_repr(params),
        }
        full_hash = hashlib.sha256(
            json.dumps(param_repr, sort_keys=True).encode()
        ).hexdigest()
        return full_hash[:self.hash_length]
    
    def _get_human_readable_name(self, params: dict) -> str:
        """
        Generate human-readable directory name.
        
        Args:
            params: Parameter dictionary
        
        Returns:
            Human-readable name (e.g., 'k1.0_p10.0_c0.0_t1000_expm')
        
        Example names:
            - 'k1.0_p0.0_c0.0_tau1000_expm' (no protamine)
            - 'k1.0_p10.0_c0.0_tau1000_expm' (with protamine)
            - 'k0.1_p100.0_c2.5_tau5000_ode' (ODE solver)
        """
        prot = params.get('prot_params', {})
        parts = [
            f"k{params.get('k_wrap', 1.0):.1f}",
            f"p{prot.get('p_conc', 0.0):.1f}",
            f"c{prot.get('cooperativity', 0.0):.1f}",
            f"eads_{params.get('eads_weight_mode', 'none')}_d{params.get('eads_delta', 0.0):.2f}",
            f"tau{int(params.get('tau_max', 1000))}",
            params.get('method', 'expm')
        ]
        if params.get('max_nucs') is not None:
            parts.extend([
                f"n{int(params.get('max_nucs'))}",
                f"s{int(params.get('max_nucs_seed', 0))}",
            ])

        tnp2 = self._tnp2_storage_repr(params)
        if tnp2.get('tnp2_enabled'):
            parts.append(
                f"tnp2_e{tnp2['tnp2_eps_cpg']:.2f}"
                f"_mu{tnp2['tnp2_mu_t0']:.2f}"
            )

        return "_".join(parts)
    
    def get_directory_name(self, params: dict) -> str:
        """
        Generate hybrid directory name: human-readable + hash.
        
        Args:
            params: Parameter dictionary
        
        Returns:
            Directory name (e.g., 'k1.0_p10.0_c0.0_t1000_expm__a3f7e9')
        """
        readable = self._get_human_readable_name(params)
        hash_suffix = self._get_param_hash(params)
        return f"{readable}__{hash_suffix}"
    
    def ensure_directory_structure(self, params: dict) -> Path:
        """
        Create directory structure for parameters.
        
        Args:
            params: Parameter dictionary
        
        Returns:
            Path to simulation directory
        """
        dirname = self.get_directory_name(params)
        result_dir = self.base_dir / dirname
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (result_dir / "survivals").mkdir(exist_ok=True)
        (result_dir / "summaries").mkdir(exist_ok=True)
        
        # Save parameters
        params_file = result_dir / "parameters.json"
        if not params_file.exists():
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=2)
        
        # Update index if enabled
        if self.use_index and not self._is_indexed(dirname):
            self._update_index(params, dirname)
        
        return result_dir
    
    def _is_indexed(self, dirname: str) -> bool:
        """Check if directory is already in index."""
        return dirname in self.index['directory'].values
    
    def _update_index(self, params: dict, dirname: str):
        """
        Add entry to index.
        
        Args:
            params: Parameter dictionary
            dirname: Directory name
        """
        prot = params.get('prot_params', {})
        
        tnp2 = self._tnp2_storage_repr(params)
        new_entry = {
            'directory': dirname,
            'k_wrap': params.get('k_wrap'),
            'tau_max': params.get('tau_max'),
            'tau_steps': params.get('tau_steps'),
            'method': params.get('method'),
            'k_bind': prot.get('k_bind'),
            'k_unbind': prot.get('k_unbind'),
            'p_conc': prot.get('p_conc'),
            'cooperativity': prot.get('cooperativity'),
            'binding_sites': params.get('binding_sites'),
            'sparse': params.get('sparse', False),
            'dimensionless': params.get('dimensionless', True),
            'eads_delta': params.get('eads_delta', 0.0),
            'eads_weight_mode': params.get('eads_weight_mode', 'none'),
            'eads_apply': params.get('eads_apply', False),
            'max_nucs': params.get('max_nucs'),
            'max_nucs_seed': params.get('max_nucs_seed', 0),
            'tnp2_enabled': tnp2.get('tnp2_enabled', False),
            'tnp2_eps_cpg': tnp2.get('tnp2_eps_cpg'),
            'tnp2_mu_t0': tnp2.get('tnp2_mu_t0'),
        }
        
        self.index = pd.concat([self.index, pd.DataFrame([new_entry])], ignore_index=True)
        self.index.to_csv(self.index_file, index=False)
    
    def get_output_paths(self, params: dict, file_id: str) -> dict:
        """
        Get output file paths for a specific file ID.
        
        Args:
            params: Markov solver parameters
            file_id: File identifier (e.g., '001', 'bound_001')
        
        Returns:
            Dictionary with paths:
                - 'summary': Path to summary TSV
                - 'results': Path to detailed results Parquet
                - 'param_dir': Path to parameter directory
                - 'config': Path to configuration file
        
        Example:
            >>> paths = storage.get_output_paths(params, '001')
            >>> # paths['summary'] = .../summaries/001.tsv
            >>> # paths['survivals'] = .../survivals/001.parquet
        """
        result_dir = self.ensure_directory_structure(params)
        return {
            'summary': result_dir / "summaries" / f"{file_id}.tsv",
            'survivals': result_dir / "survivals" / f"{file_id}.parquet",
            'param_dir': result_dir
        }
    
    def find_results(self, **query_params) -> pd.DataFrame:
        """
        Find Markov results matching parameters.
        
        Args:
            **query_params: Parameter key-value pairs to match
        
        Returns:
            DataFrame with matching results
        
        Example:
            >>> # Find all results with k_wrap=1.0 and p_conc=10.0
            >>> matches = storage.find_results(k_wrap=1.0, p_conc=10.0)
            >>> 
            >>> # Find all ODE solver results
            >>> matches = storage.find_results(method='ode')
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
    
    def list_all_results(self) -> pd.DataFrame:
        """
        List all indexed Markov results.
        
        Returns:
            DataFrame with all results
        """
        if not self.use_index:
            # Fallback: scan directories
            dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name != "temps"]
            return pd.DataFrame({'directory': [d.name for d in dirs]})
        return self.index.copy()
    
    def rebuild_index(self) -> pd.DataFrame:
        """
        Rebuild the index by scanning all directories in base_dir.
        
        Scans all subdirectories in base_dir, reads their parameters.json files,
        and rebuilds the markov_index.csv.
        
        Returns:
            DataFrame with rebuilt index
            
        Raises:
            ValueError: If index is not enabled
        """
        if not self.use_index:
            raise ValueError("Index not enabled. Set use_index=True in __init__")
        
        # Clear existing index``
        self.index = pd.DataFrame(columns=self.index.columns)
        
        # Scan directories
        for subdir in self.base_dir.iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name in ['temps', '.git', '__pycache__']:
                continue

            # Read parameters.json
            params_file = subdir / "parameters.json"
            if not params_file.exists():
                continue
            
            try:
                with open(params_file, 'r') as f:
                    params = json.load(f)
                
                # Update index
                self._update_index(params, subdir.name)
            except Exception as e:
                print(f"Warning: Could not read {params_file}: {e}")
        
        print(f"Rebuilt index with {len(self.index)} entries")
        return self.index
    
    
