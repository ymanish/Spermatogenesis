# IOPolyMC
Provides methods to read PolyMC output and write PolyMC input files



## Install

```
git clone git@github.com:eskoruppa/IOPolyMC.git
pip install IOPolyMC/.
```

# Use

Import package

```python
import iopolymc as io
```

## xyz files

Several methods to read and write xyz files
```python
io.read_xyz(filename: str) -> dict
```
returns a dictionary with the keys 'pos' and 'types'.


```python
io.load_xyz(filename: str,savenpy=True,loadnpy=True)
```
Same as read_xyz, except that it saves the configuration to a numpy binary upon first
read and reads from binary upon future reading of that file. Loads only from binary if the .xyz file has not been changed since the binary was created.


```python
io.read_xyz_atomtypes(filename: str) -> dict
```
Returns list of contained atom types.


```python
io.write_xyz(outfn: str,data: dict,add_extension=True,append=False) -> None
```
Writes xyz file. The argument data has to be a dictionary containing the keys 'pos' and 'types'. 'pos' needs to be a list of configurations, one for each snapshot. 

## IDB files
```python
io.read_idb(filename: str) -> dict
```
```python
io.write_idb(filename: str, idbdict: dict, decimals=3)
```

## restart files
```python
io.read_restart(filename: str) -> List[dict]
```
```python
io.write_restart(filename: str, snapshots: List[dict])
```

## state files
```python
io.read_state(filename: str) -> dict
```
```python
io.load_state(filename: str) -> dict
```
```python
io.read_spec(filename: str) -> dict
```
```python
io.scan_path(path: str, ext: str, recursive=False) -> List[str]
```

## theta files
```python
io.read_thetas(filename: str) -> np.ndarray
```
```python
io.load_thetas(filename: str) -> np.ndarray
```

## Generate pdb files
```python
io.gen_pdb(outfn: str, positions: np.ndarray,triads: np.ndarray,bpdicts: dict, sequence = None, center=True)
```
```python
io.state2pdb(statefn: str, outfn: str, snapshot: int ,bpdicts_fn=None, sequence=None, center=True)
```

## input files
```python
io.read_input(filename: str) -> dict
```
```python
io.write_input(filename: str,args: dict)
```
```python
io.querysims(path: str, recursive=False, extension='in',sims=None,select=None) -> List[dict]
```
```python
io.simfiles(infile: str,extension='in') -> List[str]
```

## Generate PolyMC configurations by interpolating points
```python
io.pts2config(pts : np.ndarray, disc_len : float, closed = False, numbp = None, translate_first = True) -> np.ndarray
```
```python
io.config2triads(config : np.ndarray) -> np.ndarray
```
```python
io.pts2xyz(outfn : str, pts : np.ndarray, disc_len : float, closed = False, numbp = None, translate_first = True, sequence = None, default_type = 'C', snapshotid = 0)
```
```python
io.pts2restart(outfn : str, pts : np.ndarray, disc_len : float, closed = False, numbp = None, translate_first = True, sequence = None, default_type = 'a', snapshotid = 0)
```

```python
io.unique_oligomers(num_bp: int,omit_equiv=True) -> List[str]
```
