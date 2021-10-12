# Effects of Finite Volume Reconstruction Scheme on 1D Viscous Burgers Turbulence

This project is in partial fulfillment of course requirements for [MAE 6263, Computational Fluid Dynamics](http://catalog.okstate.edu/courses/mae/)
at Oklahoma State University.

The code is contained in `cp2_burgers.py`, and the report documenting the problem description, numerical methods, and results, is contained in `braun2021burgersturbulence.pdf`.

> **Note:** To run `cp2_burgers.py`, you must have Python3.9 installed. This is because the case configuration dictionaries are constructed with [dictionary Union Operators](https://www.python.org/dev/peps/pep-0584/).

## Running the Code

To run the code, first configure which cases you'd like to run by modifying the `if __name__ == "__main__":` block at the bottom of `cp2_burgers.py`, install the requirements, and then run the code:

```bash
$ python3.9 -m pip install --user -r requirements.txt
$ python3.9 cp2_burgers.py
```

If you want to run this in the background, you can use `nohup`:

```bash
$ nohup python3.9 cp2_burgers.py &

# View process id:
$ ps -aef | grep cp2_burgers_1d

# Kill the process
$ kill <process-id>
```

The case configuration is organized by comparisons. For example, if you'd like to run several cases and compare them on the same plots, that would be one comparison. The collection of comparisons that can be run is in the aptly named `comparisons` dictionary:

```python
# For example:
comparisons = {
    # ns = no. of samples for turbulence results averaging
    # nx = no. of elements in the x direction
    "WENO_3_Resolution_Comparison": [
        WENO_3 | {"nx": 2**9, "ns": 2**6},  # nx=512, ns=32
        WENO_3 | {"nx": 2**10, "ns": 2**6},
        WENO_3 | {"nx": 2**11, "ns": 2**6},
        WENO_3 | {"nx": 2**12, "ns": 2**6},
        WENO_3 | {"nx": 2**13, "ns": 2**6},
        WENO_3 | {"nx": 2**14, "ns": 2**6},
        # nx=2**15 is considered our "DNS" results
    ],
    "WENO_5_Resolution_Comparison": [
        WENO_5 | {"nx": 2**9, "ns": 2**6},  # nx=512, ns=32
        WENO_5 | {"nx": 2**10, "ns": 2**6},
        WENO_5 | {"nx": 2**11, "ns": 2**6},
        WENO_5 | {"nx": 2**12, "ns": 2**6},
        WENO_5 | {"nx": 2**13, "ns": 2**6},
        WENO_5 | {"nx": 2**14, "ns": 2**6},
        # nx=2**15 is considered our "DNS" results
    ],
}
```

The `comparisons_to_run` list tells the code which of the comparisons you've defined it should run. Note, you can easily comment out any comparisons you may not want to run this time:

```python
# For example:
comparisons_to_run = [
    "MUSCL_Q_Limiters_Comparison",
    # "MUSCL_KT_Limiters_Comparison",
    "MUSCL_CS_Limiters_Comparison",
    # "MUSCL_3rd_Limiters_Comparison",
    # "MUSCL_Upwind_Limiters_Comparison",
    # "MUSCL_Fromm_Limiters_Comparison",
    # "MUSCL_Type_Comparison",
    # "MUSCL_Superbee_Type_Comparison",
    # "MUSCL_Q_Superbee_Viscosity_Comparison",
    # "WENO_3_Resolution_Comparison",
    # "WENO_5_Resolution_Comparison",
    # "WENO_3_Viscosity_Comparison",
    # "WENO_5_Viscosity_Comparison",
]
```
