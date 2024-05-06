# surrogateopt_openmdao

Surrogate optimization driver for OpenMDAO.

Embeds SMT's and pySOT's optimization strategies.

## Todo

- [ ] Scipy's GP seems not to work with pySOT. Maybe pySOT should update something there.

## Installation

For the `SRBF_Failsafe` strategy, please install [this](https://github.com/christianhauschel/pySOT) customized version of pySOT.

## Remarks

- pySOT's checkpoint controller does not work with MPI and mpi4py installed.
- Does not support pySOT's parallel or async modi.
