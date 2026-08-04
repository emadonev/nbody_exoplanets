# N body LTT 1445 and GJ 667 plan

- [x] gather the data for LTT 1445 and HJ 667
  - [x] stellar data
  - [x] orbits for stars
  - [x] planetary data
- [x] determine all ranges of parameter space for GJ 667
- [x] Check code for conversion between orbital elements and cartesian
- [x] write the data section of the paper
- [x] create experiment file adopting coherent 2 planet GJ 667 and 3 planet LTT 1445
  - [x] For GJ 667 - everything included
  - [x] For LTT 1445
    - [x] with 3 planets - everything
    - [x] with 2 planets - || -
- [x] Modify run.py to use the new JSONs for reading and processing data
- [x] make code for a 2 body WH integrator

---

- [ ] Validate the WH integrator with multiple tests
  - [x] test with energy and angular momentum checks
  - [ ] test with simple propagation of both the binary and triple system - testing with GJ 667 and LTT 1445 - currently GJ works, but LTT 1445 is a bit wonky, most likely due to planet ejection. Test with different time steps, check that orbital elements are calculated correctly
- [ ] **run all experiments!!!!**
- [ ] add code for propagating luminosity uncertainties in post processing
- [ ] double check and adjust code for equilibrium temperature calculation
- [ ] **one-box temperature anomaly model over several thermal-response scenarios**
  - [ ] Research simple models for temperature
  - [ ] implement the simple temperature model
- [ ] make tests and diagnostics for stability, irradiation variability, HZ residence and thermal-response metrics
- [ ] include the results of UV and X ray measurements as a diagnostic later
- [ ] **report conditional admissible fractions with uncertainty intervals**
- [ ] extend representative configurations to the maximum integration 
- [ ] make code for more representative graphs
  - [ ] changes in a, e and i
  - [ ] changes in temperature over time
- [ ] write results section
- [ ] discussion section
- [ ] write conclusion



