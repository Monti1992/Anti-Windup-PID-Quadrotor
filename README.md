This project presents a comparative evaluation of classical PID and anti-windup PID controllers for quadrotor attitude control under actuator saturation constraints. The study focuses on mitigating the integral windup phenomenon, which degrades control performance when actuator limits are reached
Key Contributions
Implementation of a quadrotor dynamic model
Design of a baseline PID controller
Integration of an anti-windup mechanism (back-calculation / clamping)
Comparative performance analysis under saturation limits
Demonstration of improved recovery and reduced overshoot
Problem Statement
In quadrotor systems, actuator saturation is inevitable due to physical motor limits. When combined with integral action in PID controllers, this leads to integrator windup, causing performance degradation, large overshoots, and delayed system recovery.
Methodology
Nonlinear quadrotor model
PID controller design
Anti-windup scheme implementation
Simulation under saturation limits 
Performance metrics evaluation
