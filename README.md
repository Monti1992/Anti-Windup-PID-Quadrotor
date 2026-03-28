# Anti-Windup PID for Quadrotor UAV

This repository contains the simulation code used in the paper:

"Performance Evaluation of Anti-Windup PID Controllers for Quadrotor UAVs"

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Simulation parameters
# =========================
dt = 0.001  # Time step for high accuracy
T = 10      # Total simulation time
time = np.arange(0, T, dt)

# =========================
# System parameters (Second-order system)
# =========================
a = 2.0     # Damping coefficient
b = 1.0     # Control effectiveness

# =========================
# PID parameters
# =========================
Kp = 8      # Proportional gain
Ki = 3      # Integral gain
Kd = 2      # Derivative gain

# Anti-windup gain (typically between 0.1 and 1)
Kaw = 0.5

# =========================
# Saturation limits
# =========================
u_max = 5       # Amplitude saturation limit
du_max = 10     # Rate saturation limit

# =========================
# Initial conditions
# =========================
psi = 0
psi_dot = 0
integral = 0
prev_error = 0
u = 0

# =========================
# Reference
# =========================
ref = 1.0

# =========================
# Storage for plotting
# =========================
psi_list = []
u_list = []
u_desired_list = []
error_list = []

# =========================
# Simulation loop (With Anti-Windup)
# =========================
for t in time:

    # Current error
    error = ref - psi
    error_list.append(error)

    # PID components
    P = Kp * error
    I = Ki * integral
    D = Kd * (error - prev_error) / dt

    # Desired control signal (before saturation)
    u_desired = P + I + D

    # =========================
    # Amplitude saturation
    # =========================
    u_sat = np.clip(u_desired, -u_max, u_max)

    # =========================
    # Anti-Windup (Back-calculation)
    # =========================
    integral += error * dt + Kaw * (u_sat - u_desired) * dt

    # =========================
    # Rate saturation
    # =========================
    du = (u_sat - u) / dt
    du_limited = np.clip(du, -du_max, du_max)
    u = u + du_limited * dt

    # =========================
    # System dynamics (second order system)
    # =========================
    psi_ddot = -a * psi_dot + b * u
    psi_dot += psi_ddot * dt
    psi += psi_dot * dt

    # =========================
    # Store data
    # =========================
    psi_list.append(psi)
    u_list.append(u)
    u_desired_list.append(u_desired)

    # Update previous error
    prev_error = error

# Convert lists to numpy arrays for easier calculations
psi_array = np.array(psi_list)
u_array = np.array(u_list)
u_desired_array = np.array(u_desired_list)
error_array = np.array(error_list)

# =========================
# Plot 1: System Response with Anti-Windup
# =========================
plt.figure(figsize=(10, 6))
plt.plot(time, psi_array, 'b-', linewidth=2.5, label='System Response (with Anti-Windup)')
plt.axhline(ref, color='r', linestyle='--', linewidth=2, label=f'Reference = {ref}')
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
plt.ylabel('ψ (Output)', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
plt.title('System Response with Anti-Windup PID Controller', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.xlim(0, T)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('system_response_with_antiwindup.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================
# Plot 2: Control Signal Analysis
# =========================
plt.figure(figsize=(10, 6))
plt.plot(time, u_array, 'g-', linewidth=2.5, label='Actual Control Input (u)')
plt.plot(time, u_desired_array, 'r--', linewidth=2, alpha=0.8, label='Desired Control Input (u_desired)')
plt.axhline(u_max, color='k', linestyle=':', linewidth=2, label=f'Upper Limit = {u_max}')
plt.axhline(-u_max, color='k', linestyle=':', linewidth=2, label=f'Lower Limit = -{u_max}')
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
plt.ylabel('Control Input', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
plt.title('Control Signal Analysis with Anti-Windup', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.xlim(0, T)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('control_signal_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================
# Plot 3: Tracking Error
# =========================
plt.figure(figsize=(10, 6))
plt.plot(time, error_array, 'm-', linewidth=2.5, label='Tracking Error')
plt.axhline(0, color='k', linestyle='-', linewidth=1.5)
plt.fill_between(time, -0.02, 0.02, alpha=0.2, color='gray', label='2% Band')
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
plt.ylabel('Error', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
plt.title('Tracking Error Analysis', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.xlim(0, T)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('tracking_error.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================
# Performance metrics (With Anti-Windup)
# =========================
print("=" * 60)
print("PERFORMANCE METRICS (WITH ANTI-WINDUP)")
print("=" * 60)

# Steady-state error (last 10% of simulation)
steady_state_start = int(0.9 * len(error_array))
steady_state_error = np.mean(np.abs(error_array[steady_state_start:]))
print(f"Steady-state absolute error: {steady_state_error:.6f}")

# Maximum overshoot
max_psi = np.max(psi_array)
if max_psi > ref:
    overshoot = ((max_psi - ref) / ref) * 100
else:
    overshoot = 0
print(f"Maximum overshoot: {overshoot:.2f}%")

# Settling time (within 2% of reference)
settling_threshold = 0.02 * abs(ref)
settling_time = T
for i in range(len(psi_array)):
    if abs(psi_array[i] - ref) < settling_threshold:
        remaining = psi_array[i:]
        if np.all(np.abs(remaining - ref) < settling_threshold):
            settling_time = time[i]
            break
print(f"Settling time (2%): {settling_time:.3f} s")

# Control effort
rms_control = np.sqrt(np.mean(np.square(u_array)))
print(f"RMS control effort: {rms_control:.3f}")

# Check for integrator windup
print(f"Max integrator value: {abs(integral):.3f}")

print("=" * 60)

# =========================
# Simulation without Anti-Windup (for comparison)
# =========================
print("\nRunning comparison simulation without Anti-Windup...")

# Reset for simulation without anti-windup
psi_no_aw = 0
psi_dot_no_aw = 0
integral_no_aw = 0
prev_error_no_aw = 0
u_no_aw = 0
psi_list_no_aw = []
u_list_no_aw = []
error_list_no_aw = []

for t in time:
    error = ref - psi_no_aw
    error_list_no_aw.append(error)

    P = Kp * error
    I = Ki * integral_no_aw
    D = Kd * (error - prev_error_no_aw) / dt

    u_desired = P + I + D
    u_sat = np.clip(u_desired, -u_max, u_max)

    # No anti-windup
    integral_no_aw += error * dt

    # Rate saturation
    du = (u_sat - u_no_aw) / dt
    du_limited = np.clip(du, -du_max, du_max)
    u_no_aw = u_no_aw + du_limited * dt

    psi_ddot_no_aw = -a * psi_dot_no_aw + b * u_no_aw
    psi_dot_no_aw += psi_ddot_no_aw * dt
    psi_no_aw += psi_dot_no_aw * dt

    psi_list_no_aw.append(psi_no_aw)
    u_list_no_aw.append(u_no_aw)

    prev_error_no_aw = error

# Convert to numpy arrays
psi_array_no_aw = np.array(psi_list_no_aw)
u_array_no_aw = np.array(u_list_no_aw)
error_array_no_aw = np.array(error_list_no_aw)

# =========================
# Performance metrics (Without Anti-Windup)
# =========================
print("\n" + "=" * 60)
print("PERFORMANCE METRICS (WITHOUT ANTI-WINDUP)")
print("=" * 60)

# Steady-state error (last 10% of simulation)
steady_state_error_no_aw = np.mean(np.abs(error_array_no_aw[steady_state_start:]))
print(f"Steady-state absolute error: {steady_state_error_no_aw:.6f}")

# Maximum overshoot
max_psi_no_aw = np.max(psi_array_no_aw)
if max_psi_no_aw > ref:
    overshoot_no_aw = ((max_psi_no_aw - ref) / ref) * 100
else:
    overshoot_no_aw = 0
print(f"Maximum overshoot: {overshoot_no_aw:.2f}%")

# Settling time (within 2% of reference)
settling_time_no_aw = T
for i in range(len(psi_array_no_aw)):
    if abs(psi_array_no_aw[i] - ref) < settling_threshold:
        remaining = psi_array_no_aw[i:]
        if np.all(np.abs(remaining - ref) < settling_threshold):
            settling_time_no_aw = time[i]
            break
print(f"Settling time (2%): {settling_time_no_aw:.3f} s")

# Control effort
rms_control_no_aw = np.sqrt(np.mean(np.square(u_array_no_aw)))
print(f"RMS control effort: {rms_control_no_aw:.3f}")

# Check for integrator windup
print(f"Max integrator value: {abs(integral_no_aw):.3f}")

print("=" * 60)

# =========================
# Plot 4: Comparison - System Response
# =========================
plt.figure(figsize=(10, 6))
plt.plot(time, psi_array, 'b-', linewidth=2.5, label='With Anti-Windup')
plt.plot(time, psi_array_no_aw, 'r--', linewidth=2.5, label='Without Anti-Windup')
plt.axhline(ref, color='k', linestyle=':', linewidth=2, label='Reference')
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
plt.ylabel('ψ (Output)', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
plt.title('Comparative Analysis: System Response', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.xlim(0, T)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('comparison_system_response.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================
# Plot 5: Comparison - Control Input
# =========================
plt.figure(figsize=(10, 6))
plt.plot(time, u_array, 'b-', linewidth=2.5, label='With Anti-Windup')
plt.plot(time, u_array_no_aw, 'r--', linewidth=2.5, label='Without Anti-Windup')
plt.axhline(u_max, color='k', linestyle=':', linewidth=2, label=f'Upper Limit = {u_max}')
plt.axhline(-u_max, color='k', linestyle=':', linewidth=2, label=f'Lower Limit = -{u_max}')
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
plt.ylabel('Control Input', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
plt.title('Comparative Analysis: Control Input', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.xlim(0, T)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('comparison_control_input.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================
# Plot 6: Detailed Analysis - Overshoot Comparison
# =========================
plt.figure(figsize=(10, 6))

# Zoomed view around the peak
zoom_start = 0
zoom_end = 3
zoom_indices = np.where((time >= zoom_start) & (time <= zoom_end))

plt.plot(time[zoom_indices], psi_array[zoom_indices], 'b-', linewidth=2.5, label='With Anti-Windup')
plt.plot(time[zoom_indices], psi_array_no_aw[zoom_indices], 'r--', linewidth=2.5, label='Without Anti-Windup')
plt.axhline(ref, color='k', linestyle=':', linewidth=2, label='Reference')

# Highlight overshoot regions
plt.fill_between(time[zoom_indices], ref, psi_array[zoom_indices],
                 where=(psi_array[zoom_indices] > ref), alpha=0.3, color='blue', label='Overshoot (With AW)')
plt.fill_between(time[zoom_indices], ref, psi_array_no_aw[zoom_indices],
                 where=(psi_array_no_aw[zoom_indices] > ref), alpha=0.3, color='red', label='Overshoot (Without AW)')

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
plt.ylabel('ψ (Output)', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
plt.title('Overshoot Analysis: With vs Without Anti-Windup', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.xlim(zoom_start, zoom_end)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('overshoot_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================
# Plot 7: Integrator State Analysis
# =========================
# Run simulation again to store integrator values
integral_list_with_aw = []
integral_list_without_aw = []

# With Anti-Windup
integral = 0
psi = 0
psi_dot = 0
prev_error = 0
u = 0
for t in time:
    error = ref - psi
    P = Kp * error
    I = Ki * integral
    D = Kd * (error - prev_error) / dt
    u_desired = P + I + D
    u_sat = np.clip(u_desired, -u_max, u_max)
    integral += error * dt + Kaw * (u_sat - u_desired) * dt
    integral_list_with_aw.append(integral)
    du = (u_sat - u) / dt
    du_limited = np.clip(du, -du_max, du_max)
    u = u + du_limited * dt
    psi_ddot = -a * psi_dot + b * u
    psi_dot += psi_ddot * dt
    psi += psi_dot * dt
    prev_error = error

# Without Anti-Windup
integral = 0
psi = 0
psi_dot = 0
prev_error = 0
u = 0
for t in time:
    error = ref - psi
    P = Kp * error
    I = Ki * integral
    D = Kd * (error - prev_error) / dt
    u_desired = P + I + D
    u_sat = np.clip(u_desired, -u_max, u_max)
    integral += error * dt
    integral_list_without_aw.append(integral)
    du = (u_sat - u) / dt
    du_limited = np.clip(du, -du_max, du_max)
    u = u + du_limited * dt
    psi_ddot = -a * psi_dot + b * u
    psi_dot += psi_ddot * dt
    psi += psi_dot * dt
    prev_error = error

plt.figure(figsize=(10, 6))
plt.plot(time, integral_list_with_aw, 'b-', linewidth=2.5, label='With Anti-Windup')
plt.plot(time, integral_list_without_aw, 'r--', linewidth=2.5, label='Without Anti-Windup')
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
plt.ylabel('Integrator State', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
plt.title('Integrator Windup Analysis', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.xlim(0, T)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('integrator_windup_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================
# Comparison Metrics
# =========================
print("\n" + "=" * 60)
print("COMPARISON METRICS")
print("=" * 60)

print(f"Overshoot without Anti-Windup: {overshoot_no_aw:.2f}%")
print(f"Overshoot with Anti-Windup: {overshoot:.2f}%")
if overshoot_no_aw != 0:
    improvement_overshoot = ((overshoot_no_aw - overshoot) / overshoot_no_aw) * 100
    print(f"Overshoot reduction: {improvement_overshoot:.1f}%")
else:
    print("Overshoot reduction: N/A")

print(f"\nSettling time without Anti-Windup: {settling_time_no_aw:.3f} s")
print(f"Settling time with Anti-Windup: {settling_time:.3f} s")
if settling_time_no_aw != 0:
    improvement_settling = ((settling_time_no_aw - settling_time) / settling_time_no_aw) * 100
    print(f"Settling time improvement: {improvement_settling:.1f}%")
else:
    print("Settling time improvement: N/A")

print(f"\nRMS control effort without Anti-Windup: {rms_control_no_aw:.3f}")
print(f"RMS control effort with Anti-Windup: {rms_control:.3f}")
if rms_control_no_aw != 0:
    improvement_rms = ((rms_control_no_aw - rms_control) / rms_control_no_aw) * 100
    print(f"Control effort reduction: {improvement_rms:.1f}%")
else:
    print("Control effort reduction: N/A")

print(f"\nSteady-state error without Anti-Windup: {steady_state_error_no_aw:.6f}")
print(f"Steady-state error with Anti-Windup: {steady_state_error:.6f}")

print("=" * 60)

# =========================
# Performance Summary Table
# =========================
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY TABLE")
print("=" * 60)
print(f"{'Metric':<35} {'Without AW':<15} {'With AW':<15} {'Improvement':<15}")
print("-" * 80)

# Overshoot
if overshoot_no_aw != 0:
    improvement_overshoot = ((overshoot_no_aw - overshoot) / overshoot_no_aw) * 100
else:
    improvement_overshoot = 0
print(f"{'Overshoot (%)':<35} {overshoot_no_aw:<15.2f} {overshoot:<15.2f} {improvement_overshoot:<15.1f}%")

# Settling Time
improvement_settling = ((settling_time_no_aw - settling_time) / settling_time_no_aw) * 100
print(f"{'Settling Time (s)':<35} {settling_time_no_aw:<15.3f} {settling_time:<15.3f} {improvement_settling:<15.1f}%")

# RMS Control Effort
improvement_rms = ((rms_control_no_aw - rms_control) / rms_control_no_aw) * 100
print(f"{'RMS Control Effort':<35} {rms_control_no_aw:<15.3f} {rms_control:<15.3f} {improvement_rms:<15.1f}%")

# Max Integrator Value
max_int_without = max(integral_list_without_aw)
max_int_with = max(integral_list_with_aw)
if max_int_without != 0:
    improvement_int = ((max_int_without - max_int_with) / max_int_without) * 100
else:
    improvement_int = 0
print(f"{'Max Integrator Value':<35} {max_int_without:<15.3f} {max_int_with:<15.3f} {improvement_int:<15.1f}%")

# Steady-State Error
print(f"{'Steady-State Error':<35} {steady_state_error_no_aw:<15.6f} {steady_state_error:<15.6f} {'N/A':<15}")

print("=" * 60)

# =========================
# Additional Analysis: Rise Time
# =========================
print("\n" + "=" * 60)
print("ADDITIONAL ANALYSIS: RISE TIME")
print("=" * 60)

def calculate_rise_time(response_array, time_array, ref_value, start_pct=0.1, end_pct=0.9):
    """Calculate rise time from start_pct to end_pct of reference value"""
    target_start = start_pct * ref_value
    target_end = end_pct * ref_value

    t_start = None
    t_end = None

    for i, val in enumerate(response_array):
        if t_start is None and val >= target_start:
            t_start = time_array[i]
        if t_end is None and val >= target_end:
            t_end = time_array[i]
            break

    if t_start is not None and t_end is not None:
        return t_end - t_start
    else:
        return None

rise_time_aw = calculate_rise_time(psi_array, time, ref)
rise_time_no_aw = calculate_rise_time(psi_array_no_aw, time, ref)

if rise_time_aw is not None:
    print(f"Rise time (10-90%) with Anti-Windup: {rise_time_aw:.3f} s")
if rise_time_no_aw is not None:
    print(f"Rise time (10-90%) without Anti-Windup: {rise_time_no_aw:.3f} s")

print("=" * 60)
print("\nSimulation completed successfully!")
print("All figures have been saved as PNG files with 300 DPI resolution.")
