# Chassis measurements (real robot, 2026-11-14)

All runs through `chassis_test.py` on B70, inside the official `openpi-xpu-client` container.

## Command axes

`/api/chassis/velocity` `data=[vx, vy, wz]`:

| command | developer doc says | measured |
|---|---|---|
| +vx | forward | **right** |
| +vy | left | **forward** |
| +wz | CCW | CCW |

To drive in the usual (forward, left, CCW) frame send `[-left, forward, wz]`.

## Motor speeds (steady state, rad/s)

| command | chassis_joint1 | chassis_joint2 | chassis_joint3 |
|---|---|---|---|
| vx +0.05 m/s (3 runs) | -0.263 / -0.247 / -0.259 | -0.255 / -0.248 / -0.256 | 0.509 / 0.505 / 0.500 |
| vx -0.05 m/s (2 runs) | 0.261 / 0.250 | 0.258 / 0.250 | -0.504 / -0.500 |
| vy +0.05 m/s (2 runs) | 0.438 / 0.440 | -0.441 / -0.421 | -0.002 / -0.001 |
| vy -0.05 m/s (2 runs) | -0.423 / -0.435 | 0.437 / 0.420 | 0.005 / 0.006 |
| wz +0.2 rad/s | 0.825 | 0.848 | 0.827 |
| wz -0.2 rad/s | -0.842 | -0.857 | -0.845 |

Motor speed per unit command (J) and its inverse, used by `teleop.py` for odometry:

```
J (rad/s per [vx m/s, vy m/s, wz rad/s])     J_INV
[[-5.1183   8.68    4.1675]                 [[-0.032365 -0.033224  0.066149]
 [-5.07    -8.595   4.2625]                  [ 0.058231 -0.057545  0.000625]
 [10.0667  -0.07    4.18  ]]                 [ 0.078921  0.07905   0.07994 ]]
```

Seen as a three-wheel omni base this is self-consistent: wheel radius about 9.8 cm, wheel distance from
the centre about 41 cm. These are command-to-motor ratios; distances have **not** been checked with a
tape measure yet.

## Other observations

- Commands flow `/api/chassis/velocity -> /mux/chassis/cmd -> /chassis_target_vel -> /chassis/joint_cmd`;
  `/chassis_target_vel` ramps at 0.3 m/s^2.
- Raw `/chassis/joint_states` velocity noise at rest: median ~0.05 rad/s, peaks ~0.3 rad/s. Positions are
  steady, so motion is judged from position changes or 0.1 s averages.
- Stopping with a zero command: wheels settle 0.17-0.39 s after the last non-zero command.
- Once, all motors ignored commands (target speeds reached the motors, actual speed and effort ~0, no
  error code); it was a robot-side issue fixed on site.
- `/api/current_mode` must show `meta_mode: 1` (API) and `enable_chassis: true`.

## Arms (2026-11-14)

- Arms must be in joint control mode for `/api/{arm}_arm/joint_cmd`; in end-effector mode the topic does
  not exist on the robot side (only `/api/{arm}_arm/target_pose` is subscribed). The mode change needs
  Save config -> Apply -> reboot; until then `/api/current_mode` keeps reporting 0.
- `arm/arm.sh raise` from the hanging pose (joint2 about +/-1.57) to the zero.py pose: FSM PAUSE -> READY
  while holding, 8 s min-jerk move, max error to target 0.011 rad.
- After `fsm/enable 0.0` (PAUSE) the arms held the raised pose: no measurable sag over 5 s.
