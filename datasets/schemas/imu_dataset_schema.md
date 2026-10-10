\# SmartCap IMU Dataset Schema



\## Unit conventions



| Field | Unit | Description |

|---|---|---|

| ax\_g, ay\_g, az\_g | g | Accelerometer axes |

| gx\_dps, gy\_dps, gz\_dps | degrees/s | Gyroscope axes |

| pitch\_deg | degrees | Relative head pitch after calibration |

| roll\_deg | degrees | Relative head roll after calibration |

| pitch\_velocity\_dps | degrees/s | Smoothed pitch angular velocity |

| head\_down\_duration\_ms | ms | Time held in head-down state |

| nodding\_score | 0–1 | Decaying motion-pattern confidence |

| motion\_energy | normalized | Relative motion intensity |

| imu\_quality | 0–1 | Signal-health quality score |



\## Labels



```text

still

normal

look\_down

nod

rapid\_turn

```



\## File naming



```text

PXX\_label\_optional-tag\_YYYYMMDD\_HHMMSS.csv

```



Example:



```text

P01\_nod\_trial1\_20261010\_224500.csv

```



\## Required metadata



```text

participant\_id

label

device\_timestamp\_ms

received\_timestamp\_ms

hardware\_version

firmware\_version

calibration\_status

```

