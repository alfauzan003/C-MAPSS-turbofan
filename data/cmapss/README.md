# NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001)

Source: NASA PHM Data Repository ("Turbofan Engine Degradation Simulation Data Set",
A. Saxena and K. Goebel, 2008).

License: Public domain (US Government work).

## Files

- `train_FD001.txt` — full run-to-failure traces for ~100 engines.
- `test_FD001.txt` — partial traces (truncated before failure).
- `RUL_FD001.txt` — true RUL value at the last cycle of each test engine, in order.

## Format (whitespace-separated, no header)

| col | name |
|----:|------|
| 1   | engine_id (int) |
| 2   | cycle (int, monotonically increasing per engine) |
| 3   | op_setting_1 (float) |
| 4   | op_setting_2 (float) |
| 5   | op_setting_3 (float) |
| 6–26| sensor_1 .. sensor_21 (float) |

For training, RUL target is derived from the train file:
RUL(engine, cycle) = max_cycle(engine) - cycle.
