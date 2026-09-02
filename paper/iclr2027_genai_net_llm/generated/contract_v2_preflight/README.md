# Contract-v2 static preflight

Overall status: **PASS**

No model calls were made. Prompt approval and the dynamic sentinel remain blocking gates.

| Task | Fixed IDs | Null ID | Allowed | Negative test | Status |
|---|---:|---:|---:|---|---|
| rpa | [1, 28] | 0 | 88 | reaction ID 1 is forbidden for this task. | pass |
| logic | [1, 2, 3, 4] | 0 | 416 | reaction ID 1 is forbidden for this task. | pass |
| classifier | [] | 0 | 30 | reaction ID 0 is forbidden for this task. | pass |
| dose_hill | [1] | 0 | 89 | reaction ID 1 is forbidden for this task. | pass |
| dose_ultrasensitive | [1] | 0 | 89 | reaction ID 1 is forbidden for this task. | pass |
| dose_biphasic | [1] | 0 | 89 | reaction ID 1 is forbidden for this task. | pass |
| oscillator_mean | [] | 0 | 90 | reaction ID 0 is forbidden for this task. | pass |
| oscillator_frequency | [1] | 0 | 89 | reaction ID 1 is forbidden for this task. | pass |
