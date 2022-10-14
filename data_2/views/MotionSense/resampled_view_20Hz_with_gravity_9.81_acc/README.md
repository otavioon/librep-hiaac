# Resampled to 20Hz MotionSense View With Gravity

This view contains train, validation and test subsets in the following proportions:
- Train: 70% of samples
- Validation: 10% of samples
- Test: 20% of samples

After splits, the datasets were balanced in relation to the activity code column, that is, each subset have the same number of activitiy samples.

## Activities:
- dws: 0 (569 train, 101 validation, 170 test)
- ups: 1 (569 train, 101 validation, 170 test)
- sit: 2 (569 train, 101 validation, 170 test)
- std: 3 (569 train, 101 validation, 170 test)
- wlk: 4 (569 train, 101 validation, 170 test)
- jog: 5 (569 train, 101 validation, 170 test)

## Users
- 16 users train dataset: 1 (218 samples), 2 (219 samples), 5 (185 samples), 6 (218 samples), 8 (233 samples), 9 (202 samples), 10 (218 samples), 11 (211 samples), 12 (197 samples), 13 (183 samples), 15 (208 samples), 16 (246 samples), 17 (209 samples), 21 (254 samples), 22 (200 samples), 23 (213 samples).
- 3 users validation dataset: 4 (190 samples), 7 (211 samples), 20 (205 samples).
- 5 users test dataset: 3 (222 samples), 14 (183 samples), 18 (223 samples), 19 (233 samples), 24 (159 samples).

**NOTE**: Each subset contain samples from distinct users, that is, samples of one user belongs exclusivelly to one of three subsets.

