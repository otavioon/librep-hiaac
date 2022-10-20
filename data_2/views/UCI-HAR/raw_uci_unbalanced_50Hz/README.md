# Unbalanced UCI-HAR View to 50Hz with gravity

The data used was the samples with gravity by autors. 
+
This view contain only the train and test files for [UCI-HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#) (70% samples train and 30% test). The data was spllited into 3s windows and was resampled to 20Hz using the [FFT method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html#scipy.signal.resample). The accelerometer meansure is in m/s² and with gravity.

## Activity codes
- 1: walking (85 train, 0 validation, 36 test) 
- 2: walking upstairs (17 train, 0 validation, 14 test) 
- 3: walking downstairs (8 train, 0 validation, 5 test) 
- 4: sitting (90 train, 0 validation, 35 test) 
- 5: standing (94 train, 0 validation, 39 test) 
 

## Standartized activity codes
- 0: sit (90 train, 0 validation, 35 test) 
- 1: stand (94 train, 0 validation, 39 test) 
- 2: walk (85 train, 0 validation, 36 test) 
- 3: stair up (17 train, 0 validation, 14 test) 
- 4: stair down (8 train, 0 validation, 5 test) 
      


