# Balanced UCI-HAR View Resampled to 20Hz with gravity

The data used was the samples with gravity by autors.

This view contain only the train and test files for [UCI-HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#) (70% samples train and 30% test). The data was spllited into 3s windows and was resampled to 20Hz using the [FFT method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html#scipy.signal.resample). The accelerometer meansure is in m/s² and with gravity.

## Activity codes
- 1: walking (403 train, 38 validation, 127 test) 
- 2: walking upstairs (390 train, 42 validation, 136 test) 
- 3: walking downstairs (399 train, 43 validation, 126 test) 
- 4: sitting (378 train, 46 validation, 144 test) 
- 5: standing (382 train, 48 validation, 138 test) 
 

## Standartized activity codes
- 0: sit (378 train, 46 validation, 144 test) 
- 1: stand (382 train, 48 validation, 138 test) 
- 2: walk (403 train, 38 validation, 127 test) 
- 3: stair up (390 train, 42 validation, 136 test) 
- 4: stair down (399 train, 43 validation, 126 test) 
      


